#!/usr/bin/env python3
"""Pure NumPy inference and deployment-state handling for the v2 assist policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


INPUT_SIZE = 150
OUTPUT_SIZE = 2
HISTORY_LENGTH = 25
ASSIST_TORQUE_LIMIT_NM = 8.0
ASSIST_TORQUE_RATE_NM_S = 80.0
POLICY_DT_S = 0.01
MAX_TORQUE_DELTA_NM = ASSIST_TORQUE_RATE_NM_S * POLICY_DT_S
COMMAND_SPEED_DEADZONE_M_S = 0.2
COMMAND_SPEED_FULL_M_S = 0.7


def command_speed_gate(forward_command_m_s: float) -> np.float32:
    """Training-matched smoothstep gate based on absolute commanded vx."""
    if not np.isfinite(forward_command_m_s):
        return np.float32(0.0)
    phase = np.clip(
        (abs(float(forward_command_m_s)) - COMMAND_SPEED_DEADZONE_M_S)
        / (COMMAND_SPEED_FULL_M_S - COMMAND_SPEED_DEADZONE_M_S),
        0.0,
        1.0,
    )
    return np.float32(phase * phase * (3.0 - 2.0 * phase))


def elu_inplace(values: np.ndarray) -> np.ndarray:
    negative = values < 0.0
    values[negative] = np.expm1(values[negative])
    return values


class V2AssistHistory:
    """Fixed 25-frame history with the exact training/deployment observation order."""

    def __init__(self) -> None:
        self.position = np.zeros((HISTORY_LENGTH, 2), dtype=np.float32)
        self.velocity = np.zeros((HISTORY_LENGTH, 2), dtype=np.float32)
        self.torque = np.zeros((HISTORY_LENGTH, 2), dtype=np.float32)
        self.next_index = 0

    def reset(self, position_rad: np.ndarray, velocity_rad_s: np.ndarray) -> None:
        position = np.asarray(position_rad, dtype=np.float32)
        velocity = np.asarray(velocity_rad_s, dtype=np.float32)
        if position.shape != (2,) or velocity.shape != (2,):
            raise ValueError("Position and velocity must each have shape (2,)")
        self.position[:] = position
        self.velocity[:] = velocity
        self.torque.fill(0.0)
        self.next_index = 0

    def append(
        self,
        position_rad: np.ndarray,
        velocity_rad_s: np.ndarray,
        previous_smoothed_torque_nm: np.ndarray,
    ) -> None:
        samples = tuple(
            np.asarray(value, dtype=np.float32)
            for value in (position_rad, velocity_rad_s, previous_smoothed_torque_nm)
        )
        if any(value.shape != (2,) for value in samples):
            raise ValueError("Every history sample must have shape (2,)")
        self.position[self.next_index], self.velocity[self.next_index], self.torque[
            self.next_index
        ] = samples
        self.next_index = (self.next_index + 1) % HISTORY_LENGTH

    def observation(self) -> np.ndarray:
        order = (np.arange(HISTORY_LENGTH) + self.next_index) % HISTORY_LENGTH
        return np.concatenate(
            (self.position[order].reshape(-1), self.velocity[order].reshape(-1), self.torque[order].reshape(-1))
        ).astype(np.float32, copy=False)


class NumpyAssistPolicy:
    """Float32 v2 MLP with no PyTorch dependency."""

    def __init__(self, weights_path: str | Path):
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"NumPy policy weights not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            self.mean = np.array(data["obs_mean"], dtype=np.float32, copy=True)
            self.std = np.array(data["obs_std"], dtype=np.float32, copy=True)
            self.eps = np.float32(data["normalizer_eps"])
            self.weights = tuple(
                np.array(data[f"weight_{i}"], dtype=np.float32, copy=True) for i in range(4)
            )
            self.biases = tuple(
                np.array(data[f"bias_{i}"], dtype=np.float32, copy=True) for i in range(4)
            )
        self._validate()

    def _validate(self) -> None:
        expected_weights = ((256, 150), (64, 256), (16, 64), (2, 16))
        expected_biases = ((256,), (64,), (16,), (2,))
        if self.mean.shape != (INPUT_SIZE,) or self.std.shape != (INPUT_SIZE,):
            raise ValueError("Expected 150-element observation normalization arrays")
        for index, (actual, expected) in enumerate(zip(self.weights, expected_weights)):
            if actual.shape != expected:
                raise ValueError(f"weight_{index} has shape {actual.shape}, expected {expected}")
        for index, (actual, expected) in enumerate(zip(self.biases, expected_biases)):
            if actual.shape != expected:
                raise ValueError(f"bias_{index} has shape {actual.shape}, expected {expected}")
        if np.any(self.std <= 0.0):
            raise ValueError("Observation standard deviations must be positive")

    def forward(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32)
        if x.ndim not in (1, 2) or x.shape[-1] != INPUT_SIZE:
            raise ValueError(f"Expected observation shape (150,) or (N,150), got {x.shape}")
        x = (x - self.mean) / (self.std + self.eps)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = elu_inplace(x @ weight.T + bias)
        return x @ self.weights[-1].T + self.biases[-1]

    __call__ = forward

    def target_torque(self, observation: np.ndarray, forward_command_m_s: float) -> np.ndarray:
        action = np.clip(self.forward(observation), -1.0, 1.0)
        return np.float32(ASSIST_TORQUE_LIMIT_NM) * command_speed_gate(forward_command_m_s) * action

    @staticmethod
    def slew_limit(previous_torque_nm: np.ndarray, target_torque_nm: np.ndarray) -> np.ndarray:
        previous = np.asarray(previous_torque_nm, dtype=np.float32)
        target = np.asarray(target_torque_nm, dtype=np.float32)
        return previous + np.clip(
            target - previous, -MAX_TORQUE_DELTA_NM, MAX_TORQUE_DELTA_NM
        )

    def step(
        self,
        observation: np.ndarray,
        previous_torque_nm: np.ndarray,
        forward_command_m_s: float,
    ) -> np.ndarray:
        return self.slew_limit(
            previous_torque_nm,
            self.target_torque(observation, forward_command_m_s),
        )


class V2AssistController:
    """Stateful 100 Hz deployment controller with a final-only output scale."""

    def __init__(self, policy: NumpyAssistPolicy, output_scale: float = 1.0):
        self.policy = policy
        self.history = V2AssistHistory()
        self.previous_smoothed_torque_nm = np.zeros(2, dtype=np.float32)
        self.first_policy_step = True
        self.output_scale = 1.0
        self.set_output_scale(output_scale)

    def set_output_scale(self, output_scale: float) -> None:
        scale = float(output_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("output_scale must be a finite non-negative number")
        self.output_scale = scale

    def reset(self, position_rad: np.ndarray, velocity_rad_s: np.ndarray) -> None:
        self.history.reset(position_rad, velocity_rad_s)
        self.previous_smoothed_torque_nm.fill(0.0)
        self.first_policy_step = True

    def step(
        self,
        position_rad: np.ndarray,
        velocity_rad_s: np.ndarray,
        forward_command_m_s: float,
    ) -> np.ndarray:
        if self.first_policy_step:
            self.first_policy_step = False
        else:
            self.history.append(
                position_rad,
                velocity_rad_s,
                self.previous_smoothed_torque_nm,
            )
        observation = self.history.observation()
        smoothed = self.policy.step(
            observation,
            self.previous_smoothed_torque_nm,
            forward_command_m_s,
        )
        self.previous_smoothed_torque_nm[:] = smoothed
        return np.clip(
            self.output_scale * smoothed,
            -ASSIST_TORQUE_LIMIT_NM,
            ASSIST_TORQUE_LIMIT_NM,
        ).astype(np.float32, copy=False)

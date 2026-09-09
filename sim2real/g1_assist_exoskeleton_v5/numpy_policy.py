#!/usr/bin/env python3
"""Pure NumPy inference and deployment-state handling for the v5 assist policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


INPUT_SIZE = 450
OUTPUT_SIZE = 2
HISTORY_LENGTH = 25
def elu_inplace(values: np.ndarray) -> np.ndarray:
    negative = values < 0.0
    values[negative] = np.expm1(values[negative])
    return values


class V5AssistHistory:
    """Fixed 25-frame history with the exact training/deployment observation order."""

    def __init__(self) -> None:
        self.position = np.zeros((HISTORY_LENGTH, 2), dtype=np.float32)
        self.velocity = np.zeros((HISTORY_LENGTH, 2), dtype=np.float32)
        self.torque = np.zeros((HISTORY_LENGTH, 2), dtype=np.float32)
        self.pulse = np.zeros((HISTORY_LENGTH, 12), dtype=np.float32)
        self.next_index = 0

    def reset(self, position_rad: np.ndarray, velocity_rad_s: np.ndarray) -> None:
        position = np.asarray(position_rad, dtype=np.float32)
        velocity = np.asarray(velocity_rad_s, dtype=np.float32)
        if position.shape != (2,) or velocity.shape != (2,):
            raise ValueError("Position and velocity must each have shape (2,)")
        self.position[:] = position
        self.velocity[:] = velocity
        self.torque.fill(0.0)
        self.pulse.fill(0.0)
        self.next_index = 0

    def append(
        self,
        position_rad: np.ndarray,
        velocity_rad_s: np.ndarray,
        previous_smoothed_torque_nm: np.ndarray,
        pulse_state: np.ndarray,
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
        state = np.asarray(pulse_state, dtype=np.float32)
        if state.shape != (12,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 12-element pulse state")
        self.pulse[self.next_index] = state
        self.next_index = (self.next_index + 1) % HISTORY_LENGTH

    def observation(self) -> np.ndarray:
        order = (np.arange(HISTORY_LENGTH) + self.next_index) % HISTORY_LENGTH
        return np.concatenate(
            (self.position[order].reshape(-1), self.velocity[order].reshape(-1), self.torque[order].reshape(-1), self.pulse[order].reshape(-1))
        ).astype(np.float32, copy=False)


class NumpyAssistPolicy:
    """Float32 v5 MLP with no PyTorch dependency."""

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
        expected_weights = ((256, 450), (64, 256), (16, 64), (2, 16))
        expected_biases = ((256,), (64,), (16,), (2,))
        if self.mean.shape != (INPUT_SIZE,) or self.std.shape != (INPUT_SIZE,):
            raise ValueError("Expected 450-element observation normalization arrays")
        for index, (actual, expected) in enumerate(zip(self.weights, expected_weights)):
            if actual.shape != expected:
                raise ValueError(f"weight_{index} has shape {actual.shape}, expected {expected}")
        for index, (actual, expected) in enumerate(zip(self.biases, expected_biases)):
            if actual.shape != expected:
                raise ValueError(f"bias_{index} has shape {actual.shape}, expected {expected}")
        if any(not np.all(np.isfinite(x)) for x in (self.mean, self.std, self.eps, *self.weights, *self.biases)) or self.eps <= 0:
            raise ValueError("Weights and normalization must be finite, epsilon positive")
        if np.any(self.std < 0.0):
            raise ValueError("Observation standard deviations must be nonnegative")

    def forward(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32)
        if x.ndim not in (1, 2) or x.shape[-1] != INPUT_SIZE:
            raise ValueError(f"Expected observation shape (450,) or (N,450), got {x.shape}")
        x = (x - self.mean) / (self.std + self.eps)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = elu_inplace(x @ weight.T + bias)
        return x @ self.weights[-1].T + self.biases[-1]

    __call__ = forward



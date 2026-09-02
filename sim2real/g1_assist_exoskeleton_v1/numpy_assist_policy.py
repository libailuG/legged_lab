#!/usr/bin/env python3
"""Pure NumPy forward pass for the G1 v1 exoskeleton assist policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


INPUT_SIZE = 100
OUTPUT_SIZE = 2
ASSIST_TORQUE_LIMIT = 8.0


def elu_inplace(values: np.ndarray) -> np.ndarray:
    """Apply ELU(alpha=1) without evaluating exp() on the positive branch."""
    negative = values < 0.0
    values[negative] = np.expm1(values[negative])
    return values


class NumpyAssistPolicy:
    """Float32 MLP inference with no PyTorch dependency."""

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
        expected_weights = ((256, 100), (64, 256), (16, 64), (2, 16))
        expected_biases = ((256,), (64,), (16,), (2,))
        if self.mean.shape != (INPUT_SIZE,) or self.std.shape != (INPUT_SIZE,):
            raise ValueError("Expected 100-element observation normalization arrays")
        for index, (actual, expected) in enumerate(zip(self.weights, expected_weights)):
            if actual.shape != expected:
                raise ValueError(f"weight_{index} has shape {actual.shape}, expected {expected}")
        for index, (actual, expected) in enumerate(zip(self.biases, expected_biases)):
            if actual.shape != expected:
                raise ValueError(f"bias_{index} has shape {actual.shape}, expected {expected}")
        if not all(
            array.dtype == np.float32
            for array in (self.mean, self.std, *self.weights, *self.biases)
        ):
            raise ValueError("All policy arrays must be float32")
        if np.any(self.std <= 0.0):
            raise ValueError("Observation standard deviations must be positive")

    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Return raw network actions for one observation or a batch.

        Accepted shapes are ``(100,)`` and ``(batch, 100)``. The returned
        shapes are respectively ``(2,)`` and ``(batch, 2)``.
        """
        x = np.asarray(observation, dtype=np.float32)
        if x.ndim not in (1, 2) or x.shape[-1] != INPUT_SIZE:
            raise ValueError(f"Expected observation shape (100,) or (N,100), got {x.shape}")
        x = (x - self.mean) / (self.std + self.eps)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = x @ weight.T + bias
            x = elu_inplace(x)
        return x @ self.weights[-1].T + self.biases[-1]

    __call__ = forward

    def torque(self, observation: np.ndarray) -> np.ndarray:
        """Return the deployment command in N.m after action clipping."""
        action = np.clip(self.forward(observation), -1.0, 1.0)
        return np.float32(ASSIST_TORQUE_LIMIT) * action


#!/usr/bin/env python3
"""Compile the generated C99 v2 policy and compare it with NumPy inference."""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from numpy_assist_policy import NumpyAssistPolicy


SCRIPT_DIR = Path(__file__).resolve().parent


class Workspace(ctypes.Structure):
    _fields_ = [
        ("hidden_256", ctypes.c_float * 256),
        ("secondary_150", ctypes.c_float * 150),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SCRIPT_DIR / "c/g1_assist_v2_policy.c")
    parser.add_argument("--weights", type=Path, default=SCRIPT_DIR / "weights/assist_policy_v2.npz")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--atol", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = NumpyAssistPolicy(args.weights)
    rng = np.random.default_rng(args.seed)
    observations = rng.normal(0.0, 3.0, size=(args.samples, 150)).astype(np.float32)
    observations[0].fill(0.0)
    expected = policy(observations)

    with tempfile.TemporaryDirectory(prefix="g1_assist_v2_c_test_") as temp_dir:
        library_path = Path(temp_dir) / "libg1_assist_v2_policy.so"
        subprocess.run(
            [
                "gcc", "-std=c99", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra",
                "-Werror", str(args.source), "-o", str(library_path), "-lm",
            ],
            check=True,
        )
        library = ctypes.CDLL(str(library_path))
        float_pointer = ctypes.POINTER(ctypes.c_float)
        library.g1_assist_v2_policy_forward.argtypes = [
            float_pointer,
            float_pointer,
            ctypes.POINTER(Workspace),
        ]
        library.g1_assist_v2_command_speed_gate.argtypes = [ctypes.c_float]
        library.g1_assist_v2_command_speed_gate.restype = ctypes.c_float
        library.g1_assist_v2_slew_limit.argtypes = [
            float_pointer,
            float_pointer,
            float_pointer,
        ]
        library.g1_assist_v2_apply_output_scale.argtypes = [
            float_pointer,
            ctypes.c_float,
            float_pointer,
        ]
        actual = np.empty((args.samples, 2), dtype=np.float32)
        workspace = Workspace()
        for index, observation in enumerate(observations):
            library.g1_assist_v2_policy_forward(
                observation.ctypes.data_as(float_pointer),
                actual[index].ctypes.data_as(float_pointer),
                ctypes.byref(workspace),
            )

        expected_gates = {0.0: 0.0, 0.2: 0.0, 0.45: 0.5, 0.7: 1.0, -0.7: 1.0}
        for speed, expected_gate in expected_gates.items():
            actual_gate = library.g1_assist_v2_command_speed_gate(ctypes.c_float(speed))
            if not np.isclose(actual_gate, expected_gate, rtol=0.0, atol=1e-7):
                raise AssertionError(
                    f"Unexpected C command gate at vx={speed}: "
                    f"{actual_gate}, expected {expected_gate}"
                )

        previous = np.array((0.0, 0.0), dtype=np.float32)
        target = np.array((8.0, -8.0), dtype=np.float32)
        smoothed = np.empty(2, dtype=np.float32)
        motor = np.empty(2, dtype=np.float32)
        library.g1_assist_v2_slew_limit(
            previous.ctypes.data_as(float_pointer),
            target.ctypes.data_as(float_pointer),
            smoothed.ctypes.data_as(float_pointer),
        )
        if not np.array_equal(smoothed, np.array((0.8, -0.8), dtype=np.float32)):
            raise AssertionError(f"Unexpected C slew-limit result: {smoothed}")
        library.g1_assist_v2_apply_output_scale(
            smoothed.ctypes.data_as(float_pointer),
            ctypes.c_float(0.5),
            motor.ctypes.data_as(float_pointer),
        )
        if not np.array_equal(motor, np.array((0.4, -0.4), dtype=np.float32)):
            raise AssertionError(f"Unexpected C final-scale result: {motor}")

    error = np.abs(expected - actual)
    maximum = float(error.max())
    print(f"Samples:        {args.samples}")
    print(f"Maximum error:  {maximum:.9g}")
    print(f"Tolerance:      {args.atol:.9g}")
    if not np.allclose(actual, expected, rtol=0.0, atol=args.atol):
        raise AssertionError("Generated C output differs from NumPy beyond tolerance")
    print("PASS: C99 and NumPy outputs match")
    print("PASS: C99 speed gate, 80 Nm/s slew limiter, and final output scale match")


if __name__ == "__main__":
    main()

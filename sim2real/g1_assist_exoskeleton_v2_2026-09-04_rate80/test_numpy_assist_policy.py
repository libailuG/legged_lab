#!/usr/bin/env python3
"""Compare v2 NumPy inference against the source TorchScript policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from numpy_assist_policy import NumpyAssistPolicy, V2AssistController, command_speed_gate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TORCH_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/"
    "2026-09-04_10-19-39_dynamics_tanh_gate_02_07_rate80/exported/policy.pt"
)
DEFAULT_NUMPY_POLICY = Path(__file__).resolve().parent / "weights/assist_policy_v2.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-policy", type=Path, default=DEFAULT_TORCH_POLICY)
    parser.add_argument("--numpy-policy", type=Path, default=DEFAULT_NUMPY_POLICY)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=3e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    import torch

    torch_policy = torch.jit.load(str(args.torch_policy), map_location="cpu").eval()
    numpy_policy = NumpyAssistPolicy(args.numpy_policy)
    rng = np.random.default_rng(args.seed)
    observations = rng.normal(0.0, 3.0, size=(args.samples, 150)).astype(np.float32)
    observations[0].fill(0.0)
    with torch.inference_mode():
        expected = torch_policy(torch.from_numpy(observations)).cpu().numpy()
    actual = numpy_policy(observations)
    absolute_error = np.abs(expected - actual)
    maximum = float(absolute_error.max())
    mean = float(absolute_error.mean())
    print(f"Samples:        {args.samples}")
    print(f"Maximum error:  {maximum:.9g}")
    print(f"Mean error:     {mean:.9g}")
    print(f"Tolerance:      {args.atol:.9g}")
    if not np.allclose(actual, expected, rtol=0.0, atol=args.atol):
        raise AssertionError("NumPy output differs from TorchScript beyond tolerance")
    print("PASS: NumPy and TorchScript outputs match")

    if command_speed_gate(0.2) != 0.0 or command_speed_gate(0.7) != 1.0:
        raise AssertionError("Command-speed gate endpoints do not match training")
    if command_speed_gate(float("nan")) != 0.0:
        raise AssertionError("Invalid command must fail safe to zero assist")
    if not np.isclose(command_speed_gate(0.45), 0.5, rtol=0.0, atol=1e-7):
        raise AssertionError("Command-speed gate midpoint is not smoothstep(0.5)")
    print("PASS: Command-speed gate matches 0.2-to-0.7 m/s training settings")

    full = V2AssistController(numpy_policy, output_scale=1.0)
    half = V2AssistController(numpy_policy, output_scale=0.5)
    initial_position = np.array((0.1, -0.2), dtype=np.float32)
    initial_velocity = np.array((0.0, 0.0), dtype=np.float32)
    full.reset(initial_position, initial_velocity)
    half.reset(initial_position, initial_velocity)
    for _ in range(20):
        position = rng.normal(0.0, 0.3, size=2).astype(np.float32)
        velocity = rng.normal(0.0, 1.0, size=2).astype(np.float32)
        full_motor = full.step(position, velocity, forward_command_m_s=0.7)
        half_motor = half.step(position, velocity, forward_command_m_s=0.7)
        if not np.array_equal(
            full.previous_smoothed_torque_nm,
            half.previous_smoothed_torque_nm,
        ):
            raise AssertionError("Final output scale changed internal smoothed torque")
        if not np.array_equal(full.history.observation(), half.history.observation()):
            raise AssertionError("Final output scale changed policy observation/history")
        if not np.allclose(half_motor, 0.5 * full_motor, rtol=0.0, atol=1e-7):
            raise AssertionError("Final output scale was not applied only at motor output")
    print("PASS: Output scale does not change policy observations or torque history")


if __name__ == "__main__":
    main()

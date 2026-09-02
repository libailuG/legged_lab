#!/usr/bin/env python3
"""Compare the pure NumPy policy against its TorchScript source."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from numpy_assist_policy import NumpyAssistPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TORCH_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_v1_ppo/2026-08-18_13-27-09/exported/policy.pt"
)
DEFAULT_NUMPY_POLICY = Path(__file__).resolve().parent / "weights/assist_policy_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-policy", type=Path, default=DEFAULT_TORCH_POLICY)
    parser.add_argument("--numpy-policy", type=Path, default=DEFAULT_NUMPY_POLICY)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("This comparison test requires PyTorch") from exc

    torch_policy = torch.jit.load(str(args.torch_policy), map_location="cpu").eval()
    numpy_policy = NumpyAssistPolicy(args.numpy_policy)
    rng = np.random.default_rng(args.seed)

    # Include reset-like zero inputs and broad randomized observations.
    observations = rng.normal(0.0, 3.0, size=(args.samples, 100)).astype(np.float32)
    observations[0].fill(0.0)
    with torch.inference_mode():
        expected = torch_policy(torch.from_numpy(observations)).cpu().numpy()
    actual = numpy_policy(observations)
    absolute_error = np.abs(expected - actual)
    max_error = float(absolute_error.max())
    mean_error = float(absolute_error.mean())
    print(f"Samples:        {args.samples}")
    print(f"Maximum error:  {max_error:.9g}")
    print(f"Mean error:     {mean_error:.9g}")
    print(f"Tolerance:      {args.atol:.9g}")
    if not np.allclose(actual, expected, rtol=0.0, atol=args.atol):
        worst = np.unravel_index(int(absolute_error.argmax()), absolute_error.shape)
        raise AssertionError(
            f"NumPy result differs at {worst}: numpy={actual[worst]}, "
            f"torch={expected[worst]}, error={absolute_error[worst]}"
        )
    print("PASS: NumPy and TorchScript outputs match")


if __name__ == "__main__":
    main()

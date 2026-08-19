#!/usr/bin/env python3
"""Export the v1 exoskeleton TorchScript policy to a NumPy ``.npz`` file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_v1_ppo/2026-08-18_13-27-09/exported/policy.pt"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "weights/assist_policy_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def as_float32(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=True)


def main() -> None:
    args = parse_args()
    policy_path = args.policy.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"TorchScript policy not found: {policy_path}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Export requires a Python environment containing PyTorch. "
            "The exported policy does not require PyTorch."
        ) from exc

    model = torch.jit.load(str(policy_path), map_location="cpu").eval()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    expected_parameters = {
        "actor.0.weight",
        "actor.0.bias",
        "actor.2.weight",
        "actor.2.bias",
        "actor.4.weight",
        "actor.4.bias",
        "actor.6.weight",
        "actor.6.bias",
    }
    if set(parameters) != expected_parameters:
        raise ValueError(
            "Unexpected actor structure. Expected the v1 100-256-64-16-2 MLP, "
            f"but found parameters: {sorted(parameters)}"
        )
    for name in ("normalizer._mean", "normalizer._std"):
        if name not in buffers:
            raise ValueError(f"TorchScript policy is missing {name}")

    arrays = {
        "obs_mean": as_float32(buffers["normalizer._mean"]).reshape(-1),
        "obs_std": as_float32(buffers["normalizer._std"]).reshape(-1),
        "normalizer_eps": np.array(float(model.normalizer.eps), dtype=np.float32),
        "layer_sizes": np.array((100, 256, 64, 16, 2), dtype=np.int32),
    }
    for output_index, module_index in enumerate((0, 2, 4, 6)):
        arrays[f"weight_{output_index}"] = as_float32(
            parameters[f"actor.{module_index}.weight"]
        )
        arrays[f"bias_{output_index}"] = as_float32(
            parameters[f"actor.{module_index}.bias"]
        )

    expected_shapes = {
        "obs_mean": (100,),
        "obs_std": (100,),
        "weight_0": (256, 100),
        "bias_0": (256,),
        "weight_1": (64, 256),
        "bias_1": (64,),
        "weight_2": (16, 64),
        "bias_2": (16,),
        "weight_3": (2, 16),
        "bias_3": (2,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Unexpected {name} shape: {arrays[name].shape}, expected {shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    manifest = {
        "source_policy": str(policy_path),
        "weights_file": output_path.name,
        "dtype": "float32",
        "input_order": "25 oldest-to-newest [left_vel,right_vel], then 25 oldest-to-newest [left_torque,right_torque]",
        "normalization": "(input - obs_mean) / (obs_std + normalizer_eps)",
        "layers": [100, 256, 64, 16, 2],
        "hidden_activation": "ELU(alpha=1.0)",
        "output_activation": "none (clip to [-1,1] before multiplying by 8 Nm)",
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    parameter_count = sum(
        arrays[f"weight_{i}"].size + arrays[f"bias_{i}"].size for i in range(4)
    )
    print(f"Exported:   {output_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"Parameters: {parameter_count} float32 ({parameter_count * 4} bytes)")


if __name__ == "__main__":
    main()


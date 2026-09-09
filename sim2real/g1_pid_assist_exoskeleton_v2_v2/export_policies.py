#!/usr/bin/env python3
"""Export the current v2 assist TorchScript policy to portable NumPy weights."""

from __future__ import annotations

import argparse
import json
import shutil
import hashlib
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_v2_v2_ppo/"
    "2026-09-04_17-54-31_lift10_rate80_press4_rate40_motion_gate_resume/exported/policy.pt"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "weights/assist_policy_v2.npz"
LAYER_SIZES = (150, 256, 64, 16, 2)


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
        raise RuntimeError("Export requires a Python environment containing PyTorch") from exc

    model = torch.jit.load(str(policy_path), map_location="cpu").eval()
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    expected_parameters = {
        f"actor.{module_index}.{kind}"
        for module_index in (0, 2, 4, 6)
        for kind in ("weight", "bias")
    }
    if set(parameters) != expected_parameters:
        raise ValueError(
            "Unexpected actor structure. Expected the v2 150-256-64-16-2 MLP, "
            f"found: {sorted(parameters)}"
        )
    for name in ("normalizer._mean", "normalizer._std"):
        if name not in buffers:
            raise ValueError(f"TorchScript policy is missing {name}")

    arrays = {
        "obs_mean": as_float32(buffers["normalizer._mean"]).reshape(-1),
        "obs_std": as_float32(buffers["normalizer._std"]).reshape(-1),
        "normalizer_eps": np.array(float(model.normalizer.eps), dtype=np.float32),
        "layer_sizes": np.array(LAYER_SIZES, dtype=np.int32),
    }
    for output_index, module_index in enumerate((0, 2, 4, 6)):
        arrays[f"weight_{output_index}"] = as_float32(
            parameters[f"actor.{module_index}.weight"]
        )
        arrays[f"bias_{output_index}"] = as_float32(
            parameters[f"actor.{module_index}.bias"]
        )

    expected_shapes = {
        "obs_mean": (150,),
        "obs_std": (150,),
        "weight_0": (256, 150),
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
        "policy_rate_hz": 100,
        "history_length": 25,
        "input_order": [
            "25 oldest-to-newest [left_position_rad,right_position_rad]",
            "25 oldest-to-newest [left_velocity_rad_s,right_velocity_rad_s]",
            "25 oldest-to-newest [left_smoothed_torque_nm,right_smoothed_torque_nm]",
        ],
        "normalization": "(input - obs_mean) / (obs_std + normalizer_eps)",
        "layers": list(LAYER_SIZES),
        "hidden_activation": "ELU(alpha=1.0)",
        "output": "motion gate from filtered joint velocity, target -10/+4 Nm, directional slew -80/+40 Nm/s, final physical clamp -8/+4 Nm",
        "motion_gate": {"deadzone_rad_s": 0.15, "full_rad_s": 0.8, "filter_tau_s": 0.05},
        "physical_limit_nm": [-8.0, 4.0],
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    parameter_count = sum(
        arrays[f"weight_{i}"].size + arrays[f"bias_{i}"].size for i in range(4)
    )
    # Bundle immutable source copies so deployment does not depend on repository paths.
    sources = {
        "assist_policy.pt": policy_path,
        "gait_policy.pt": PROJECT_ROOT / "logs/rsl_rl/g1_assist_v1_amp/2026-09-04_20-03-58_rtx5090_bounded_noise_entropy0_remaining_29600/exported/policy.pt",
    }
    provenance = {}
    for name, source in sources.items():
        destination = output_path.parent / name
        shutil.copy2(source, destination)
        provenance[name] = {"source": str(source), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (output_path.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported:   {output_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"Parameters: {parameter_count} float32 ({parameter_count * 4} bytes)")


if __name__ == "__main__":
    main()

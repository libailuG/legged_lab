#!/usr/bin/env python3
"""Launch the G1 29-DoF assist model in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/g1_29dof_assist.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"MuJoCo XML path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Simulation speed relative to real time (default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    if args.realtime_factor <= 0.0:
        raise ValueError("--realtime-factor must be greater than zero")

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is not installed in this Python environment. "
            "Install it with: pip install mujoco"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded: {model_path}")
    print(f"Model: {model.names.decode().split(chr(0), 1)[0] if model.names else 'unknown'}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, bodies={model.nbody}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_started = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()

            target_step_time = model.opt.timestep / args.realtime_factor
            remaining = target_step_time - (time.perf_counter() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the G1 gait policy with pure NumPy v1 exoskeleton inference in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# A 43k-parameter single-sample MLP is faster without a large BLAS thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUJOCO_SCRIPTS = PROJECT_ROOT / "scripts/mujoco"
if str(MUJOCO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPTS))

from numpy_assist_policy import NumpyAssistPolicy  # noqa: E402
from sim2sim_g1_29dof_assist import (  # noqa: E402
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    EFFORT_LIMIT,
    KD,
    KP,
    NUM_ACTIONS,
    NUM_OBSERVATIONS,
    VelocityCommand,
    make_observation,
    require_file,
)
from sim2sim_g1_assist_exoskeleton import (  # noqa: E402
    ASSIST_ACTION_SCALE,
    DECIMATION,
    SIDES,
    SIM_DT,
    build_indices,
    create_plot,
    reset_robot,
    update_plot,
)
from sim2sim_g1_assist_exoskeleton_v1 import V1AssistHistory  # noqa: E402


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/"
    "g1_29dof_assist_exoskeleton.xml"
)
DEFAULT_GAIT_POLICY = (
    PROJECT_ROOT / "logs/rsl_rl/g1_assist_amp/2026-08-13_13-45-16/exported/policy.pt"
)
DEFAULT_NUMPY_POLICY = Path(__file__).resolve().parent / "weights/assist_policy_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gait-policy", type=Path, default=DEFAULT_GAIT_POLICY)
    parser.add_argument("--numpy-policy", type=Path, default=DEFAULT_NUMPY_POLICY)
    parser.add_argument("--vx", type=float, default=0.7)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--no-live-plot", action="store_true")
    parser.add_argument("--plot-window", type=float, default=5.0)
    parser.add_argument("--plot-interval", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--observation-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-reset-height", type=float, default=0.35)
    parser.add_argument("--auto-reset-angle-deg", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = require_file(args.model, "MuJoCo model")
    gait_policy_path = require_file(args.gait_policy, "Gait TorchScript policy")
    numpy_policy_path = require_file(args.numpy_policy, "NumPy assist policy")
    if args.headless and args.duration is None:
        args.duration = 10.0
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")

    import mujoco
    import torch

    import matplotlib

    if args.headless or args.no_live_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gait_policy = torch.jit.load(str(gait_policy_path), map_location="cpu").eval()
    assist_policy = NumpyAssistPolicy(numpy_policy_path)
    with torch.inference_mode():
        gait_test = gait_policy(torch.zeros(1, NUM_OBSERVATIONS))
    assist_test = assist_policy(np.zeros(100, dtype=np.float32))
    if tuple(gait_test.shape) != (1, NUM_ACTIONS):
        raise ValueError("Gait policy must map 96 observations to 29 actions")
    if assist_test.shape != (2,):
        raise ValueError("NumPy assist policy must map 100 observations to 2 actions")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    indices = build_indices(mujoco, model)
    model.dof_armature[indices["policy_qvel"]] = 0.01
    data = mujoco.MjData(model)
    history = V1AssistHistory(args.observation_noise, args.seed)
    reset_robot(mujoco, model, data, indices, history)

    command = VelocityCommand(args.vx, args.vy, args.yaw)
    command.clamp()
    last_gait_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    gait_target = DEFAULT_JOINT_POS.copy()
    assist_torque = np.zeros(2, dtype=np.float64)
    reset_requested = False

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "g1_assist_numpy_sim2sim.csv"
    png_path = output_dir / "g1_assist_numpy_sim2sim.png"
    signal_names = ["time"] + [
        f"{side}_{signal}"
        for side in SIDES
        for signal in (
            "hip_pos",
            "assist_pos",
            "pos_error",
            "hip_vel",
            "assist_vel",
            "hip_torque",
            "assist_torque",
        )
    ]
    plot_data = {name: [] for name in signal_names}
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(signal_names)
    fig, axes, lines = create_plot()
    live_plot = not args.headless and not args.no_live_plot
    if live_plot:
        plt.ion()
        fig.show()

    def key_callback(keycode: int) -> None:
        nonlocal reset_requested
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "W":
            command.vx += 0.1
        elif key == "S":
            command.vx -= 0.1
        elif key == "A":
            command.vy += 0.1
        elif key == "D":
            command.vy -= 0.1
        elif key == "Q":
            command.yaw += 0.1
        elif key == "E":
            command.yaw -= 0.1
        elif key == " ":
            command.zero()
        elif key == "R":
            reset_requested = True
        else:
            return
        command.clamp()
        print(f"command: vx={command.vx:.2f}, vy={command.vy:.2f}, yaw={command.yaw:.2f}")

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

    print(f"Model:                {model_path}")
    print(f"Gait policy (Torch):  {gait_policy_path}")
    print(f"Assist policy (NumPy): {numpy_policy_path}")
    print(f"Control: dt={SIM_DT}, decimation={DECIMATION}, policy rate=50 Hz")
    print(f"Output:               {output_dir}")
    if not args.headless:
        print("Keys: W/S vx, A/D vy, Q/E yaw, Space zero command, R reset")

    step = 0
    control_step = 0
    resets = 0
    assist_inference_seconds = 0.0
    last_plot_time = 0.0
    started = time.perf_counter()
    try:
        while viewer is None or viewer.is_running():
            sim_time = step * SIM_DT
            if args.duration is not None and sim_time >= args.duration:
                break
            step_started = time.perf_counter()

            if step % DECIMATION == 0:
                angle_error_deg = np.rad2deg(
                    data.qpos[indices["assist_qpos"]] - data.qpos[indices["hip_qpos"]]
                )
                angle_reset = (
                    args.auto_reset_angle_deg > 0.0
                    and np.max(np.abs(angle_error_deg)) > args.auto_reset_angle_deg
                )
                height_reset = (
                    args.auto_reset_height > 0.0
                    and float(data.qpos[2]) < args.auto_reset_height
                )
                if reset_requested or angle_reset or height_reset:
                    reset_robot(mujoco, model, data, indices, history)
                    last_gait_action.fill(0.0)
                    gait_target[:] = DEFAULT_JOINT_POS
                    assist_torque.fill(0.0)
                    reset_requested = False
                    control_step = 0
                    resets += 1

                if control_step > 0:
                    history.append(
                        data.qpos[indices["assist_qpos"]],
                        data.qvel[indices["assist_qvel"]],
                        assist_torque,
                    )
                gait_obs = make_observation(
                    data,
                    indices["policy_qpos"],
                    indices["policy_qvel"],
                    indices["pelvis"],
                    indices["gyro_adr"],
                    command,
                    last_gait_action,
                )
                with torch.inference_mode():
                    gait_output = gait_policy(torch.from_numpy(gait_obs).unsqueeze(0))
                last_gait_action = gait_output.squeeze(0).numpy().astype(np.float64)
                gait_target = DEFAULT_JOINT_POS + ACTION_SCALE * last_gait_action

                inference_started = time.perf_counter()
                assist_torque = assist_policy.torque(history.observation()).astype(np.float64)
                assist_inference_seconds += time.perf_counter() - inference_started
                control_step += 1

            joint_pos = data.qpos[indices["policy_qpos"]]
            joint_vel = data.qvel[indices["policy_qvel"]]
            gait_torque = np.clip(
                KP * (gait_target - joint_pos) - KD * joint_vel,
                -EFFORT_LIMIT,
                EFFORT_LIMIT,
            )
            data.ctrl[indices["policy_actuator"]] = gait_torque
            data.ctrl[indices["assist_actuator"]] = np.clip(
                assist_torque, -ASSIST_ACTION_SCALE, ASSIST_ACTION_SCALE
            )
            mujoco.mj_step(model, data)

            if step % DECIMATION == 0:
                log_time = float(data.time)
                hip_pos = np.rad2deg(data.qpos[indices["hip_qpos"]])
                assist_pos = np.rad2deg(data.qpos[indices["assist_qpos"]])
                hip_vel = np.rad2deg(data.qvel[indices["hip_qvel"]])
                assist_vel = np.rad2deg(data.qvel[indices["assist_qvel"]])
                hip_torque = np.asarray(
                    data.actuator_force[indices["hip_actuator"]]
                ).copy()
                actual_assist_torque = np.asarray(
                    data.actuator_force[indices["assist_actuator"]]
                ).copy()
                position_error = assist_pos - hip_pos

                row = [log_time]
                plot_data["time"].append(log_time)
                for index, side in enumerate(SIDES):
                    values = {
                        "hip_pos": hip_pos[index],
                        "assist_pos": assist_pos[index],
                        "pos_error": position_error[index],
                        "hip_vel": hip_vel[index],
                        "assist_vel": assist_vel[index],
                        "hip_torque": hip_torque[index],
                        "assist_torque": actual_assist_torque[index],
                    }
                    for signal, value in values.items():
                        scalar = float(value)
                        plot_data[f"{side}_{signal}"].append(scalar)
                        row.append(scalar)
                csv_writer.writerow(row)

                if viewer is not None:
                    viewer.cam.lookat[:] = data.qpos[:3]
                    viewer.sync()
                if live_plot and log_time - last_plot_time >= args.plot_interval:
                    update_plot(axes, lines, plot_data, args.plot_window)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    last_plot_time = log_time
            if not args.no_realtime:
                remaining = SIM_DT - (time.perf_counter() - step_started)
                if remaining > 0.0:
                    time.sleep(remaining)
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        simulated_time = step * SIM_DT
        update_plot(axes, lines, plot_data, max(simulated_time, args.plot_window))
        fig.suptitle(
            f"G1 Assist Exoskeleton NumPy MuJoCo Sim2Sim ({simulated_time:.2f} s)",
            y=1.002,
        )
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        csv_file.close()
        if viewer is not None:
            viewer.close()
        plt.ioff()
        plt.close(fig)

    wall_time = time.perf_counter() - started
    average_us = 1e6 * assist_inference_seconds / max(control_step, 1)
    print(
        f"Finished: simulated={step * SIM_DT:.2f}s wall={wall_time:.2f}s "
        f"position=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) "
        f"resets={resets}"
    )
    print(f"NumPy assist inference: {average_us:.1f} us/call over {control_step} calls")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()

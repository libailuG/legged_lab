#!/usr/bin/env python3
"""Run the frozen gait policy plus the 100-observation v1 assist PPO in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from sim2sim_g1_29dof_assist import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    EFFORT_LIMIT,
    KD,
    KP,
    NUM_ACTIONS,
    NUM_OBSERVATIONS,
    PROJECT_ROOT,
    VelocityCommand,
    make_observation,
    require_file,
)
from sim2sim_g1_assist_exoskeleton import (
    ASSIST_ACTION_SCALE,
    DECIMATION,
    SIM_DT,
    SIDES,
    build_indices,
    create_plot,
    reset_robot,
    update_plot,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/"
    "g1_29dof_assist_exoskeleton.xml"
)
DEFAULT_GAIT_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_amp/2026-08-13_13-45-16/exported/policy.pt"
)
DEFAULT_ASSIST_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_v1_ppo/2026-08-18_13-27-09/exported/policy.pt"
)

ASSIST_HISTORY_LENGTH = 25
ASSIST_OBSERVATIONS = 2 * 2 * ASSIST_HISTORY_LENGTH
ASSIST_VELOCITY_NOISE = 0.5
ASSIST_TORQUE_NOISE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gait-policy", type=Path, default=DEFAULT_GAIT_POLICY)
    parser.add_argument("--assist-policy", type=Path, default=DEFAULT_ASSIST_POLICY)
    parser.add_argument("--vx", type=float, default=0.7, help="Initial forward command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial lateral command in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw-rate command in rad/s.")
    parser.add_argument("--duration", type=float, default=None, help="Unlimited with viewer.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace to wall-clock time.")
    parser.add_argument("--no-live-plot", action="store_true", help="Only save the final plot.")
    parser.add_argument("--plot-window", type=float, default=5.0)
    parser.add_argument("--plot-interval", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--observation-noise",
        action="store_true",
        help=(
            "Apply the v1 training noise to velocity and torque observations. "
            "Disabled by default for deployment-style evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Observation-noise seed.")
    parser.add_argument("--auto-reset-height", type=float, default=0.35)
    parser.add_argument(
        "--auto-reset-angle-deg",
        type=float,
        default=0.0,
        help="Reset above this assist-to-hip angle error; use 4 for Isaac parity.",
    )
    return parser.parse_args()


class V1AssistHistory:
    """Oldest-to-newest v1 history: 25 velocity samples then 25 torque samples."""

    def __init__(self, add_noise: bool, seed: int):
        self.velocity = deque(maxlen=ASSIST_HISTORY_LENGTH)
        self.torque = deque(maxlen=ASSIST_HISTORY_LENGTH)
        self.add_noise = add_noise
        self.rng = np.random.default_rng(seed)

    def _velocity_sample(self, velocity: np.ndarray) -> np.ndarray:
        sample = np.asarray(velocity, dtype=np.float32).copy()
        if self.add_noise:
            sample += self.rng.uniform(
                -ASSIST_VELOCITY_NOISE, ASSIST_VELOCITY_NOISE, size=2
            ).astype(np.float32)
        return sample

    def _torque_sample(self, torque: np.ndarray) -> np.ndarray:
        sample = np.asarray(torque, dtype=np.float32).copy()
        if self.add_noise:
            sample += self.rng.uniform(
                -ASSIST_TORQUE_NOISE, ASSIST_TORQUE_NOISE, size=2
            ).astype(np.float32)
        return sample

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        del position
        self.velocity.clear()
        self.torque.clear()
        for _ in range(ASSIST_HISTORY_LENGTH):
            self.velocity.append(self._velocity_sample(velocity))
            self.torque.append(self._torque_sample(np.zeros(2, dtype=np.float32)))

    def append(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        del position
        self.velocity.append(self._velocity_sample(velocity))
        self.torque.append(self._torque_sample(torque))

    def observation(self) -> np.ndarray:
        observation = np.concatenate(
            (
                np.asarray(self.velocity, dtype=np.float32).reshape(-1),
                np.asarray(self.torque, dtype=np.float32).reshape(-1),
            )
        )
        if observation.shape != (ASSIST_OBSERVATIONS,):
            raise RuntimeError(f"Unexpected v1 assist observation shape: {observation.shape}")
        return observation


def main() -> None:
    args = parse_args()
    model_path = require_file(args.model, "MuJoCo model")
    gait_policy_path = require_file(args.gait_policy, "Gait TorchScript policy")
    assist_policy_path = require_file(args.assist_policy, "V1 assist TorchScript policy")
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

    gait_policy = torch.jit.load(str(gait_policy_path), map_location="cpu")
    assist_policy = torch.jit.load(str(assist_policy_path), map_location="cpu")
    gait_policy.eval()
    assist_policy.eval()
    with torch.inference_mode():
        gait_test_output = gait_policy(torch.zeros(1, NUM_OBSERVATIONS))
        assist_test_output = assist_policy(torch.zeros(1, ASSIST_OBSERVATIONS))
    if tuple(gait_test_output.shape) != (1, NUM_ACTIONS):
        raise ValueError("Frozen gait policy must map 96 observations to 29 actions")
    if tuple(assist_test_output.shape) != (1, 2):
        raise ValueError("V1 assist policy must map 100 observations to 2 actions")

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

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else assist_policy_path.parent.parent / "sim2sim_analysis_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "g1_assist_exoskeleton_v1_sim2sim.csv"
    png_path = output_dir / "g1_assist_exoskeleton_v1_sim2sim.png"
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
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(signal_names)

    max_plot_points = max(
        100, math.ceil(max(args.plot_window, 1.0) / (SIM_DT * DECIMATION)) * 2
    )
    plot_data = {name: deque(maxlen=max_plot_points) for name in signal_names}
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

    print(f"Model:         {model_path}")
    print(f"Gait policy:   {gait_policy_path}")
    print(f"V1 policy:     {assist_policy_path}")
    print(f"Assist obs:    {ASSIST_OBSERVATIONS} (25 velocity + 25 torque samples)")
    print(f"Obs noise:     {'enabled' if args.observation_noise else 'disabled'}")
    print(f"Control:       dt={SIM_DT}, decimation={DECIMATION}, policy rate=50 Hz")
    print(f"Output:        {output_dir}")
    if not args.headless:
        print("Keys: W/S vx, A/D vy, Q/E yaw, Space zero command, R reset")

    step = 0
    control_step = 0
    resets = 0
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
                assist_obs = history.observation()
                with torch.inference_mode():
                    gait_action_tensor = gait_policy(torch.from_numpy(gait_obs).unsqueeze(0))
                    assist_action_tensor = assist_policy(
                        torch.from_numpy(assist_obs).unsqueeze(0)
                    )
                last_gait_action = gait_action_tensor.squeeze(0).numpy().astype(np.float64)
                assist_action = np.clip(
                    assist_action_tensor.squeeze(0).numpy(), -1.0, 1.0
                )
                gait_target = DEFAULT_JOINT_POS + ACTION_SCALE * last_gait_action
                assist_torque = ASSIST_ACTION_SCALE * assist_action.astype(np.float64)
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

                row = [sim_time]
                plot_data["time"].append(sim_time)
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
                if live_plot and sim_time - last_plot_time >= args.plot_interval:
                    update_plot(axes, lines, plot_data, args.plot_window)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    last_plot_time = sim_time

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
            f"G1 Assist Exoskeleton v1 MuJoCo Sim2Sim ({simulated_time:.2f} s)",
            y=1.002,
        )
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        csv_file.close()
        if viewer is not None:
            viewer.close()
        plt.ioff()
        plt.close(fig)

    wall_time = time.perf_counter() - started
    print(
        f"Finished: simulated={step * SIM_DT:.2f}s wall={wall_time:.2f}s "
        f"position=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) "
        f"resets={resets}"
    )
    print(f"CSV:  {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()

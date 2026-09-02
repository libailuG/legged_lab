#!/usr/bin/env python3
"""Run the frozen gait policy plus assist PPO on the G1 exoskeleton in MuJoCo."""

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
    POLICY_JOINT_NAMES,
    PROJECT_ROOT,
    VelocityCommand,
    make_observation,
    name_to_id,
    require_file,
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
    / "logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-14_11-06-58/exported/policy.pt"
)

ASSIST_JOINT_NAMES = (
    "left_hip_pitch_assist_joint",
    "right_hip_pitch_assist_joint",
)
HIP_PITCH_JOINT_NAMES = ("left_hip_pitch_joint", "right_hip_pitch_joint")
SIDES = ("left", "right")
ASSIST_HISTORY_LENGTH = 25
ASSIST_OBSERVATIONS = 2 * 3 * ASSIST_HISTORY_LENGTH
ASSIST_ACTION_SCALE = 8.0

# Match Isaac: 1 ms physics and one policy action every 20 ms (50 Hz).
SIM_DT = 0.001
DECIMATION = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Exoskeleton MuJoCo XML path.")
    parser.add_argument("--gait-policy", type=Path, default=DEFAULT_GAIT_POLICY, help="Frozen 29-joint policy.pt.")
    parser.add_argument("--assist-policy", type=Path, default=DEFAULT_ASSIST_POLICY, help="Exported assist policy.pt.")
    parser.add_argument("--vx", type=float, default=0.7, help="Initial forward command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial lateral command in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw-rate command in rad/s.")
    parser.add_argument("--duration", type=float, default=None, help="Run duration; unlimited with viewer.")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer; defaults to 10 s.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace simulation to wall-clock time.")
    parser.add_argument("--no-live-plot", action="store_true", help="Only save the final plot.")
    parser.add_argument("--plot-window", type=float, default=5.0, help="Seconds shown by the live plot.")
    parser.add_argument("--plot-interval", type=float, default=0.05, help="Live-plot update period in seconds.")
    parser.add_argument("--output-dir", type=Path, default=None, help="CSV/PNG output directory.")
    parser.add_argument(
        "--auto-reset-height",
        type=float,
        default=0.35,
        help="Reset below this pelvis height; <=0 disables it.",
    )
    parser.add_argument(
        "--auto-reset-angle-deg",
        type=float,
        default=0.0,
        help="Reset when |assist - hip| exceeds this value; disabled by default, use 4 for Isaac parity.",
    )
    return parser.parse_args()


def build_indices(mujoco, model):
    policy_joint_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in POLICY_JOINT_NAMES],
        dtype=int,
    )
    policy_actuator_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in POLICY_JOINT_NAMES],
        dtype=int,
    )
    assist_joint_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ASSIST_JOINT_NAMES],
        dtype=int,
    )
    assist_actuator_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ASSIST_JOINT_NAMES],
        dtype=int,
    )
    hip_joint_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HIP_PITCH_JOINT_NAMES],
        dtype=int,
    )
    hip_actuator_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in HIP_PITCH_JOINT_NAMES],
        dtype=int,
    )

    pelvis_id = name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    gyro_id = name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")
    gyro_adr = int(model.sensor_adr[gyro_id])
    return {
        "policy_qpos": model.jnt_qposadr[policy_joint_ids].astype(int),
        "policy_qvel": model.jnt_dofadr[policy_joint_ids].astype(int),
        "policy_actuator": policy_actuator_ids,
        "assist_qpos": model.jnt_qposadr[assist_joint_ids].astype(int),
        "assist_qvel": model.jnt_dofadr[assist_joint_ids].astype(int),
        "assist_actuator": assist_actuator_ids,
        "hip_qpos": model.jnt_qposadr[hip_joint_ids].astype(int),
        "hip_qvel": model.jnt_dofadr[hip_joint_ids].astype(int),
        "hip_actuator": hip_actuator_ids,
        "pelvis": pelvis_id,
        "gyro_adr": gyro_adr,
    }


class AssistHistory:
    """Isaac-compatible oldest-to-newest history for three 2-D observation terms."""

    def __init__(self):
        self.position = deque(maxlen=ASSIST_HISTORY_LENGTH)
        self.velocity = deque(maxlen=ASSIST_HISTORY_LENGTH)
        self.torque = deque(maxlen=ASSIST_HISTORY_LENGTH)

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position.clear()
        self.velocity.clear()
        self.torque.clear()
        for _ in range(ASSIST_HISTORY_LENGTH):
            self.position.append(np.asarray(position, dtype=np.float32).copy())
            self.velocity.append(np.asarray(velocity, dtype=np.float32).copy())
            self.torque.append(np.zeros(2, dtype=np.float32))

    def append(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        self.position.append(np.asarray(position, dtype=np.float32).copy())
        self.velocity.append(np.asarray(velocity, dtype=np.float32).copy())
        self.torque.append(np.asarray(torque, dtype=np.float32).copy())

    def observation(self) -> np.ndarray:
        observation = np.concatenate(
            (
                np.asarray(self.position, dtype=np.float32).reshape(-1),
                np.asarray(self.velocity, dtype=np.float32).reshape(-1),
                np.asarray(self.torque, dtype=np.float32).reshape(-1),
            )
        )
        if observation.shape != (ASSIST_OBSERVATIONS,):
            raise RuntimeError(f"Unexpected assist observation shape: {observation.shape}")
        return observation


def reset_robot(mujoco, model, data, indices, history: AssistHistory) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[indices["policy_qpos"]] = DEFAULT_JOINT_POS
    # The Isaac reset synchronizes each assist joint with its paired hip joint.
    data.qpos[indices["assist_qpos"]] = data.qpos[indices["hip_qpos"]]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    history.reset(data.qpos[indices["assist_qpos"]], data.qvel[indices["assist_qvel"]])


def create_plot():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex="col")
    lines = {}
    for col, side in enumerate(SIDES):
        lines[f"{side}_hip_pos"], = axes[0, col].plot([], [], label="hip pitch")
        lines[f"{side}_assist_pos"], = axes[0, col].plot([], [], label="assist")
        axes[0, col].set_title(f"{side.capitalize()} joint angle")
        axes[0, col].set_ylabel("Angle (deg)")

        lines[f"{side}_pos_error"], = axes[1, col].plot([], [], color="tab:purple", label="assist - hip")
        axes[1, col].axhline(4.0, color="tab:red", linestyle="--", label="±4°")
        axes[1, col].axhline(-4.0, color="tab:red", linestyle="--")
        axes[1, col].set_title(f"{side.capitalize()} angle difference")
        axes[1, col].set_ylabel("Difference (deg)")

        lines[f"{side}_hip_vel"], = axes[2, col].plot([], [], label="hip pitch")
        lines[f"{side}_assist_vel"], = axes[2, col].plot([], [], label="assist")
        axes[2, col].set_title(f"{side.capitalize()} joint velocity")
        axes[2, col].set_ylabel("Velocity (deg/s)")

        lines[f"{side}_hip_torque"], = axes[3, col].plot([], [], label="hip pitch (max |tau|=0.00 N.m)")
        lines[f"{side}_assist_torque"], = axes[3, col].plot(
            [], [], color="tab:green", label="assist (max |tau|=0.00 N.m)"
        )
        axes[3, col].set_title(f"{side.capitalize()} normalized actuator torque")
        axes[3, col].set_xlabel("Time (s)")
        axes[3, col].set_ylabel("Torque / max |Torque|")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, axes, lines


def update_plot(axes, lines, plot_data, plot_window: float) -> None:
    times = np.asarray(plot_data["time"])
    if len(times) < 2:
        return
    mask = times >= max(0.0, times[-1] - plot_window)
    time_view = times[mask]
    for col, side in enumerate(SIDES):
        for signal in ("hip_pos", "assist_pos", "pos_error", "hip_vel", "assist_vel"):
            key = f"{side}_{signal}"
            lines[key].set_data(time_view, np.asarray(plot_data[key])[mask])
        hip_torque = np.asarray(plot_data[f"{side}_hip_torque"])[mask]
        assist_torque = np.asarray(plot_data[f"{side}_assist_torque"])[mask]
        hip_max = float(np.max(np.abs(hip_torque))) if hip_torque.size else 0.0
        assist_max = float(np.max(np.abs(assist_torque))) if assist_torque.size else 0.0
        lines[f"{side}_hip_torque"].set_data(
            time_view, hip_torque / hip_max if hip_max > 0.0 else hip_torque
        )
        lines[f"{side}_assist_torque"].set_data(
            time_view,
            assist_torque / assist_max if assist_max > 0.0 else assist_torque,
        )
        lines[f"{side}_hip_torque"].set_label(
            f"hip pitch (max |tau|={hip_max:.2f} N.m)"
        )
        lines[f"{side}_assist_torque"].set_label(
            f"assist (max |tau|={assist_max:.2f} N.m)"
        )
        axes[3, col].legend(loc="upper right")
    for ax in axes.flat:
        ax.set_xlim(time_view[0], time_view[-1] + 0.02)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
    for col in range(2):
        axes[1, col].set_ylim(-5.0, 5.0)
        axes[3, col].set_ylim(-1.1, 1.1)


def main() -> None:
    args = parse_args()
    model_path = require_file(args.model, "MuJoCo model")
    gait_policy_path = require_file(args.gait_policy, "Gait TorchScript policy")
    assist_policy_path = require_file(args.assist_policy, "Assist TorchScript policy")
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
        raise ValueError("Assist policy must map 150 observations to 2 actions")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    indices = build_indices(mujoco, model)
    model.dof_armature[indices["policy_qvel"]] = 0.01
    data = mujoco.MjData(model)
    history = AssistHistory()
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
        else assist_policy_path.parent.parent / "sim2sim_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "g1_assist_exoskeleton_sim2sim.csv"
    png_path = output_dir / "g1_assist_exoskeleton_sim2sim.png"
    signal_names = ["time"] + [
        f"{side}_{signal}"
        for side in SIDES
        for signal in ("hip_pos", "assist_pos", "pos_error", "hip_vel", "assist_vel", "hip_torque", "assist_torque")
    ]
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(signal_names)

    max_plot_points = max(100, math.ceil(max(args.plot_window, 1.0) / (SIM_DT * DECIMATION)) * 2)
    plot_data = {name: deque(maxlen=max_plot_points) for name in signal_names}
    fig, axes, lines = create_plot()
    live_plot = not args.headless and not args.no_live_plot
    if live_plot:
        plt.ion()
        fig.show()

    def key_callback(keycode: int) -> None:
        nonlocal reset_requested
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "W": command.vx += 0.1
        elif key == "S": command.vx -= 0.1
        elif key == "A": command.vy += 0.1
        elif key == "D": command.vy -= 0.1
        elif key == "Q": command.yaw += 0.1
        elif key == "E": command.yaw -= 0.1
        elif key == " ": command.zero()
        elif key == "R": reset_requested = True
        else: return
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
    print(f"Assist policy: {assist_policy_path}")
    print(f"Control: dt={SIM_DT}, decimation={DECIMATION}, policy rate=50 Hz")
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
                # Isaac evaluates terminations once per environment/control step,
                # not at every 1 ms physics substep. Check at the same 50 Hz rate.
                angle_error_deg = np.rad2deg(
                    data.qpos[indices["assist_qpos"]] - data.qpos[indices["hip_qpos"]]
                )
                angle_reset = (
                    args.auto_reset_angle_deg > 0.0
                    and np.max(np.abs(angle_error_deg)) > args.auto_reset_angle_deg
                )
                height_reset = (
                    args.auto_reset_height > 0.0 and float(data.qpos[2]) < args.auto_reset_height
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
                    assist_action_tensor = assist_policy(torch.from_numpy(assist_obs).unsqueeze(0))
                last_gait_action = gait_action_tensor.squeeze(0).numpy().astype(np.float64)
                assist_action = np.clip(assist_action_tensor.squeeze(0).numpy(), -1.0, 1.0)
                gait_target = DEFAULT_JOINT_POS + ACTION_SCALE * last_gait_action
                assist_torque = ASSIST_ACTION_SCALE * assist_action.astype(np.float64)
                control_step += 1

            joint_pos = data.qpos[indices["policy_qpos"]]
            joint_vel = data.qvel[indices["policy_qvel"]]
            gait_torque = np.clip(KP * (gait_target - joint_pos) - KD * joint_vel, -EFFORT_LIMIT, EFFORT_LIMIT)
            data.ctrl[indices["policy_actuator"]] = gait_torque
            data.ctrl[indices["assist_actuator"]] = np.clip(assist_torque, -ASSIST_ACTION_SCALE, ASSIST_ACTION_SCALE)
            mujoco.mj_step(model, data)

            if step % DECIMATION == 0:
                hip_pos = np.rad2deg(data.qpos[indices["hip_qpos"]])
                assist_pos = np.rad2deg(data.qpos[indices["assist_qpos"]])
                hip_vel = np.rad2deg(data.qvel[indices["hip_qvel"]])
                assist_vel = np.rad2deg(data.qvel[indices["assist_qvel"]])
                hip_torque = np.asarray(data.actuator_force[indices["hip_actuator"]]).copy()
                actual_assist_torque = np.asarray(data.actuator_force[indices["assist_actuator"]]).copy()
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
        fig.suptitle(f"G1 Assist Exoskeleton MuJoCo Sim2Sim ({simulated_time:.2f} s)", y=1.002)
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        csv_file.close()
        if viewer is not None:
            viewer.close()
        plt.ioff()
        plt.close(fig)

    wall_time = time.perf_counter() - started
    print(
        f"Finished: simulated={step * SIM_DT:.2f}s wall={wall_time:.2f}s "
        f"position=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) resets={resets}"
    )
    print(f"CSV:  {csv_path}")
    print(f"Plot: {png_path}")


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play one G1 assist-exoskeleton robot and plot joint measurements."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


DEFAULT_TASK = "LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0"

parser = argparse.ArgumentParser(description="Single-robot exoskeleton play with CSV logging and plots.")
parser.add_argument("--video", action="store_true", default=False, help="Record an Isaac Sim video.")
parser.add_argument("--video_length", type=int, default=1000, help="Recorded video length in control steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of Fabric.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Registered play task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent configuration registry entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Try to run at wall-clock speed.")
parser.add_argument(
    "--duration", type=float, default=None, help="Stop after this many simulated seconds; default runs until closed."
)
parser.add_argument("--plot_window", type=float, default=5.0, help="Seconds shown in the live plot.")
parser.add_argument("--plot_interval", type=float, default=0.05, help="Live-plot refresh interval in seconds.")
parser.add_argument("--no_live_plot", action="store_true", help="Disable live refresh; CSV and final PNG are still saved.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory; defaults inside checkpoint run.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""The remaining imports require Isaac Sim to be running."""

import csv
import math
import os
import time
from collections import deque

import gymnasium as gym
import torch

import matplotlib

if args_cli.headless or args_cli.no_live_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import legged_lab.tasks  # noqa: F401


HIP_JOINT_NAMES = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
ASSIST_JOINT_NAMES = ["left_hip_pitch_assist_joint", "right_hip_pitch_assist_joint"]
SIDES = ("left", "right")


def _ordered_ring(buffer: deque) -> np.ndarray:
    return np.asarray(buffer, dtype=np.float64)


def _create_figure():
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex="col")
    lines = {}
    for col, side in enumerate(SIDES):
        ax = axes[0, col]
        lines[f"{side}_hip_pos"], = ax.plot([], [], label="hip pitch", color="tab:blue")
        lines[f"{side}_assist_pos"], = ax.plot([], [], label="assist", color="tab:orange")
        ax.set_title(f"{side.capitalize()} joint angle")
        ax.set_ylabel("Angle (deg)")
        ax.legend(loc="upper right")

        ax = axes[1, col]
        lines[f"{side}_pos_error"], = ax.plot([], [], label="assist - hip", color="tab:purple")
        ax.axhline(4.0, color="tab:red", linestyle="--", linewidth=1.0, label="termination ±4°")
        ax.axhline(-4.0, color="tab:red", linestyle="--", linewidth=1.0)
        ax.set_title(f"{side.capitalize()} angle difference")
        ax.set_ylabel("Difference (deg)")
        ax.legend(loc="upper right")

        ax = axes[2, col]
        lines[f"{side}_hip_vel"], = ax.plot([], [], label="hip pitch", color="tab:blue")
        lines[f"{side}_assist_vel"], = ax.plot([], [], label="assist", color="tab:orange")
        ax.set_title(f"{side.capitalize()} joint velocity")
        ax.set_ylabel("Velocity (deg/s)")
        ax.legend(loc="upper right")

        ax = axes[3, col]
        lines[f"{side}_hip_torque"], = ax.plot([], [], label="hip pitch", color="tab:blue")
        lines[f"{side}_assist_torque"], = ax.plot([], [], label="assist", color="tab:green")
        ax.set_title(f"{side.capitalize()} actuator torque")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque (N.m)")
        ax.legend(loc="upper right")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="gray", linewidth=0.5)
    fig.tight_layout()
    return fig, axes, lines


def _update_plot(axes, lines, data, plot_window: float):
    time_data = _ordered_ring(data["time"])
    if time_data.size < 2:
        return
    mask = time_data >= max(0.0, time_data[-1] - plot_window)
    time_view = time_data[mask]

    for side in SIDES:
        for signal in ("hip_pos", "assist_pos", "pos_error", "hip_vel", "assist_vel", "hip_torque", "assist_torque"):
            key = f"{side}_{signal}"
            lines[key].set_data(time_view, _ordered_ring(data[key])[mask])

    for ax in axes.flat:
        ax.set_xlim(time_view[0], time_view[-1] + 0.02)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
    # Preserve a useful view of the safety threshold even when error is small.
    for col in range(2):
        axes[1, col].set_ylim(-5.0, 5.0)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load one environment, run the policy, and collect/plot both joint pairs."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    output_dir = os.path.abspath(args_cli.output_dir) if args_cli.output_dir else os.path.join(log_dir, "play_analysis")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "g1_assist_exoskeleton_play.csv")
    png_path = os.path.join(output_dir, "g1_assist_exoskeleton_play.png")

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(output_dir, "video"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    hip_ids, hip_names = robot.find_joints(HIP_JOINT_NAMES, preserve_order=True)
    assist_ids, assist_names = robot.find_joints(ASSIST_JOINT_NAMES, preserve_order=True)
    if hip_names != HIP_JOINT_NAMES or assist_names != ASSIST_JOINT_NAMES:
        raise RuntimeError(f"Joint resolution failed: hip={hip_names}, assist={assist_names}")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=raw_env.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    max_plot_points = max(100, math.ceil(max(args_cli.plot_window, 1.0) / raw_env.step_dt) * 2)
    signal_names = ["time"] + [
        f"{side}_{signal}"
        for side in SIDES
        for signal in ("hip_pos", "assist_pos", "pos_error", "hip_vel", "assist_vel", "hip_torque", "assist_torque")
    ]
    data = {name: deque(maxlen=max_plot_points) for name in signal_names}
    fig, axes, lines = _create_figure()
    live_plot = not args_cli.headless and not args_cli.no_live_plot
    if live_plot:
        plt.ion()
        fig.show()

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(signal_names)
    obs = env.get_observations()
    sim_time = 0.0
    last_plot_time = -args_cli.plot_interval
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            hip_pos = torch.rad2deg(robot.data.joint_pos[0, hip_ids]).cpu().numpy()
            assist_pos = torch.rad2deg(robot.data.joint_pos[0, assist_ids]).cpu().numpy()
            hip_vel = torch.rad2deg(robot.data.joint_vel[0, hip_ids]).cpu().numpy()
            assist_vel = torch.rad2deg(robot.data.joint_vel[0, assist_ids]).cpu().numpy()
            hip_torque = robot.data.applied_torque[0, hip_ids].cpu().numpy()
            assist_torque = robot.data.applied_torque[0, assist_ids].cpu().numpy()
            pos_error = assist_pos - hip_pos

            row = [sim_time]
            data["time"].append(sim_time)
            for index, side in enumerate(SIDES):
                values = {
                    "hip_pos": hip_pos[index],
                    "assist_pos": assist_pos[index],
                    "pos_error": pos_error[index],
                    "hip_vel": hip_vel[index],
                    "assist_vel": assist_vel[index],
                    "hip_torque": hip_torque[index],
                    "assist_torque": assist_torque[index],
                }
                for signal, value in values.items():
                    scalar = float(value)
                    data[f"{side}_{signal}"].append(scalar)
                    row.append(scalar)
            csv_writer.writerow(row)

            timestep += 1
            sim_time += raw_env.step_dt
            if live_plot and sim_time - last_plot_time >= args_cli.plot_interval:
                _update_plot(axes, lines, data, args_cli.plot_window)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_plot_time = sim_time

            if args_cli.duration is not None and sim_time >= args_cli.duration:
                break
            if args_cli.video and timestep >= args_cli.video_length:
                break
            sleep_time = raw_env.step_dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        _update_plot(axes, lines, data, max(sim_time, args_cli.plot_window))
        fig.suptitle(f"G1 Assist Exoskeleton Play ({sim_time:.2f} s)", y=1.002)
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        csv_file.close()
        env.close()
        plt.ioff()
        plt.close(fig)
        print(f"[INFO] CSV saved to: {csv_path}")
        print(f"[INFO] Plot saved to: {png_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

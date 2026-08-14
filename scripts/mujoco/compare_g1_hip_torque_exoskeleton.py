#!/usr/bin/env python3
"""Compare G1 hip-pitch mechanical power with and without the assist exoskeleton."""

from __future__ import annotations

import argparse
import csv
import math
import time
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
    build_indices as build_baseline_indices,
    make_observation,
    require_file,
    reset_robot as reset_baseline_robot,
)
from sim2sim_g1_assist_exoskeleton import (
    ASSIST_ACTION_SCALE,
    ASSIST_OBSERVATIONS,
    DECIMATION,
    SIM_DT,
    AssistHistory,
    build_indices as build_exoskeleton_indices,
    reset_robot as reset_exoskeleton_robot,
)


DEFAULT_BASELINE_MODEL = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/g1_29dof_assist.xml"
)
DEFAULT_EXOSKELETON_MODEL = (
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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_ppo/2026-08-14_11-06-58/hip_power_comparison"
)
SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--exoskeleton-model", type=Path, default=DEFAULT_EXOSKELETON_MODEL)
    parser.add_argument("--gait-policy", type=Path, default=DEFAULT_GAIT_POLICY)
    parser.add_argument("--assist-policy", type=Path, default=DEFAULT_ASSIST_POLICY)
    parser.add_argument("--vx", type=float, default=0.7, help="Forward command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral command in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw-rate command in rad/s.")
    parser.add_argument("--duration", type=float, default=10.0, help="Compared simulation duration in seconds.")
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Initial seconds excluded from summary metrics but retained in plots/CSV.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--realtime", action="store_true", help="Pace both simulations to wall-clock time.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if args.warmup < 0.0 or args.warmup >= args.duration:
        raise ValueError("--warmup must satisfy 0 <= warmup < duration")


def load_policies(torch, gait_path: Path, assist_path: Path):
    gait_policy = torch.jit.load(str(gait_path), map_location="cpu")
    assist_policy = torch.jit.load(str(assist_path), map_location="cpu")
    gait_policy.eval()
    assist_policy.eval()
    with torch.inference_mode():
        gait_output = gait_policy(torch.zeros(1, NUM_OBSERVATIONS))
        assist_output = assist_policy(torch.zeros(1, ASSIST_OBSERVATIONS))
    if tuple(gait_output.shape) != (1, NUM_ACTIONS):
        raise ValueError(f"Expected gait policy output (1, 29), got {tuple(gait_output.shape)}")
    if tuple(assist_output.shape) != (1, 2):
        raise ValueError(f"Expected assist policy output (1, 2), got {tuple(assist_output.shape)}")
    return gait_policy, assist_policy


def compute_statistics(time_values, baseline_power, exoskeleton_power, warmup: float):
    time_values = np.asarray(time_values)
    baseline_power = np.asarray(baseline_power)
    exoskeleton_power = np.asarray(exoskeleton_power)
    mask = time_values >= warmup
    if not np.any(mask):
        raise RuntimeError("No samples remain after the warm-up interval")

    compared_time = time_values[mask]

    stats = {}
    for index, side in enumerate(SIDES):
        baseline = baseline_power[mask, index]
        exoskeleton = exoskeleton_power[mask, index]
        side_stats = {}
        for name, values in (
            ("baseline_hip", baseline),
            ("exoskeleton_hip", exoskeleton),
        ):
            positive = np.maximum(values, 0.0)
            negative = np.minimum(values, 0.0)
            side_stats[name] = {
                "mean_net_power": float(np.mean(values)),
                "mean_positive_power": float(np.mean(positive)),
                "mean_negative_power": float(np.mean(negative)),
                "positive_energy": float(np.trapz(positive, compared_time)),
                "negative_energy": float(np.trapz(negative, compared_time)),
                "net_energy": float(np.trapz(values, compared_time)),
                "peak_positive_power": float(np.max(values)),
                "peak_negative_power": float(np.min(values)),
            }
        side_stats["hip_reduction_percent"] = {}
        for metric in ("mean_positive_power", "positive_energy", "peak_positive_power"):
            base_value = side_stats["baseline_hip"][metric]
            exo_value = side_stats["exoskeleton_hip"][metric]
            side_stats["hip_reduction_percent"][metric] = (
                100.0 * (base_value - exo_value) / base_value if base_value > 1.0e-8 else float("nan")
            )
        stats[side] = side_stats

    stats["bilateral"] = {}
    aggregate_metrics = (
        "mean_net_power",
        "mean_positive_power",
        "mean_negative_power",
        "positive_energy",
        "negative_energy",
        "net_energy",
    )
    for condition in ("baseline_hip", "exoskeleton_hip"):
        stats["bilateral"][condition] = {
            metric: sum(stats[side][condition][metric] for side in SIDES)
            for metric in aggregate_metrics
        }
    baseline_positive = stats["bilateral"]["baseline_hip"]["mean_positive_power"]
    exoskeleton_positive = stats["bilateral"]["exoskeleton_hip"]["mean_positive_power"]
    stats["bilateral"]["hip_positive_power_reduction_percent"] = (
        100.0 * (baseline_positive - exoskeleton_positive) / baseline_positive
    )
    return stats


def save_summary(path: Path, stats, args: argparse.Namespace) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "side", "metric", "value", "unit"])
        writer.writerow(["settings", "both", "vx", args.vx, "m/s"])
        writer.writerow(["settings", "both", "vy", args.vy, "m/s"])
        writer.writerow(["settings", "both", "yaw", args.yaw, "rad/s"])
        writer.writerow(["settings", "both", "duration", args.duration, "s"])
        writer.writerow(["settings", "both", "warmup", args.warmup, "s"])
        power_units = {
            "mean_net_power": "W",
            "mean_positive_power": "W",
            "mean_negative_power": "W",
            "positive_energy": "J",
            "negative_energy": "J",
            "net_energy": "J",
            "peak_positive_power": "W",
            "peak_negative_power": "W",
        }
        for side in SIDES:
            for condition in ("baseline_hip", "exoskeleton_hip"):
                for metric, value in stats[side][condition].items():
                    writer.writerow([condition, side, metric, value, power_units[metric]])
            for metric, value in stats[side]["hip_reduction_percent"].items():
                writer.writerow(["hip_reduction", side, metric, value, "%"])
        for condition in ("baseline_hip", "exoskeleton_hip"):
            for metric, value in stats["bilateral"][condition].items():
                writer.writerow([condition, "bilateral", metric, value, power_units[metric]])
        writer.writerow(
            [
                "hip_reduction",
                "bilateral",
                "mean_positive_power",
                stats["bilateral"]["hip_positive_power_reduction_percent"],
                "%",
            ]
        )


def create_plot(
    time_values,
    baseline_power,
    exoskeleton_power,
    stats,
    warmup: float,
    output_path: Path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_values = np.asarray(time_values)
    baseline_power = np.asarray(baseline_power)
    exoskeleton_power = np.asarray(exoskeleton_power)
    metrics = ("mean_positive_power", "positive_energy", "peak_positive_power")
    metric_labels = ("Mean positive\npower (W)", "Positive\nenergy (J)", "Peak positive\npower (W)")

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex="row")
    for index, side in enumerate(SIDES):
        ax = axes[0, index]
        ax.plot(time_values, baseline_power[:, index], label="without exoskeleton", linewidth=1.0)
        ax.plot(time_values, exoskeleton_power[:, index], label="with exoskeleton", linewidth=1.0)
        ax.axvline(warmup, color="gray", linestyle="--", label="metrics start")
        ax.set_title(f"{side.capitalize()} hip-pitch actuator mechanical power")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right")

        ax = axes[1, index]
        ax.plot(
            time_values,
            np.maximum(baseline_power[:, index], 0.0),
            label="without exoskeleton",
            linewidth=1.0,
        )
        ax.plot(
            time_values,
            np.maximum(exoskeleton_power[:, index], 0.0),
            label="with exoskeleton",
            linewidth=1.0,
        )
        ax.axvline(warmup, color="gray", linestyle="--")
        ax.set_title(f"{side.capitalize()} hip-pitch positive mechanical power")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right")

        ax = axes[2, index]
        x = np.arange(len(metrics))
        width = 0.36
        baseline_values = [stats[side]["baseline_hip"][metric] for metric in metrics]
        exoskeleton_values = [stats[side]["exoskeleton_hip"][metric] for metric in metrics]
        ax.bar(x - width / 2, baseline_values, width, label="without exoskeleton")
        ax.bar(x + width / 2, exoskeleton_values, width, label="with exoskeleton")
        ax.set_xticks(x, metric_labels)
        ax.set_ylabel("Power / energy")
        ax.set_title(f"{side.capitalize()} summary after {warmup:.1f} s")
        ax.legend(loc="upper right")

        for row in range(3):
            axes[row, index].grid(True, alpha=0.3)
            axes[row, index].axhline(0.0, color="gray", linewidth=0.5)

    fig.suptitle("G1 Body Hip-Pitch Power: Without vs With Assist Exoskeleton")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    baseline_model_path = require_file(args.baseline_model, "Baseline MuJoCo model")
    exoskeleton_model_path = require_file(args.exoskeleton_model, "Exoskeleton MuJoCo model")
    gait_policy_path = require_file(args.gait_policy, "Gait policy")
    assist_policy_path = require_file(args.assist_policy, "Assist policy")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import mujoco
    import torch

    gait_policy, assist_policy = load_policies(torch, gait_policy_path, assist_policy_path)

    baseline_model = mujoco.MjModel.from_xml_path(str(baseline_model_path))
    exoskeleton_model = mujoco.MjModel.from_xml_path(str(exoskeleton_model_path))
    baseline_model.opt.timestep = SIM_DT
    exoskeleton_model.opt.timestep = SIM_DT

    baseline_qpos, baseline_qvel, baseline_actuator, baseline_pelvis, baseline_gyro = (
        build_baseline_indices(mujoco, baseline_model)
    )
    exoskeleton_indices = build_exoskeleton_indices(mujoco, exoskeleton_model)
    baseline_model.dof_armature[baseline_qvel] = 0.01
    exoskeleton_model.dof_armature[exoskeleton_indices["policy_qvel"]] = 0.01

    baseline_data = mujoco.MjData(baseline_model)
    exoskeleton_data = mujoco.MjData(exoskeleton_model)
    reset_baseline_robot(mujoco, baseline_model, baseline_data, baseline_qpos)
    assist_history = AssistHistory()
    reset_exoskeleton_robot(
        mujoco, exoskeleton_model, exoskeleton_data, exoskeleton_indices, assist_history
    )

    command = VelocityCommand(args.vx, args.vy, args.yaw)
    command.clamp()
    baseline_last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    exoskeleton_last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    baseline_target = DEFAULT_JOINT_POS.copy()
    exoskeleton_target = DEFAULT_JOINT_POS.copy()
    assist_torque = np.zeros(2, dtype=np.float64)

    time_values = []
    baseline_torque_values = []
    exoskeleton_torque_values = []
    baseline_velocity_values = []
    exoskeleton_velocity_values = []
    baseline_height_values = []
    exoskeleton_height_values = []
    baseline_xy_values = []
    exoskeleton_xy_values = []

    step = 0
    control_step = 0
    total_steps = math.ceil(args.duration / SIM_DT)
    started = time.perf_counter()
    while step < total_steps:
        step_started = time.perf_counter()
        if step % DECIMATION == 0:
            if control_step > 0:
                assist_history.append(
                    exoskeleton_data.qpos[exoskeleton_indices["assist_qpos"]],
                    exoskeleton_data.qvel[exoskeleton_indices["assist_qvel"]],
                    assist_torque,
                )

            baseline_obs = make_observation(
                baseline_data,
                baseline_qpos,
                baseline_qvel,
                baseline_pelvis,
                baseline_gyro,
                command,
                baseline_last_action,
            )
            exoskeleton_obs = make_observation(
                exoskeleton_data,
                exoskeleton_indices["policy_qpos"],
                exoskeleton_indices["policy_qvel"],
                exoskeleton_indices["pelvis"],
                exoskeleton_indices["gyro_adr"],
                command,
                exoskeleton_last_action,
            )
            assist_obs = assist_history.observation()
            with torch.inference_mode():
                baseline_action_tensor = gait_policy(torch.from_numpy(baseline_obs).unsqueeze(0))
                exoskeleton_action_tensor = gait_policy(torch.from_numpy(exoskeleton_obs).unsqueeze(0))
                assist_action_tensor = assist_policy(torch.from_numpy(assist_obs).unsqueeze(0))
            baseline_last_action = baseline_action_tensor.squeeze(0).numpy().astype(np.float64)
            exoskeleton_last_action = exoskeleton_action_tensor.squeeze(0).numpy().astype(np.float64)
            assist_action = np.clip(assist_action_tensor.squeeze(0).numpy(), -1.0, 1.0)
            baseline_target = DEFAULT_JOINT_POS + ACTION_SCALE * baseline_last_action
            exoskeleton_target = DEFAULT_JOINT_POS + ACTION_SCALE * exoskeleton_last_action
            assist_torque = ASSIST_ACTION_SCALE * assist_action.astype(np.float64)
            control_step += 1

        baseline_torque = np.clip(
            KP * (baseline_target - baseline_data.qpos[baseline_qpos])
            - KD * baseline_data.qvel[baseline_qvel],
            -EFFORT_LIMIT,
            EFFORT_LIMIT,
        )
        exoskeleton_torque = np.clip(
            KP * (exoskeleton_target - exoskeleton_data.qpos[exoskeleton_indices["policy_qpos"]])
            - KD * exoskeleton_data.qvel[exoskeleton_indices["policy_qvel"]],
            -EFFORT_LIMIT,
            EFFORT_LIMIT,
        )
        baseline_data.ctrl[baseline_actuator] = baseline_torque
        exoskeleton_data.ctrl[exoskeleton_indices["policy_actuator"]] = exoskeleton_torque
        exoskeleton_data.ctrl[exoskeleton_indices["assist_actuator"]] = np.clip(
            assist_torque, -ASSIST_ACTION_SCALE, ASSIST_ACTION_SCALE
        )
        mujoco.mj_step(baseline_model, baseline_data)
        mujoco.mj_step(exoskeleton_model, exoskeleton_data)

        # Sample at the physics rate. Mechanical joint power is P = torque * angular velocity.
        time_values.append((step + 1) * SIM_DT)
        baseline_torque_values.append(
            np.asarray(baseline_data.actuator_force[baseline_actuator[[0, 1]]]).copy()
        )
        exoskeleton_torque_values.append(
            np.asarray(exoskeleton_data.actuator_force[exoskeleton_indices["hip_actuator"]]).copy()
        )
        baseline_velocity_values.append(np.asarray(baseline_data.qvel[baseline_qvel[[0, 1]]]).copy())
        exoskeleton_velocity_values.append(
            np.asarray(exoskeleton_data.qvel[exoskeleton_indices["hip_qvel"]]).copy()
        )
        baseline_height_values.append(float(baseline_data.qpos[2]))
        exoskeleton_height_values.append(float(exoskeleton_data.qpos[2]))
        baseline_xy_values.append(np.asarray(baseline_data.qpos[:2]).copy())
        exoskeleton_xy_values.append(np.asarray(exoskeleton_data.qpos[:2]).copy())

        if args.realtime:
            remaining = SIM_DT - (time.perf_counter() - step_started)
            if remaining > 0.0:
                time.sleep(remaining)
        step += 1

    time_values = np.asarray(time_values)
    baseline_torque_values = np.asarray(baseline_torque_values)
    exoskeleton_torque_values = np.asarray(exoskeleton_torque_values)
    baseline_velocity_values = np.asarray(baseline_velocity_values)
    exoskeleton_velocity_values = np.asarray(exoskeleton_velocity_values)
    baseline_power_values = baseline_torque_values * baseline_velocity_values
    exoskeleton_power_values = exoskeleton_torque_values * exoskeleton_velocity_values
    baseline_xy_values = np.asarray(baseline_xy_values)
    exoskeleton_xy_values = np.asarray(exoskeleton_xy_values)

    csv_path = output_dir / "hip_pitch_power_timeseries.csv"
    summary_path = output_dir / "hip_pitch_power_summary.csv"
    plot_path = output_dir / "hip_pitch_power_comparison.png"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time",
                "baseline_left_hip_torque",
                "baseline_left_hip_velocity",
                "baseline_left_hip_power",
                "exoskeleton_left_hip_torque",
                "exoskeleton_left_hip_velocity",
                "exoskeleton_left_hip_power",
                "baseline_right_hip_torque",
                "baseline_right_hip_velocity",
                "baseline_right_hip_power",
                "exoskeleton_right_hip_torque",
                "exoskeleton_right_hip_velocity",
                "exoskeleton_right_hip_power",
                "baseline_height",
                "exoskeleton_height",
                "baseline_x",
                "baseline_y",
                "exoskeleton_x",
                "exoskeleton_y",
            ]
        )
        for index, sample_time in enumerate(time_values):
            writer.writerow(
                [
                    sample_time,
                    baseline_torque_values[index, 0],
                    baseline_velocity_values[index, 0],
                    baseline_power_values[index, 0],
                    exoskeleton_torque_values[index, 0],
                    exoskeleton_velocity_values[index, 0],
                    exoskeleton_power_values[index, 0],
                    baseline_torque_values[index, 1],
                    baseline_velocity_values[index, 1],
                    baseline_power_values[index, 1],
                    exoskeleton_torque_values[index, 1],
                    exoskeleton_velocity_values[index, 1],
                    exoskeleton_power_values[index, 1],
                    baseline_height_values[index],
                    exoskeleton_height_values[index],
                    baseline_xy_values[index, 0],
                    baseline_xy_values[index, 1],
                    exoskeleton_xy_values[index, 0],
                    exoskeleton_xy_values[index, 1],
                ]
            )

    stats = compute_statistics(
        time_values,
        baseline_power_values,
        exoskeleton_power_values,
        args.warmup,
    )
    save_summary(summary_path, stats, args)
    create_plot(
        time_values,
        baseline_power_values,
        exoskeleton_power_values,
        stats,
        args.warmup,
        plot_path,
    )

    wall_time = time.perf_counter() - started
    print(
        f"Finished equal-condition comparison: simulated={args.duration:.2f}s "
        f"wall={wall_time:.2f}s command=({command.vx:.2f},{command.vy:.2f},{command.yaw:.2f})"
    )
    for side in SIDES:
        base = stats[side]["baseline_hip"]
        exo = stats[side]["exoskeleton_hip"]
        reduction = stats[side]["hip_reduction_percent"]
        print(
            f"{side:>5}: hip mean positive power "
            f"{base['mean_positive_power']:.3f} -> {exo['mean_positive_power']:.3f} W "
            f"({reduction['mean_positive_power']:+.2f}% reduction)"
        )
        print(
            f"       hip positive energy {base['positive_energy']:.3f} -> "
            f"{exo['positive_energy']:.3f} J; net power "
            f"{base['mean_net_power']:.3f} -> {exo['mean_net_power']:.3f} W"
        )
    bilateral = stats["bilateral"]
    baseline_positive = bilateral["baseline_hip"]["mean_positive_power"]
    exoskeleton_positive = bilateral["exoskeleton_hip"]["mean_positive_power"]
    print(
        f" both: hip mean positive power {baseline_positive:.3f} -> "
        f"{exoskeleton_positive:.3f} W "
        f"({bilateral['hip_positive_power_reduction_percent']:+.2f}% reduction)"
    )
    print(
        f"Final x: baseline={baseline_xy_values[-1, 0]:.3f} m, "
        f"exoskeleton={exoskeleton_xy_values[-1, 0]:.3f} m"
    )
    print(f"Time series: {csv_path}")
    print(f"Summary:     {summary_path}")
    print(f"Plot:        {plot_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the G1 AMP v3 staged-PID policy in MuJoCo.

This runner mirrors the Isaac configuration used by
``LeggedLab-Isaac-AMP-G1-v3``: a 96-observation/29-action policy, explicit
per-physics-step PID with torque-bounded integral and conditional anti-windup,
two unactuated hip-assist joints, and PD-controlled rear mechanism joints.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    "g1_29dof_assist_exoskeleton_2.xml"
)
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_amp_v3/2026-09-03_13-48-47_pid_curriculum_30000/"
    "exported/policy.pt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_amp_v3/2026-09-03_13-48-47_pid_curriculum_30000/"
    "sim2sim_model_7000"
)

# Mirrors g1_v3/robot_cfg.py in POLICY_JOINT_NAMES order.
KI = np.array(
    [
        18.0, 18.0, 36.0, 18.0, 18.0, 7.2, 18.0, 18.0, 7.2,
        27.0, 27.0, 5.6, 5.6, 7.2, 7.2, 5.6, 5.6, 7.2, 7.2,
        5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
    ],
    dtype=np.float64,
)
INTEGRAL_EFFORT_LIMIT = 10.0
CURRICULUM_START = 5000
CURRICULUM_END = 10000

FREE_ASSIST_JOINT_NAMES = (
    "left_hip_pitch_assist_joint",
    "right_hip_pitch_assist_joint",
)
FREE_ASSIST_INITIAL_POS = np.array((-0.1, -0.1), dtype=np.float64)
REAR_JOINT_NAMES = (
    "pelvis_rear_upper_box_assist_joint",
    "pelvis_rear_cylinder_assist_joint",
)
REAR_TARGET = np.zeros(2, dtype=np.float64)
REAR_KP = np.array((10.0, 10.0), dtype=np.float64)
REAR_KD = np.array((1.0, 1.0), dtype=np.float64)
REAR_EFFORT_LIMIT = np.array((300.0, 200.0), dtype=np.float64)

# Match Isaac training exactly: 1 ms physics, 20 ms policy period.
SIM_DT = 0.001
DECIMATION = 20


def curriculum_alpha(iteration: int) -> float:
    """Return the staged integral multiplier for a checkpoint iteration."""

    return float(np.clip((iteration - CURRICULUM_START) / (CURRICULUM_END - CURRICULUM_START), 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-iteration", type=int, default=7000)
    parser.add_argument(
        "--pid-alpha",
        type=float,
        default=None,
        help="Override staged integral alpha; otherwise derive it from checkpoint iteration.",
    )
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--auto-reset-height", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _joint_indices(mujoco, model, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
        dtype=int,
    )
    actuator_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names],
        dtype=int,
    )
    return (
        model.jnt_qposadr[joint_ids].astype(int),
        model.jnt_dofadr[joint_ids].astype(int),
        actuator_ids,
    )


def build_indices(mujoco, model) -> dict[str, object]:
    policy_qpos, policy_qvel, policy_actuator = _joint_indices(mujoco, model, POLICY_JOINT_NAMES)
    free_qpos, free_qvel, free_actuator = _joint_indices(mujoco, model, FREE_ASSIST_JOINT_NAMES)
    rear_qpos, rear_qvel, rear_actuator = _joint_indices(mujoco, model, REAR_JOINT_NAMES)
    pelvis = name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    gyro = name_to_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity"
    )
    if int(model.sensor_dim[gyro]) != 3:
        raise ValueError("Pelvis gyro must have dimension 3")
    return {
        "policy_qpos": policy_qpos,
        "policy_qvel": policy_qvel,
        "policy_actuator": policy_actuator,
        "free_qpos": free_qpos,
        "free_qvel": free_qvel,
        "free_actuator": free_actuator,
        "rear_qpos": rear_qpos,
        "rear_qvel": rear_qvel,
        "rear_actuator": rear_actuator,
        "pelvis": pelvis,
        "gyro_adr": int(model.sensor_adr[gyro]),
    }


def _configure_effort_limits(model, actuator_ids: np.ndarray, limits: np.ndarray) -> None:
    joint_ids = model.actuator_trnid[actuator_ids, 0].astype(int)
    if np.any(joint_ids < 0):
        raise ValueError("Every controlled actuator must be attached to a joint")
    ranges = np.column_stack((-limits, limits))
    model.jnt_actfrclimited[joint_ids] = 1
    model.jnt_actfrcrange[joint_ids] = ranges
    model.actuator_ctrllimited[actuator_ids] = 1
    model.actuator_ctrlrange[actuator_ids] = ranges


def reset_robot(mujoco, model, data, indices: dict[str, object]) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[indices["policy_qpos"]] = DEFAULT_JOINT_POS
    data.qpos[indices["free_qpos"]] = FREE_ASSIST_INITIAL_POS
    data.qpos[indices["rear_qpos"]] = REAR_TARGET
    data.qvel[indices["free_qvel"]] = 0.0
    data.qvel[indices["rear_qvel"]] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def pid_torque(
    target: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    integral_effort: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror IdealPIDActuator.compute, including conditional anti-windup."""

    error = target - position
    integral_limit = INTEGRAL_EFFORT_LIMIT * alpha
    candidate_integral = np.clip(
        integral_effort + alpha * KI * error * SIM_DT,
        -integral_limit,
        integral_limit,
    )
    candidate_effort = KP * error - KD * velocity + candidate_integral
    candidate_applied = np.clip(candidate_effort, -EFFORT_LIMIT, EFFORT_LIMIT)
    saturated = candidate_effort != candidate_applied
    pushes_further = error * candidate_effort > 0.0
    accept_integral = ~(saturated & pushes_further)
    updated_integral = np.where(accept_integral, candidate_integral, integral_effort)
    effort = np.clip(KP * error - KD * velocity + updated_integral, -EFFORT_LIMIT, EFFORT_LIMIT)
    return effort, updated_integral


def main() -> None:
    args = parse_args()
    model_path = require_file(args.model, "MuJoCo model")
    policy_path = require_file(args.policy, "TorchScript policy")
    if args.headless and args.duration is None:
        args.duration = 10.0
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    alpha = curriculum_alpha(args.checkpoint_iteration) if args.pid_alpha is None else args.pid_alpha
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("PID alpha must be finite and between 0 and 1")

    import mujoco
    import torch

    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    with torch.inference_mode():
        test_output = policy(torch.zeros((1, NUM_OBSERVATIONS), dtype=torch.float32))
    if tuple(test_output.shape) != (1, NUM_ACTIONS):
        raise ValueError(
            f"Policy shape mismatch: expected (1, {NUM_ACTIONS}), got {tuple(test_output.shape)}"
        )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    indices = build_indices(mujoco, model)
    _configure_effort_limits(model, indices["policy_actuator"], EFFORT_LIMIT)
    _configure_effort_limits(model, indices["rear_actuator"], REAR_EFFORT_LIMIT)
    _configure_effort_limits(model, indices["free_actuator"], np.full(2, 8.0))
    model.dof_armature[indices["policy_qvel"]] = 0.01
    data = mujoco.MjData(model)
    reset_robot(mujoco, model, data, indices)

    command = VelocityCommand(args.vx, args.vy, args.yaw)
    command.clamp()
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    target = DEFAULT_JOINT_POS.copy()
    integral_effort = np.zeros(NUM_ACTIONS, dtype=np.float64)
    reset_requested = False

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "g1_pid_v3_sim2sim.csv"
    summary_path = output_dir / "summary.json"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "time", "x", "y", "height", "vx_body", "vy_body", "yaw_rate",
            "error_vel_xy", "error_vel_yaw", "max_abs_torque",
            "max_abs_integral_torque", "resets",
        ]
    )

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

    print(f"Model:      {model_path}")
    print(f"Policy:     {policy_path}")
    print(f"PID:        checkpoint_iteration={args.checkpoint_iteration}, alpha={alpha:.4f}")
    print(f"Control:    dt={SIM_DT}, decimation={DECIMATION}, policy_rate=50 Hz")
    print(f"Command:    vx={command.vx:.2f}, vy={command.vy:.2f}, yaw={command.yaw:.2f}")
    print(f"Output:     {output_dir}")

    step = 0
    resets = 0
    error_xy_samples: list[float] = []
    error_yaw_samples: list[float] = []
    min_height = float("inf")
    max_abs_torque = 0.0
    max_abs_integral = 0.0
    next_report = 1.0
    started = time.perf_counter()
    try:
        while viewer is None or viewer.is_running():
            sim_time = step * SIM_DT
            if args.duration is not None and sim_time >= args.duration:
                break
            step_started = time.perf_counter()

            if reset_requested or (
                args.auto_reset_height > 0.0 and float(data.qpos[2]) < args.auto_reset_height
            ):
                reset_robot(mujoco, model, data, indices)
                last_action.fill(0.0)
                target[:] = DEFAULT_JOINT_POS
                integral_effort.fill(0.0)
                reset_requested = False
                resets += 1

            if step % DECIMATION == 0:
                observation = make_observation(
                    data,
                    indices["policy_qpos"],
                    indices["policy_qvel"],
                    indices["pelvis"],
                    indices["gyro_adr"],
                    command,
                    last_action,
                )
                with torch.inference_mode():
                    action_tensor = policy(torch.from_numpy(observation).unsqueeze(0))
                last_action = action_tensor.squeeze(0).numpy().astype(np.float64)
                target = DEFAULT_JOINT_POS + ACTION_SCALE * last_action

            torque, integral_effort = pid_torque(
                target,
                data.qpos[indices["policy_qpos"]],
                data.qvel[indices["policy_qvel"]],
                integral_effort,
                alpha,
            )
            rear_torque = np.clip(
                REAR_KP * (REAR_TARGET - data.qpos[indices["rear_qpos"]])
                - REAR_KD * data.qvel[indices["rear_qvel"]],
                -REAR_EFFORT_LIMIT,
                REAR_EFFORT_LIMIT,
            )
            data.ctrl[indices["policy_actuator"]] = torque
            data.ctrl[indices["free_actuator"]] = 0.0
            data.ctrl[indices["rear_actuator"]] = rear_torque
            mujoco.mj_step(model, data)

            min_height = min(min_height, float(data.qpos[2]))
            max_abs_torque = max(max_abs_torque, float(np.max(np.abs(torque))))
            max_abs_integral = max(max_abs_integral, float(np.max(np.abs(integral_effort))))

            if step % DECIMATION == 0:
                rotation_local_to_world = np.asarray(data.xmat[indices["pelvis"]]).reshape(3, 3)
                world_linear_velocity = np.asarray(data.qvel[:3], dtype=np.float64)
                body_linear_velocity = rotation_local_to_world.T @ world_linear_velocity
                yaw_rate = float(data.sensordata[indices["gyro_adr"] + 2])
                error_xy = float(
                    np.linalg.norm(body_linear_velocity[:2] - np.array((command.vx, command.vy)))
                )
                error_yaw = abs(yaw_rate - command.yaw)
                if resets == 0 or sim_time > 0.0:
                    error_xy_samples.append(error_xy)
                    error_yaw_samples.append(error_yaw)
                csv_writer.writerow(
                    [
                        sim_time, data.qpos[0], data.qpos[1], data.qpos[2],
                        body_linear_velocity[0], body_linear_velocity[1], yaw_rate,
                        error_xy, error_yaw, np.max(np.abs(torque)),
                        np.max(np.abs(integral_effort)), resets,
                    ]
                )
                if viewer is not None:
                    viewer.cam.lookat[:] = data.qpos[:3]
                    viewer.sync()

            if sim_time >= next_report:
                print(
                    f"t={sim_time:5.1f}s height={data.qpos[2]:.3f} "
                    f"xy=({data.qpos[0]:.2f},{data.qpos[1]:.2f}) "
                    f"mean_err_xy={np.mean(error_xy_samples):.3f} "
                    f"max|tau|={max_abs_torque:.1f} max|I|={max_abs_integral:.2f} "
                    f"resets={resets}"
                )
                next_report += 1.0

            if not args.no_realtime:
                remaining = SIM_DT - (time.perf_counter() - step_started)
                if remaining > 0.0:
                    time.sleep(remaining)
            step += 1
    finally:
        csv_file.close()
        if viewer is not None:
            viewer.close()

    wall_time = time.perf_counter() - started
    summary = {
        "checkpoint_iteration": args.checkpoint_iteration,
        "pid_alpha": alpha,
        "simulated_seconds": step * SIM_DT,
        "wall_seconds": wall_time,
        "final_position": [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])],
        "min_height": min_height,
        "resets": resets,
        "mean_error_vel_xy": float(np.mean(error_xy_samples)),
        "mean_error_vel_yaw": float(np.mean(error_yaw_samples)),
        "max_abs_torque": max_abs_torque,
        "max_abs_integral_torque": max_abs_integral,
        "finite_state": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the G1 assist-v1 PID locomotion policy in MuJoCo to MP4."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# This workstation has a display-backed OpenGL context available at DISPLAY=:0.
os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUJOCO_SCRIPTS = PROJECT_ROOT / "scripts/mujoco"
if str(MUJOCO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPTS))

from sim2sim_g1_29dof_assist import (  # noqa: E402
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    EFFORT_LIMIT,
    KD,
    KP,
    NUM_ACTIONS,
    NUM_OBSERVATIONS,
    POLICY_JOINT_NAMES,
    VelocityCommand,
    build_indices,
    make_observation,
    require_file,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/g1_29dof_assist.xml"
)
SIM_DT = 0.001
DECIMATION = 20
INTEGRAL_LIMIT = 10.0
KI = np.array(
    [
        18.0, 18.0, 36.0, 18.0, 18.0, 7.2, 18.0, 18.0, 7.2,
        27.0, 27.0, 5.6, 5.6, 7.2, 7.2, 5.6, 5.6, 7.2, 7.2,
        5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6, 5.6,
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--auto-reset-height", type=float, default=0.35)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    return parser.parse_args()


def start_ffmpeg(output: Path, width: int, height: int, fps: int, crf: int):
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")
    return process


def reset_robot(mujoco_model, data, qpos_ids: np.ndarray) -> None:
    mujoco.mj_resetData(mujoco_model, data)
    data.qpos[qpos_ids] = DEFAULT_JOINT_POS
    data.ctrl[:] = 0.0
    mujoco.mj_forward(mujoco_model, data)


def draw_hud(
    frame: np.ndarray,
    sim_time: float,
    command: VelocityCommand,
    root_velocity_b: np.ndarray,
    height: float,
    resets: int,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (530, 174), (12, 18, 28), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0.0, dst=frame)
    cv2.rectangle(frame, (18, 18), (530, 174), (105, 125, 145), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text: str, y: int, color=(238, 242, 246), scale=0.50) -> None:
        cv2.putText(frame, text, (34, y), font, scale, color, 1, cv2.LINE_AA)

    put("G1 ASSIST-V1 | MODEL 37000 | FULL PID", 46, (250, 250, 250), 0.58)
    put(f"SIM TIME   {sim_time:6.2f} s", 75)
    put(
        f"COMMAND    vx {command.vx:+.2f}  vy {command.vy:+.2f}  yaw {command.yaw:+.2f}",
        103,
        (255, 225, 55),
    )
    put(
        f"ROOT VEL   x {root_velocity_b[0]:+.2f}  y {root_velocity_b[1]:+.2f} m/s",
        131,
        (45, 225, 235),
    )
    put(f"HEIGHT     {height:.3f} m     RESETS {resets}", 159)


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0 or args.fps <= 0:
        raise ValueError("duration and fps must be positive")
    model_path = require_file(args.model, "MuJoCo model")
    policy_path = require_file(args.policy, "TorchScript policy")
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    test_output = policy(torch.zeros((1, NUM_OBSERVATIONS), dtype=torch.float32))
    if tuple(test_output.shape) != (1, NUM_ACTIONS):
        raise ValueError(f"Policy shape mismatch: {tuple(test_output.shape)}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    qpos_ids, qvel_ids, actuator_ids, pelvis_id, gyro_adr = build_indices(mujoco, model)
    model.dof_armature[qvel_ids] = 0.01
    data = mujoco.MjData(model)
    reset_robot(model, data, qpos_ids)

    command = VelocityCommand(args.vx, args.vy, args.yaw)
    command.clamp()
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    target_joint_pos = DEFAULT_JOINT_POS.copy()
    integral_effort = np.zeros(NUM_ACTIONS, dtype=np.float64)
    resets = 0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.lookat[:] = data.qpos[:3]
    ffmpeg = start_ffmpeg(output_path, args.width, args.height, args.fps, args.crf)

    total_steps = int(round(args.duration / SIM_DT))
    next_frame_time = 0.0
    frame_period = 1.0 / args.fps
    first_xy = np.asarray(data.qpos[:2], dtype=np.float64).copy()
    max_abs_integral = 0.0

    try:
        for step in range(total_steps):
            sim_time = step * SIM_DT
            if args.auto_reset_height > 0.0 and float(data.qpos[2]) < args.auto_reset_height:
                reset_robot(model, data, qpos_ids)
                last_action.fill(0.0)
                target_joint_pos[:] = DEFAULT_JOINT_POS
                integral_effort.fill(0.0)
                resets += 1

            if step % DECIMATION == 0:
                observation = make_observation(
                    data, qpos_ids, qvel_ids, pelvis_id, gyro_adr, command, last_action
                )
                with torch.inference_mode():
                    action = policy(torch.from_numpy(observation).unsqueeze(0))
                last_action = action.squeeze(0).cpu().numpy().astype(np.float64)
                target_joint_pos = DEFAULT_JOINT_POS + ACTION_SCALE * last_action

            joint_pos = np.asarray(data.qpos[qpos_ids], dtype=np.float64)
            joint_vel = np.asarray(data.qvel[qvel_ids], dtype=np.float64)
            error_pos = target_joint_pos - joint_pos
            candidate_integral = np.clip(
                integral_effort + KI * error_pos * SIM_DT,
                -INTEGRAL_LIMIT,
                INTEGRAL_LIMIT,
            )
            candidate_effort = KP * error_pos - KD * joint_vel + candidate_integral
            candidate_applied = np.clip(candidate_effort, -EFFORT_LIMIT, EFFORT_LIMIT)
            saturated = candidate_effort != candidate_applied
            pushes_further = error_pos * candidate_effort > 0.0
            accept_integral = ~(saturated & pushes_further)
            integral_effort[accept_integral] = candidate_integral[accept_integral]
            torque = np.clip(
                KP * error_pos - KD * joint_vel + integral_effort,
                -EFFORT_LIMIT,
                EFFORT_LIMIT,
            )
            data.ctrl[actuator_ids] = torque
            max_abs_integral = max(max_abs_integral, float(np.max(np.abs(integral_effort))))
            mujoco.mj_step(model, data)

            if sim_time + 1e-9 >= next_frame_time:
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                root_velocity_b = np.asarray(data.qvel[:3], dtype=np.float64)
                draw_hud(
                    frame,
                    sim_time,
                    command,
                    root_velocity_b,
                    float(data.qpos[2]),
                    resets,
                )
                if ffmpeg.stdin is None:
                    raise RuntimeError("ffmpeg stdin closed unexpectedly")
                ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
                next_frame_time += frame_period
    finally:
        renderer.close()
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

    displacement = np.asarray(data.qpos[:2], dtype=np.float64) - first_xy
    print(
        "VIDEO_SUMMARY "
        f"duration={args.duration:.2f}s frames={round(args.duration * args.fps)} "
        f"position=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) "
        f"displacement=({displacement[0]:.3f},{displacement[1]:.3f}) "
        f"resets={resets} max_abs_integral={max_abs_integral:.3f} "
        f"joints={len(POLICY_JOINT_NAMES)} output={output_path}"
    )


if __name__ == "__main__":
    main()

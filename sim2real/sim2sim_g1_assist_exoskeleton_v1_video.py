#!/usr/bin/env python3
"""Run the G1 v1 assist sim2sim and render the MuJoCo result to MP4.

This is a video-oriented rewrite of
``scripts/mujoco/sim2sim_g1_assist_exoskeleton_v1.py``.  The controller is
unchanged: a 96-observation locomotion policy produces 29 position actions and
a 100-observation assist policy produces two torque actions at 50 Hz.  MuJoCo
physics and the joint PD controller run at 1 kHz.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

# The installed EGL Python binding is incompatible on this workstation, while
# the display-backed GLFW context supports offscreen MuJoCo rendering.
os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    VelocityCommand,
    make_observation,
    name_to_id,
    require_file,
)
from sim2sim_g1_assist_exoskeleton import (  # noqa: E402
    ASSIST_ACTION_SCALE,
    ASSIST_JOINT_NAMES,
    DECIMATION,
    SIM_DT,
    build_indices,
    reset_robot,
)
from sim2sim_g1_assist_exoskeleton_v1 import (  # noqa: E402
    ASSIST_OBSERVATIONS,
    V1AssistHistory,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/"
    "g1_29dof_assist_exoskeleton.xml"
)
DEFAULT_GAIT_POLICY = (
    PROJECT_ROOT / "logs/rsl_rl/g1_assist_amp/2026-08-13_13-45-16/exported/policy.pt"
)
DEFAULT_ASSIST_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_v1_ppo/2026-08-18_13-27-09/"
    "exported/policy.pt"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "video/g1_assist_exoskeleton_v1.mp4"

LEFT_COLOR = (20, 220, 255)
RIGHT_COLOR = (255, 220, 45)
SATURATION_COLOR = (255, 55, 55)
PLOT_WINDOW_SECONDS = 5.0
PLOT_TORQUE_LIMIT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gait-policy", type=Path, default=DEFAULT_GAIT_POLICY)
    parser.add_argument("--assist-policy", type=Path, default=DEFAULT_ASSIST_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality; lower is higher quality.")
    parser.add_argument("--vx", type=float, default=0.7)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--observation-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-reset-height", type=float, default=0.35)
    parser.add_argument("--auto-reset-angle-deg", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--torque-arrow-outward-offset",
        type=float,
        default=0.22,
        help="Move left/right torque arrows outward along their joint axes, in meters.",
    )
    return parser.parse_args()


def start_ffmpeg(output: Path, width: int, height: int, fps: int, crf: int):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")
    return process


def draw_hud(
    frame: np.ndarray,
    sim_time: float,
    command: VelocityCommand,
    root_velocity: np.ndarray,
    resets: int,
) -> None:
    """Draw synchronized command and root-state values."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (475, 154), (10, 17, 27), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, dst=frame)
    cv2.rectangle(frame, (18, 18), (475, 154), (100, 120, 145), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(value, position, scale=0.5, color=(235, 240, 245), thickness=1):
        cv2.putText(frame, value, position, font, scale, color, thickness, cv2.LINE_AA)

    text("G1 + BILATERAL HIP EXOSKELETON", (34, 45), 0.56, (245, 248, 250), 2)
    text(f"SIM TIME   {sim_time:6.2f} s", (34, 73), 0.49)
    text(
        f"COMMAND    vx {command.vx:+.2f}   vy {command.vy:+.2f}   yaw {command.yaw:+.2f}",
        (34, 101),
        0.46,
        (255, 224, 45),
    )
    text(
        f"ROOT VEL   x {root_velocity[0]:+.2f}   y {root_velocity[1]:+.2f} m/s",
        (34, 128),
        0.46,
        (35, 225, 235),
    )
    text(f"RESETS     {resets}", (34, 149), 0.38, (180, 190, 205))



def add_connector(scene, geom_type, width: float, start, end, rgba) -> None:
    """Append a fully initialized connector geom to the current render scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.array((width, width, width), dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        geom_type,
        float(width),
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scene.ngeom += 1


def add_sphere(scene, position, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((radius, radius, radius), dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a right-handed basis spanning the plane normal to a joint axis."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    reference = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    first = np.cross(axis, reference)
    first /= max(float(np.linalg.norm(first)), 1e-9)
    second = np.cross(axis, first)
    return first, second


def add_torque_arc(scene, anchor, axis, torque: float, color) -> None:
    """Draw a 3-D right-hand-rule torque arrow around the true joint axis."""
    magnitude_ratio = min(abs(float(torque)) / ASSIST_ACTION_SCALE, 1.0)
    saturated = abs(float(torque)) >= ASSIST_ACTION_SCALE - 1e-3
    display_color = SATURATION_COLOR if saturated else color
    rgb = tuple(channel / 255.0 for channel in display_color)

    # The marker remains visible at zero torque.
    add_sphere(scene, anchor, 0.017 + 0.006 * magnitude_ratio, (*rgb, 0.95))
    if magnitude_ratio < 0.012:
        return

    first, second = perpendicular_basis(axis)
    direction = 1.0 if torque >= 0.0 else -1.0
    radius = 0.095 + 0.045 * magnitude_ratio
    sweep = 0.40 + 4.0 * magnitude_ratio
    angles = np.linspace(-1.05, -1.05 + direction * sweep, 15)
    points = [
        np.asarray(anchor, dtype=np.float64)
        + radius * (math.cos(angle) * first + math.sin(angle) * second)
        for angle in angles
    ]
    glow_width = 0.011 + 0.022 * magnitude_ratio
    core_width = 0.004 + 0.008 * magnitude_ratio
    for start, end in zip(points[:-1], points[1:]):
        add_connector(
            scene,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            glow_width,
            start,
            end,
            (*rgb, 0.12 + 0.22 * magnitude_ratio),
        )
        add_connector(
            scene,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            core_width,
            start,
            end,
            (*rgb, 0.95),
        )
    tangent_end = points[-1] + (points[-1] - points[-2]) * (0.55 + 0.45 * magnitude_ratio)
    add_connector(
        scene,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.011 + 0.010 * magnitude_ratio,
        points[-2],
        tangent_end,
        (*rgb, 1.0),
    )


def rolling_statistics(samples) -> tuple[float, float, float, float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    left = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    right = np.asarray([sample[2] for sample in samples], dtype=np.float64)
    rms_left = float(np.sqrt(np.mean(left * left)))
    rms_right = float(np.sqrt(np.mean(right * right)))
    peak_left = float(np.max(np.abs(left)))
    peak_right = float(np.max(np.abs(right)))
    mean_amplitude = max(0.5 * (rms_left + rms_right), 1e-9)
    amplitude_gap = 100.0 * abs(rms_left - rms_right) / mean_amplitude
    saturation_count = sum(int(sample[3]) + int(sample[4]) for sample in samples)
    saturation_rate = 100.0 * saturation_count / (2.0 * len(samples))
    return rms_left, rms_right, peak_left, peak_right, amplitude_gap, saturation_rate


def draw_assist_panel(frame: np.ndarray, samples, current_time: float) -> None:
    """Draw a fixed-scale, synchronized five-second assist-torque panel."""
    x0, y0, x1, y1 = 820, 20, 1255, 265
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 17, 27), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (100, 120, 145), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(value, position, scale=0.42, color=(235, 240, 245), thickness=1):
        cv2.putText(frame, value, position, font, scale, color, thickness, cv2.LINE_AA)

    current_left = float(samples[-1][1]) if samples else 0.0
    current_right = float(samples[-1][2]) if samples else 0.0
    current_left_sat = bool(samples[-1][3]) if samples else False
    current_right_sat = bool(samples[-1][4]) if samples else False
    rms_l, rms_r, peak_l, peak_r, gap, saturation_rate = rolling_statistics(samples)
    left_color = SATURATION_COLOR if current_left_sat else LEFT_COLOR
    right_color = SATURATION_COLOR if current_right_sat else RIGHT_COLOR

    text("EXOSKELETON ASSISTANCE - ACTUAL TORQUE", (838, 43), 0.47, (245, 248, 250), 1)
    text(f"L {current_left:+6.2f} N.m  RMS {rms_l:4.2f}  PEAK {peak_l:4.2f}", (838, 66), 0.43, left_color, 1)
    text(f"R {current_right:+6.2f} N.m  RMS {rms_r:4.2f}  PEAK {peak_r:4.2f}", (838, 88), 0.43, right_color, 1)

    plot_x0, plot_y0, plot_x1, plot_y1 = 852, 105, 1238, 224
    cv2.rectangle(frame, (plot_x0, plot_y0), (plot_x1, plot_y1), (20, 27, 38), -1)
    cv2.rectangle(frame, (plot_x0, plot_y0), (plot_x1, plot_y1), (80, 95, 112), 1)
    zero_y = (plot_y0 + plot_y1) // 2
    for dash_x in range(plot_x0, plot_x1, 12):
        cv2.line(frame, (dash_x, zero_y), (min(dash_x + 6, plot_x1), zero_y), (120, 130, 142), 1)
    text("+10", (821, plot_y0 + 5), 0.29, (170, 180, 192), 1)
    text("0", (836, zero_y + 4), 0.29, (170, 180, 192), 1)
    text("-10", (821, plot_y1), 0.29, (170, 180, 192), 1)
    text("N.m", (821, plot_y1 + 14), 0.28, (170, 180, 192), 1)

    start_time = current_time - PLOT_WINDOW_SECONDS

    def plot_point(sample, value_index: int) -> tuple[int, int]:
        x = int(
            plot_x0
            + (sample[0] - start_time) / PLOT_WINDOW_SECONDS * (plot_x1 - plot_x0)
        )
        value = float(np.clip(sample[value_index], -PLOT_TORQUE_LIMIT, PLOT_TORQUE_LIMIT))
        y = int(zero_y - value / PLOT_TORQUE_LIMIT * (plot_y1 - plot_y0) * 0.5)
        return x, y

    sample_list = list(samples)
    for value_index, saturation_index, base_color in (
        (1, 3, LEFT_COLOR),
        (2, 4, RIGHT_COLOR),
    ):
        for previous, current in zip(sample_list[:-1], sample_list[1:]):
            color = (
                SATURATION_COLOR
                if previous[saturation_index] or current[saturation_index]
                else base_color
            )
            cv2.line(
                frame,
                plot_point(previous, value_index),
                plot_point(current, value_index),
                color,
                2,
                cv2.LINE_AA,
            )

    text(f"AMPLITUDE GAP {gap:4.1f}%   SATURATION {saturation_rate:4.1f}%", (838, 247), 0.39, (225, 232, 238), 1)
    text("LAST 5.0 s", (1166, 247), 0.34, (175, 188, 200), 1)


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("--fps, --width and --height must be positive")
    if args.torque_arrow_outward_offset < 0.0:
        raise ValueError("--torque-arrow-outward-offset must be non-negative")

    model_path = require_file(args.model, "MuJoCo model")
    gait_policy_path = require_file(args.gait_policy, "Gait TorchScript policy")
    assist_policy_path = require_file(args.assist_policy, "Assist TorchScript policy")
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gait_policy = torch.jit.load(str(gait_policy_path), map_location="cpu").eval()
    assist_policy = torch.jit.load(str(assist_policy_path), map_location="cpu").eval()
    with torch.inference_mode():
        gait_test = gait_policy(torch.zeros(1, NUM_OBSERVATIONS))
        assist_test = assist_policy(torch.zeros(1, ASSIST_OBSERVATIONS))
    if tuple(gait_test.shape) != (1, NUM_ACTIONS):
        raise ValueError(f"Gait policy shape is {tuple(gait_test.shape)}, expected (1, 29)")
    if tuple(assist_test.shape) != (1, 2):
        raise ValueError(f"Assist policy shape is {tuple(assist_test.shape)}, expected (1, 2)")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    indices = build_indices(mujoco, model)
    assist_joint_ids = np.array(
        [
            name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ASSIST_JOINT_NAMES
        ],
        dtype=int,
    )
    model.dof_armature[indices["policy_qvel"]] = 0.01
    data = mujoco.MjData(model)
    history = V1AssistHistory(args.observation_noise, args.seed)
    reset_robot(mujoco, model, data, indices, history)

    command = VelocityCommand(args.vx, args.vy, args.yaw)
    command.clamp()
    last_gait_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    gait_target = DEFAULT_JOINT_POS.copy()
    assist_command = np.zeros(2, dtype=np.float64)
    control_step = 0
    resets = 0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    ffmpeg = start_ffmpeg(output_path, args.width, args.height, args.fps, args.crf)

    total_frames = int(round(args.duration * args.fps))
    next_frame = 0
    step = 0
    torque_samples = deque()
    started = time.perf_counter()
    try:
        while next_frame < total_frames:
            sim_time = step * SIM_DT

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
                if angle_reset or height_reset:
                    reset_robot(mujoco, model, data, indices, history)
                    last_gait_action.fill(0.0)
                    gait_target[:] = DEFAULT_JOINT_POS
                    assist_command.fill(0.0)
                    control_step = 0
                    resets += 1

                if control_step > 0:
                    history.append(
                        data.qpos[indices["assist_qpos"]],
                        data.qvel[indices["assist_qvel"]],
                        assist_command,
                    )
                gait_observation = make_observation(
                    data,
                    indices["policy_qpos"],
                    indices["policy_qvel"],
                    indices["pelvis"],
                    indices["gyro_adr"],
                    command,
                    last_gait_action,
                )
                assist_observation = history.observation()
                with torch.inference_mode():
                    gait_output = gait_policy(torch.from_numpy(gait_observation).unsqueeze(0))
                    assist_output = assist_policy(
                        torch.from_numpy(assist_observation).unsqueeze(0)
                    )
                last_gait_action = gait_output.squeeze(0).numpy().astype(np.float64)
                assist_action = np.clip(assist_output.squeeze(0).numpy(), -1.0, 1.0)
                gait_target = DEFAULT_JOINT_POS + ACTION_SCALE * last_gait_action
                assist_command = ASSIST_ACTION_SCALE * assist_action.astype(np.float64)
                control_step += 1

            joint_position = data.qpos[indices["policy_qpos"]]
            joint_velocity = data.qvel[indices["policy_qvel"]]
            gait_torque = np.clip(
                KP * (gait_target - joint_position) - KD * joint_velocity,
                -EFFORT_LIMIT,
                EFFORT_LIMIT,
            )
            data.ctrl[indices["policy_actuator"]] = gait_torque
            data.ctrl[indices["assist_actuator"]] = np.clip(
                assist_command, -ASSIST_ACTION_SCALE, ASSIST_ACTION_SCALE
            )
            mujoco.mj_step(model, data)
            step += 1

            post_step_time = step * SIM_DT
            while next_frame < total_frames and post_step_time + 1e-12 >= next_frame / args.fps:
                frame_time = next_frame / args.fps
                actual_assist_torque = np.asarray(
                    data.actuator_force[indices["assist_actuator"]], dtype=np.float64
                ).copy()
                saturated = np.abs(actual_assist_torque) >= ASSIST_ACTION_SCALE - 1e-3
                torque_samples.append(
                    (
                        frame_time,
                        float(actual_assist_torque[0]),
                        float(actual_assist_torque[1]),
                        bool(saturated[0]),
                        bool(saturated[1]),
                    )
                )
                while torque_samples and torque_samples[0][0] < frame_time - PLOT_WINDOW_SECONDS:
                    torque_samples.popleft()

                camera.lookat[:] = data.qpos[:3] + np.array((0.3, 0.0, 0.05))
                renderer.update_scene(data, camera=camera)
                for index, color in enumerate((LEFT_COLOR, RIGHT_COLOR)):
                    joint_id = assist_joint_ids[index]
                    joint_anchor = data.xanchor[joint_id].copy()
                    joint_axis = data.xaxis[joint_id].copy()
                    joint_axis /= max(float(np.linalg.norm(joint_axis)), 1e-9)
                    outward_sign = 1.0 if index == 0 else -1.0
                    display_anchor = (
                        joint_anchor
                        + outward_sign * args.torque_arrow_outward_offset * joint_axis
                    )
                    connector_rgb = tuple(channel / 255.0 for channel in color)
                    add_sphere(renderer.scene, joint_anchor, 0.012, (*connector_rgb, 0.85))
                    add_connector(
                        renderer.scene,
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.0035,
                        joint_anchor,
                        display_anchor,
                        (*connector_rgb, 0.55),
                    )
                    add_torque_arc(
                        renderer.scene,
                        display_anchor,
                        joint_axis,
                        float(actual_assist_torque[index]),
                        color,
                    )
                frame = renderer.render()
                draw_hud(
                    frame,
                    frame_time,
                    command,
                    np.asarray(data.qvel[:3], dtype=np.float64),
                    resets,
                )
                draw_assist_panel(frame, torque_samples, frame_time)
                if ffmpeg.stdin is None:
                    raise RuntimeError("ffmpeg stdin closed unexpectedly")
                ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
                next_frame += 1
                if next_frame % args.fps == 0:
                    print(
                        f"Rendered {next_frame}/{total_frames} frames "
                        f"({next_frame / args.fps:.1f}/{args.duration:.1f} s)"
                    )
    finally:
        renderer.close()
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

    wall_time = time.perf_counter() - started
    print(
        f"Finished: simulated={step * SIM_DT:.2f}s wall={wall_time:.2f}s "
        f"position=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) "
        f"resets={resets}"
    )
    print(f"Video: {output_path}")


if __name__ == "__main__":
    main()

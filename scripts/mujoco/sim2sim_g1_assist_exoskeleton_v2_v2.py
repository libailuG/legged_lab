#!/usr/bin/env python3
"""Run the frozen gait policy and asymmetric v2-v2 assist policy in MuJoCo.

The two additional mechanism joints are held at zero by per-physics-step PD
controllers, independently of the learned gait and hip-assist policies.
Pass ``--disable-assist`` to command zero torque at both hip-assist actuators.
"""

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
    name_to_id,
    require_file,
)
from sim2sim_g1_assist_exoskeleton import (
    SIM_DT,
    SIDES,
    build_indices as build_base_indices,
    create_plot,
    reset_robot as reset_base_robot,
    update_plot,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/"
    "g1_29dof_assist_exoskeleton_2.xml"
)
DEFAULT_GAIT_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_amp/2026-08-13_13-45-16/exported/policy.pt"
)
DEFAULT_ASSIST_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_exoskeleton_v2_v2_ppo/latest/exported/policy.pt"
)

DECIMATION = 10
ASSIST_HISTORY_LENGTH = 25
ASSIST_OBSERVATIONS = 3 * 2 * ASSIST_HISTORY_LENGTH
ASSIST_POSITION_NOISE = np.deg2rad(0.5)
ASSIST_VELOCITY_NOISE = 0.5
ASSIST_TORQUE_NOISE = 0.2
ASSIST_LIFT_TORQUE_RATE_LIMIT = 80.0
ASSIST_PRESS_TORQUE_RATE_LIMIT = 40.0
ASSIST_MOTION_SPEED_DEADZONE = 0.15
ASSIST_MOTION_SPEED_FULL = 0.8
ASSIST_MOTION_FILTER_TIME_CONSTANT = 0.05
ASSIST_LIFT_TORQUE_LIMIT = 10.0
ASSIST_PRESS_TORQUE_LIMIT = 4.0

EXTRA_JOINT_NAMES = (
    "pelvis_rear_upper_box_assist_joint",
    "pelvis_rear_cylinder_assist_joint",
)
EXTRA_TARGET = np.zeros(2, dtype=np.float64)
EXTRA_KP = np.array((10000.0, 4.0), dtype=np.float64)
EXTRA_KD = np.array((20.0, 2.0), dtype=np.float64)
EXTRA_EFFORT_LIMIT = np.array((300.0, 200.0), dtype=np.float64)


def joint_motion_gate(filtered_joint_velocity: np.ndarray) -> np.ndarray:
    """Return the training-matched gate from measured exoskeleton joint speed."""
    phase = np.clip(
        (np.abs(filtered_joint_velocity) - ASSIST_MOTION_SPEED_DEADZONE)
        / (ASSIST_MOTION_SPEED_FULL - ASSIST_MOTION_SPEED_DEADZONE),
        0.0,
        1.0,
    )
    return phase * phase * (3.0 - 2.0 * phase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gait-policy", type=Path, default=DEFAULT_GAIT_POLICY)
    parser.add_argument("--assist-policy", type=Path, default=DEFAULT_ASSIST_POLICY)
    parser.add_argument(
        "--assist-numpy-policy",
        type=Path,
        default=None,
        help="Use exported NumPy assist weights instead of the assist TorchScript policy.",
    )
    parser.add_argument(
        "--disable-assist",
        action="store_true",
        help=(
            "Disable the learned hip-assist policy and command 0 Nm at both "
            "assist actuators. Gait control and rear mechanism PD remain enabled."
        ),
    )
    parser.add_argument(
        "--assist-torque-scale",
        type=float,
        default=1.0,
        help=(
            "Final multiplier applied only when writing the assist actuator command "
            "(default: 1.0). Policy inference, observations, torque history, and the "
            f"-{ASSIST_LIFT_TORQUE_RATE_LIMIT:g}/+{ASSIST_PRESS_TORQUE_RATE_LIMIT:g} "
            "Nm/s directional slew limiters are unchanged; "
            f"final output is -{ASSIST_LIFT_TORQUE_LIMIT:g}/+{ASSIST_PRESS_TORQUE_LIMIT:g} Nm."
        ),
    )
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
            "Apply the v2 training noise to position, velocity, and torque observations. "
            "Disabled by default for deployment-style evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Observation-noise seed.")
    parser.add_argument("--auto-reset-height", type=float, default=0.35)
    parser.add_argument(
        "--auto-reset-angle-deg",
        type=float,
        default=20.0,
        help="Reset above this assist-to-hip angle error; 20 matches the v2 task.",
    )
    return parser.parse_args()


class V2AssistHistory:
    """Oldest-to-newest v2 history: position, velocity, then torque samples."""

    def __init__(self, add_noise: bool, seed: int):
        self.position = deque(maxlen=ASSIST_HISTORY_LENGTH)
        self.velocity = deque(maxlen=ASSIST_HISTORY_LENGTH)
        self.torque = deque(maxlen=ASSIST_HISTORY_LENGTH)
        self.add_noise = add_noise
        self.rng = np.random.default_rng(seed)

    def _position_sample(self, position: np.ndarray) -> np.ndarray:
        sample = np.asarray(position, dtype=np.float32).copy()
        if self.add_noise:
            sample += self.rng.uniform(
                -ASSIST_POSITION_NOISE, ASSIST_POSITION_NOISE, size=2
            ).astype(np.float32)
        return sample

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
        self.position.clear()
        self.velocity.clear()
        self.torque.clear()
        for _ in range(ASSIST_HISTORY_LENGTH):
            self.position.append(self._position_sample(position))
            self.velocity.append(self._velocity_sample(velocity))
            self.torque.append(self._torque_sample(np.zeros(2, dtype=np.float32)))

    def append(self, position: np.ndarray, velocity: np.ndarray, torque: np.ndarray) -> None:
        self.position.append(self._position_sample(position))
        self.velocity.append(self._velocity_sample(velocity))
        self.torque.append(self._torque_sample(torque))

    def observation(self) -> np.ndarray:
        observation = np.concatenate(
            (
                np.asarray(self.position, dtype=np.float32).reshape(-1),
                np.asarray(self.velocity, dtype=np.float32).reshape(-1),
                np.asarray(self.torque, dtype=np.float32).reshape(-1),
            )
        )
        if observation.shape != (ASSIST_OBSERVATIONS,):
            raise RuntimeError(f"Unexpected v2 assist observation shape: {observation.shape}")
        return observation


def build_indices(mujoco, model) -> dict[str, object]:
    """Extend the original exoskeleton indices with the two mechanism joints."""
    indices = build_base_indices(mujoco, model)
    joint_ids = np.array(
        [
            name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in EXTRA_JOINT_NAMES
        ],
        dtype=int,
    )
    actuator_ids = np.array(
        [
            name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in EXTRA_JOINT_NAMES
        ],
        dtype=int,
    )
    indices.update(
        {
            "extra_joint": joint_ids,
            "extra_qpos": model.jnt_qposadr[joint_ids].astype(int),
            "extra_qvel": model.jnt_dofadr[joint_ids].astype(int),
            "extra_actuator": actuator_ids,
        }
    )
    return indices


def configure_extra_joint_limits(model, indices: dict[str, object]) -> None:
    """Make MuJoCo's runtime limits agree with the requested PD limits."""
    joint_ids = indices["extra_joint"]
    actuator_ids = indices["extra_actuator"]
    ranges = np.column_stack((-EXTRA_EFFORT_LIMIT, EXTRA_EFFORT_LIMIT))
    model.jnt_actfrclimited[joint_ids] = 1
    model.jnt_actfrcrange[joint_ids] = ranges
    model.actuator_ctrllimited[actuator_ids] = 1
    model.actuator_ctrlrange[actuator_ids] = ranges


def configure_policy_joint_effort_limits(model, indices: dict[str, object]) -> None:
    """Override URDF-era limits with the Isaac gait actuator effort limits."""
    actuator_ids = np.asarray(indices["policy_actuator"], dtype=int)
    if actuator_ids.size != EFFORT_LIMIT.size:
        raise ValueError(
            f"Expected {EFFORT_LIMIT.size} gait actuators, got {actuator_ids.size}"
        )
    joint_ids = model.actuator_trnid[actuator_ids, 0].astype(int)
    if np.any(joint_ids < 0):
        raise ValueError("Every gait actuator must be attached to a joint")
    ranges = np.column_stack((-EFFORT_LIMIT, EFFORT_LIMIT))
    # MuJoCo applies joint-level actuator-force limits after ctrl clipping.  The
    # XML retains the original URDF limits (for example, 88 N.m at the hip),
    # while the frozen gait policy was trained with the larger Isaac limits.
    model.jnt_actfrclimited[joint_ids] = 1
    model.jnt_actfrcrange[joint_ids] = ranges
    model.actuator_ctrllimited[actuator_ids] = 1
    model.actuator_ctrlrange[actuator_ids] = ranges


def reset_robot(mujoco, model, data, indices, history: V2AssistHistory) -> None:
    """Reset the base controller state and explicitly zero the extra joints."""
    reset_base_robot(mujoco, model, data, indices, history)
    data.qpos[indices["extra_qpos"]] = EXTRA_TARGET
    data.qvel[indices["extra_qvel"]] = 0.0
    data.ctrl[indices["extra_actuator"]] = 0.0
    mujoco.mj_forward(model, data)


def extra_joint_pd_torque(data, indices: dict[str, object]) -> np.ndarray:
    """Return clipped PD efforts for box translation and cylinder rotation."""
    position = data.qpos[indices["extra_qpos"]]
    velocity = data.qvel[indices["extra_qvel"]]
    effort = EXTRA_KP * (EXTRA_TARGET - position) - EXTRA_KD * velocity
    return np.clip(effort, -EXTRA_EFFORT_LIMIT, EXTRA_EFFORT_LIMIT)


def main() -> None:
    args = parse_args()
    model_path = require_file(args.model, "MuJoCo model")
    gait_policy_path = require_file(args.gait_policy, "Gait TorchScript policy")
    assist_policy_path = None
    assist_numpy_policy_path = None
    if not args.disable_assist:
        if args.assist_numpy_policy is None:
            assist_policy_path = require_file(
                args.assist_policy, "V2 assist TorchScript policy"
            )
        else:
            assist_numpy_policy_path = require_file(
                args.assist_numpy_policy, "V2 assist NumPy policy"
            )
    if args.headless and args.duration is None:
        args.duration = 10.0
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if not math.isfinite(args.assist_torque_scale) or args.assist_torque_scale < 0.0:
        raise ValueError("--assist-torque-scale must be a finite non-negative number")

    import mujoco
    import torch

    import matplotlib

    if args.headless or args.no_live_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gait_policy = torch.jit.load(str(gait_policy_path), map_location="cpu")
    gait_policy.eval()
    with torch.inference_mode():
        gait_test_output = gait_policy(torch.zeros(1, NUM_OBSERVATIONS))
    if tuple(gait_test_output.shape) != (1, NUM_ACTIONS):
        raise ValueError("Frozen gait policy must map 96 observations to 29 actions")

    assist_policy = None
    assist_policy_is_numpy = False
    if assist_policy_path is not None:
        assist_policy = torch.jit.load(str(assist_policy_path), map_location="cpu")
        assist_policy.eval()
        with torch.inference_mode():
            assist_test_output = assist_policy(torch.zeros(1, ASSIST_OBSERVATIONS))
        if tuple(assist_test_output.shape) != (1, 2):
            raise ValueError(
                f"V2 assist policy must map {ASSIST_OBSERVATIONS} observations to 2 actions"
            )
    elif assist_numpy_policy_path is not None:
        import sys

        numpy_policy_dir = PROJECT_ROOT / "sim2real/g1_assist_exoskeleton_v2"
        if str(numpy_policy_dir) not in sys.path:
            sys.path.insert(0, str(numpy_policy_dir))
        from numpy_assist_policy import NumpyAssistPolicy

        assist_policy = NumpyAssistPolicy(assist_numpy_policy_path)
        assist_test_output = assist_policy(np.zeros(ASSIST_OBSERVATIONS, dtype=np.float32))
        if assist_test_output.shape != (2,):
            raise ValueError(
                f"V2 NumPy assist policy must map {ASSIST_OBSERVATIONS} observations to 2 actions"
            )
        assist_policy_is_numpy = True

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    indices = build_indices(mujoco, model)
    configure_policy_joint_effort_limits(model, indices)
    configure_extra_joint_limits(model, indices)
    model.dof_armature[indices["policy_qvel"]] = 0.01
    data = mujoco.MjData(model)
    history = V2AssistHistory(args.observation_noise, args.seed)
    reset_robot(mujoco, model, data, indices, history)

    command = VelocityCommand(args.vx, args.vy, args.yaw)
    command.clamp()
    last_gait_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    gait_target = DEFAULT_JOINT_POS.copy()
    assist_torque = np.zeros(2, dtype=np.float64)
    filtered_assist_joint_velocity = np.zeros(2, dtype=np.float64)
    reset_requested = False

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            PROJECT_ROOT / "logs/sim2sim_analysis_exoskeleton_2_no_assist"
            if assist_policy is None
            else (
                assist_numpy_policy_path.parent.parent / "output"
                if assist_numpy_policy_path is not None
                else assist_policy_path.parent.parent / "sim2sim_analysis_exoskeleton_2"
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "g1_assist_exoskeleton_2_sim2sim.csv"
    png_path = output_dir / "g1_assist_exoskeleton_2_sim2sim.png"
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
    fig, axes, lines = create_plot(args.auto_reset_angle_deg)
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
    assist_mode = "disabled (0 Nm)" if args.disable_assist else (
        "NumPy policy" if assist_policy_is_numpy else "TorchScript policy"
    )
    loaded_policy_path = assist_numpy_policy_path or assist_policy_path
    print(f"Assist mode:   {assist_mode}")
    print(f"V2 policy:     {loaded_policy_path if loaded_policy_path else 'not loaded'}")
    print(f"Assist scale:  {args.assist_torque_scale:g}")
    print(
        f"Assist obs:    {ASSIST_OBSERVATIONS} "
        "(25 position + 25 velocity + 25 torque samples)"
    )
    print(f"Obs noise:     {'enabled' if args.observation_noise else 'disabled'}")
    print(
        f"Control:       dt={SIM_DT}, decimation={DECIMATION}, "
        f"policy rate={1.0 / (SIM_DT * DECIMATION):g} Hz"
    )
    print(
        f"Torque slew:   -direction {ASSIST_LIFT_TORQUE_RATE_LIMIT:g} Nm/s / "
        f"+direction {ASSIST_PRESS_TORQUE_RATE_LIMIT:g} Nm/s"
    )
    print(
        f"Torque range:  -{ASSIST_LIFT_TORQUE_LIMIT:g} Nm lift / "
        f"+{ASSIST_PRESS_TORQUE_LIMIT:g} Nm press"
    )
    print(
        "Motion gate:   "
        f"0 below |joint vel|={ASSIST_MOTION_SPEED_DEADZONE:g} rad/s, "
        f"1 at {ASSIST_MOTION_SPEED_FULL:g} rad/s, "
        f"filter tau={ASSIST_MOTION_FILTER_TIME_CONSTANT:g}s"
    )
    print(f"Gait effort:   Isaac limits applied to {EFFORT_LIMIT.size} joints")
    print(
        "Extra PD:      "
        f"box target=0 kp={EXTRA_KP[0]:g} kd={EXTRA_KD[0]:g} limit={EXTRA_EFFORT_LIMIT[0]:g}; "
        f"cylinder target=0 kp={EXTRA_KP[1]:g} kd={EXTRA_KD[1]:g} "
        f"limit={EXTRA_EFFORT_LIMIT[1]:g}"
    )
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
                    filtered_assist_joint_velocity.fill(0.0)
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
                    gait_action_tensor = gait_policy(torch.from_numpy(gait_obs).unsqueeze(0))
                last_gait_action = gait_action_tensor.squeeze(0).numpy().astype(np.float64)
                gait_target = DEFAULT_JOINT_POS + ACTION_SCALE * last_gait_action
                if assist_policy is None:
                    assist_torque.fill(0.0)
                else:
                    assist_obs = history.observation()
                    if assist_policy_is_numpy:
                        assist_action = np.clip(assist_policy(assist_obs), -1.0, 1.0)
                    else:
                        with torch.inference_mode():
                            assist_action_tensor = assist_policy(
                                torch.from_numpy(assist_obs).unsqueeze(0)
                            )
                        assist_action = np.clip(
                            assist_action_tensor.squeeze(0).numpy(), -1.0, 1.0
                        )
                    directional_limit = np.where(
                        assist_action < 0.0,
                        ASSIST_LIFT_TORQUE_LIMIT,
                        ASSIST_PRESS_TORQUE_LIMIT,
                    )
                    motion_filter_alpha = 1.0 - math.exp(
                        -(SIM_DT * DECIMATION) / ASSIST_MOTION_FILTER_TIME_CONSTANT
                    )
                    filtered_assist_joint_velocity += motion_filter_alpha * (
                        data.qvel[indices["assist_qvel"]]
                        - filtered_assist_joint_velocity
                    )
                    assist_torque_target = (
                        directional_limit
                        * joint_motion_gate(filtered_assist_joint_velocity)
                        * assist_action.astype(np.float64)
                    )
                    requested_delta = assist_torque_target - assist_torque
                    max_torque_delta = np.where(
                        requested_delta < 0.0,
                        ASSIST_LIFT_TORQUE_RATE_LIMIT,
                        ASSIST_PRESS_TORQUE_RATE_LIMIT,
                    ) * SIM_DT * DECIMATION
                    assist_torque += np.clip(
                        requested_delta,
                        -max_torque_delta,
                        max_torque_delta,
                    )
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
                args.assist_torque_scale * assist_torque,
                -ASSIST_LIFT_TORQUE_LIMIT,
                ASSIST_PRESS_TORQUE_LIMIT,
            )
            data.ctrl[indices["extra_actuator"]] = extra_joint_pd_torque(data, indices)
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
            f"G1 Assist Exoskeleton 2 MuJoCo Sim2Sim ({simulated_time:.2f} s)",
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

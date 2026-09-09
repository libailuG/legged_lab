#!/usr/bin/env python3
"""Run LeggedLab-Isaac-AMP-G1-assist-v1 in MuJoCo with 1 kHz PID.

Use --policy for an exported TorchScript actor (including its observation
normalizer), not a raw model_N.pt training checkpoint. The default export
belongs to the locally downloaded model_50400 run.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/g1_29dof_assist.xml"
)
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "logs/rsl_rl/g1_assist_v1_amp/"
    "2026-09-04_20-03-58_rtx5090_bounded_noise_entropy0_remaining_29600/exported/policy.pt"
)

POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_JOINT_POS = np.array(
    [
        -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.3, 0.3, 0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0.0, 0.0,
        0.0, 0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# Values mirror g1_assist_v1/robot_cfg.py; kept local for independent editing.
KP = np.array(
    [
        180.0, 180.0, 360.0, 180.0, 180.0, 72.0, 180.0, 180.0, 72.0,
        270.0, 270.0, 56.0, 56.0, 72.0, 72.0, 56.0, 56.0, 72.0, 72.0,
        56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
    ],
    dtype=np.float64,
)
KD = np.array(
    [
        3.0, 3.0, 7.5, 3.0, 3.0, 7.5, 3.0, 3.0, 7.5,
        6.0, 6.0, 1.3, 1.3, 3.0, 3.0, 1.3, 1.3, 3.0, 3.0,
        1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
    ],
    dtype=np.float64,
)
EFFORT_LIMIT = np.array(
    [
        200.0, 200.0, 200.0, 316.0, 316.0, 57.0, 200.0, 200.0, 57.0,
        316.0, 316.0, 38.0, 38.0, 57.0, 57.0, 38.0, 38.0, 57.0, 57.0,
        38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 8.0, 8.0, 8.0, 8.0,
    ],
    dtype=np.float64,
)

NUM_ACTIONS = len(POLICY_JOINT_NAMES)
NUM_OBSERVATIONS = 3 + 3 + 3 + NUM_ACTIONS * 3
ACTION_SCALE = 0.25
# Match v1: 1 kHz physics/PID and 50 Hz policy.
SIM_DT = 0.001
DECIMATION = 20
KI = KP * 0.1
INTEGRAL_EFFORT_LIMIT = 10.0


@dataclass
class VelocityCommand:
    vx: float
    vy: float
    yaw: float

    def clamp(self) -> None:
        self.vx = float(np.clip(self.vx, -0.5, 3.0))
        self.vy = float(np.clip(self.vy, -0.5, 0.5))
        self.yaw = float(np.clip(self.yaw, -1.0, 1.0))

    def zero(self) -> None:
        self.vx = self.vy = self.yaw = 0.0

    def as_array(self) -> np.ndarray:
        return np.array((self.vx, self.vy, self.yaw), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="Exported TorchScript policy path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML model path.")
    parser.add_argument("--vx", type=float, default=0.7, help="Initial forward command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial lateral command in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Initial yaw-rate command in rad/s.")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds; unlimited with viewer.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer (defaults to 10 seconds).")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace simulation to wall-clock time.")
    parser.add_argument(
        "--auto-reset-height",
        type=float,
        default=0.35,
        help="Reset when pelvis height falls below this value; set <=0 to disable.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def name_to_id(mujoco, model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Required MuJoCo object is missing: {name}")
    return object_id


def build_indices(mujoco, model):
    joint_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in POLICY_JOINT_NAMES],
        dtype=int,
    )
    actuator_ids = np.array(
        [name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in POLICY_JOINT_NAMES],
        dtype=int,
    )
    qpos_ids = model.jnt_qposadr[joint_ids].astype(int)
    qvel_ids = model.jnt_dofadr[joint_ids].astype(int)
    pelvis_id = name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    gyro_id = name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")
    gyro_adr = int(model.sensor_adr[gyro_id])
    if int(model.sensor_dim[gyro_id]) != 3:
        raise ValueError("Pelvis gyro must have dimension 3")
    return qpos_ids, qvel_ids, actuator_ids, pelvis_id, gyro_adr


def reset_robot(mujoco, model, data, qpos_ids: np.ndarray) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_ids] = DEFAULT_JOINT_POS
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def make_observation(
    data,
    qpos_ids: np.ndarray,
    qvel_ids: np.ndarray,
    pelvis_id: int,
    gyro_adr: int,
    command: VelocityCommand,
    last_action: np.ndarray,
) -> np.ndarray:
    base_ang_vel = np.asarray(data.sensordata[gyro_adr : gyro_adr + 3], dtype=np.float32)
    rotation_local_to_world = np.asarray(data.xmat[pelvis_id], dtype=np.float64).reshape(3, 3)
    projected_gravity = rotation_local_to_world.T @ np.array((0.0, 0.0, -1.0), dtype=np.float64)
    joint_pos_rel = np.asarray(data.qpos[qpos_ids] - DEFAULT_JOINT_POS, dtype=np.float32)
    joint_vel = np.asarray(data.qvel[qvel_ids], dtype=np.float32)

    observation = np.concatenate(
        (
            base_ang_vel,
            projected_gravity.astype(np.float32),
            command.as_array(),
            joint_pos_rel,
            joint_vel,
            last_action.astype(np.float32),
        )
    )
    if observation.shape != (NUM_OBSERVATIONS,):
        raise RuntimeError(f"Unexpected observation shape: {observation.shape}")
    return observation


def pid_torque(target, position, velocity, integral_effort):
    """Update integral torque in place, matching the Isaac v1 actuator."""
    error = target - position
    candidate_integral = np.clip(
        integral_effort + KI * error * SIM_DT,
        -INTEGRAL_EFFORT_LIMIT, INTEGRAL_EFFORT_LIMIT,
    )
    candidate = KP * error - KD * velocity + candidate_integral
    clipped = np.clip(candidate, -EFFORT_LIMIT, EFFORT_LIMIT)
    accept = ~((candidate != clipped) & (error * candidate > 0.0))
    integral_effort[accept] = candidate_integral[accept]
    return np.clip(KP * error - KD * velocity + integral_effort, -EFFORT_LIMIT, EFFORT_LIMIT)


def configure_actuators(model, qvel_ids, actuator_ids):
    """Apply v1 effort bounds at both motor and joint levels."""
    if not np.allclose(model.actuator_gear[actuator_ids, 0], 1.0):
        raise ValueError("Expected unit-gear torque motors")
    joint_ids = model.actuator_trnid[actuator_ids, 0].astype(int)
    limits = np.column_stack((-EFFORT_LIMIT, EFFORT_LIMIT))
    model.jnt_actfrclimited[joint_ids] = 1
    model.jnt_actfrcrange[joint_ids] = limits
    model.actuator_ctrllimited[actuator_ids] = 1
    model.actuator_ctrlrange[actuator_ids] = limits
    model.actuator_forcelimited[actuator_ids] = 1
    model.actuator_forcerange[actuator_ids] = limits
    model.dof_armature[qvel_ids] = 0.01


class CommandPanel:
    """Non-blocking slider window, serviced on the simulation's main thread."""

    def __init__(self, command: VelocityCommand, reset_callback):
        import tkinter as tk
        from tkinter import ttk

        self.command = command
        self.closed = False
        self.root = tk.Tk()
        self.root.title("G1 assist-v1 | Velocity control")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Velocity commands", font=("sans-serif", 14, "bold")).pack(anchor="w")
        self.sliders = {}
        for name, label, lower, upper in (
            ("vx", "Forward velocity (m/s)", -0.5, 3.0),
            ("vy", "Lateral velocity (m/s)", -0.5, 0.5),
            ("yaw", "Yaw rate (rad/s)", -1.0, 1.0),
        ):
            slider = tk.Scale(
                frame, label=label, from_=lower, to=upper, resolution=0.01,
                orient="horizontal", length=360, showvalue=True,
                command=lambda value, field=name: setattr(self.command, field, float(value)),
            )
            slider.set(getattr(command, name))
            slider.pack(fill="x", pady=6)
            self.sliders[name] = slider
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons, text="Zero commands", command=self.zero).pack(side="left")
        ttk.Button(buttons, text="Reset robot", command=reset_callback).pack(side="right")

    def zero(self):
        self.command.zero()
        for slider in self.sliders.values():
            slider.set(0.0)

    def update(self):
        if not self.closed:
            self.root.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()


def main() -> None:
    args = parse_args()
    model_path = require_file(args.model, "MuJoCo model")
    policy_path = require_file(args.policy, "TorchScript policy")
    if args.duration is not None and args.duration <= 0.0:
        raise ValueError("--duration must be greater than zero")
    if args.headless and args.duration is None:
        args.duration = 10.0

    import mujoco
    import torch

    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    test_output = policy(torch.zeros((1, NUM_OBSERVATIONS), dtype=torch.float32))
    if tuple(test_output.shape) != (1, NUM_ACTIONS):
        raise ValueError(
            f"Policy shape mismatch: expected (1, {NUM_ACTIONS}), got {tuple(test_output.shape)}"
        )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = SIM_DT
    qpos_ids, qvel_ids, actuator_ids, pelvis_id, gyro_adr = build_indices(mujoco, model)
    configure_actuators(model, qvel_ids, actuator_ids)
    data = mujoco.MjData(model)
    reset_robot(mujoco, model, data, qpos_ids)

    command = VelocityCommand(args.vx, args.vy, args.yaw)
    command.clamp()
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    target_joint_pos = DEFAULT_JOINT_POS.copy()
    integral_effort = np.zeros(NUM_ACTIONS, dtype=np.float64)
    reset_requested = False

    def request_reset() -> None:
        nonlocal reset_requested
        reset_requested = True

    print(f"Model:  {model_path}")
    print(f"Policy: {policy_path}")
    print(f"Control: dt={SIM_DT}, decimation={DECIMATION}, policy_rate={1.0 / (SIM_DT * DECIMATION):.1f} Hz")
    print(f"Initial command: vx={command.vx:.2f}, vy={command.vy:.2f}, yaw={command.yaw:.2f}")
    viewer = None
    panel = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=False, show_right_ui=False,
        )
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        try:
            panel = CommandPanel(command, request_reset)
        except Exception:
            viewer.close()
            raise

    started = time.perf_counter()
    step = 0
    policy_step = 0
    next_report = 1.0
    resets = 0
    try:
        while viewer is None or viewer.is_running():
            elapsed = step * SIM_DT
            if args.duration is not None and elapsed >= args.duration:
                break
            step_started = time.perf_counter()

            if panel is not None and step % DECIMATION == 0:
                panel.update()
                if panel.closed:
                    break

            if reset_requested or (
                args.auto_reset_height > 0.0 and float(data.qpos[2]) < args.auto_reset_height
            ):
                reset_robot(mujoco, model, data, qpos_ids)
                last_action.fill(0.0)
                target_joint_pos[:] = DEFAULT_JOINT_POS
                integral_effort.fill(0.0)
                reset_requested = False
                resets += 1
                policy_step = 0

            if policy_step % DECIMATION == 0:
                observation = make_observation(
                    data, qpos_ids, qvel_ids, pelvis_id, gyro_adr, command, last_action
                )
                with torch.inference_mode():
                    action_tensor = policy(torch.from_numpy(observation).unsqueeze(0))
                last_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float64)
                if not np.all(np.isfinite(last_action)):
                    raise RuntimeError("Policy produced non-finite actions")
                target_joint_pos = DEFAULT_JOINT_POS + ACTION_SCALE * last_action

            joint_pos = data.qpos[qpos_ids]
            joint_vel = data.qvel[qvel_ids]
            torque = pid_torque(target_joint_pos, joint_pos, joint_vel, integral_effort)
            data.ctrl[actuator_ids] = torque
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                raise RuntimeError("MuJoCo produced a non-finite state")

            if viewer is not None and step % DECIMATION == 0:
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()

            if elapsed >= next_report:
                print(
                    f"t={elapsed:6.1f}s height={data.qpos[2]:.3f} "
                    f"xy=({data.qpos[0]:.2f},{data.qpos[1]:.2f}) "
                    f"cmd=({command.vx:.2f},{command.vy:.2f},{command.yaw:.2f}) "
                    f"max|tau|={np.max(np.abs(np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT))):.1f} "
                    f"resets={resets}"
                )
                next_report += 1.0

            if not args.no_realtime:
                remaining = SIM_DT - (time.perf_counter() - step_started)
                if remaining > 0.0:
                    time.sleep(remaining)
            step += 1
            policy_step += 1
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        if viewer is not None:
            viewer.close()

    wall_time = time.perf_counter() - started
    print(
        f"Finished: simulated={step * SIM_DT:.2f}s wall={wall_time:.2f}s "
        f"position=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) resets={resets}"
    )


if __name__ == "__main__":
    main()

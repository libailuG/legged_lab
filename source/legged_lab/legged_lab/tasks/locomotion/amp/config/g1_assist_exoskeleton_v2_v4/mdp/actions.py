"""Frozen gait plus a deployable single-pulse paired assist controller."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .pulse import SinglePulse
from .paired_assist import paired_torques, shared_motion_gate, slew_paired_scalar


POLICY_JOINT_NAMES = [
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
]

ASSIST_JOINT_NAMES = ["left_hip_pitch_assist_joint", "right_hip_pitch_assist_joint"]
EXTRA_POSITION_JOINT_NAMES = [
    "pelvis_rear_upper_box_assist_joint",
    "pelvis_rear_cylinder_assist_joint",
]


class FrozenGaitAssistTorqueAction(ActionTerm):
    """Run gait/assist policies while holding the two mechanism joints at zero."""

    cfg: FrozenGaitAssistTorqueActionCfg
    _asset: Articulation

    def __init__(self, cfg: FrozenGaitAssistTorqueActionCfg, env):
        super().__init__(cfg, env)

        if not os.path.isfile(cfg.frozen_policy_path):
            raise FileNotFoundError(f"Frozen gait policy was not found: {cfg.frozen_policy_path}")
        if cfg.torque_rate_limit <= 0.0 or cfg.torque_limit <= 0.0:
            raise ValueError("torque_rate_limit and torque_limit must be greater than zero")
        if cfg.motion_speed_deadzone < 0.0 or cfg.motion_speed_full <= cfg.motion_speed_deadzone:
            raise ValueError("Motion gate requires 0 <= deadzone < full_motion_speed")
        if cfg.motion_filter_time_constant <= 0.0:
            raise ValueError("motion_filter_time_constant must be greater than zero")

        ratio = cfg.gait_policy_period / env.step_dt
        self._gait_interval = round(ratio)
        if self._gait_interval < 1 or not math.isclose(ratio, self._gait_interval):
            raise ValueError("gait_policy_period must be an integer multiple of assist step_dt")
        self._gait_tick = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._policy_joint_ids, resolved_policy_names = self._asset.find_joints(
            cfg.policy_joint_names, preserve_order=True
        )
        self._assist_joint_ids, resolved_assist_names = self._asset.find_joints(
            cfg.assist_joint_names, preserve_order=True
        )
        self._extra_position_joint_ids, resolved_extra_names = self._asset.find_joints(
            cfg.extra_position_joint_names, preserve_order=True
        )
        if resolved_policy_names != cfg.policy_joint_names:
            raise RuntimeError(
                f"Frozen gait joint order mismatch: expected {cfg.policy_joint_names}, "
                f"got {resolved_policy_names}"
            )
        if resolved_assist_names != cfg.assist_joint_names:
            raise RuntimeError(
                f"Assist joint order mismatch: expected {cfg.assist_joint_names}, "
                f"got {resolved_assist_names}"
            )
        required_effort = cfg.torque_limit
        physical_limits = self._asset.data.joint_effort_limits[:, self._assist_joint_ids]
        if torch.any(physical_limits < required_effort).item():
            raise ValueError(
                f"Assist actuator physical limit must be at least {required_effort} Nm; "
                "increase exoskeleton_torque.effort_limit_sim to match the action limits"
            )
        if resolved_extra_names != cfg.extra_position_joint_names:
            raise RuntimeError(
                f"Extra position joint order mismatch: expected "
                f"{cfg.extra_position_joint_names}, got {resolved_extra_names}"
            )

        self._frozen_policy = torch.jit.load(cfg.frozen_policy_path, map_location=self.device)
        self._frozen_policy.eval()

        self._pulse = SinglePulse(self.num_envs, self.device, env.step_dt,
                                  cfg.pulse_duration, cfg.pulse_release_duration,
                                  cfg.torque_limit, cfg.torque_rate_limit, cfg.pulse_confirm_time)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._previous_processed_actions = torch.zeros_like(self._processed_actions)
        self._requested_torque_delta = torch.zeros_like(self._processed_actions)
        self._filtered_assist_joint_velocity = torch.zeros_like(self._processed_actions)
        self._motion_gate = torch.zeros_like(self._raw_actions)
        self._gait_actions = torch.zeros(
            self.num_envs, len(POLICY_JOINT_NAMES), device=self.device
        )
        self._previous_gait_actions = torch.zeros_like(self._gait_actions)
        self._gait_position_targets = self._asset.data.default_joint_pos[
            :, self._policy_joint_ids
        ].clone()
        self._extra_position_targets = torch.zeros(
            self.num_envs,
            len(self._extra_position_joint_ids),
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Commanded assist torque in N.m."""
        return self._processed_actions

    @property
    def assist_torque_rate(self) -> torch.Tensor:
        """Actual commanded assist-torque rate in N.m/s after rate limiting."""
        return (
            self._processed_actions - self._previous_processed_actions
        ) / self._env.step_dt

    @property
    def requested_torque_delta(self) -> torch.Tensor:
        """Requested torque change from the previously applied command in N.m."""
        return self._requested_torque_delta

    @property
    def motion_gate(self) -> torch.Tensor:
        """Shared bilateral gate derived only from measured exoskeleton motion, shape (N, 1)."""
        return self._motion_gate

    @property
    def gait_actions(self) -> torch.Tensor:
        """Frozen gait policy's latest 29 normalized position actions."""
        return self._gait_actions

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Paired assist expects action shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}")
        self._raw_actions[:] = actions
        filter_alpha = 1.0 - math.exp(
            -self._env.step_dt / self.cfg.motion_filter_time_constant
        )
        self._filtered_assist_joint_velocity.lerp_(
            self._asset.data.joint_vel[:, self._assist_joint_ids], filter_alpha
        )
        self._motion_gate[:] = shared_motion_gate(
            self._filtered_assist_joint_velocity,
            self.cfg.motion_speed_deadzone,
            self.cfg.motion_speed_full,
        )
        # No contact, base state, hip state or command information enters this
        # assist-control path. The frozen gait policy below is simulation-only.
        self._previous_processed_actions[:] = self._processed_actions
        scalar = self._pulse.step(actions, self._filtered_assist_joint_velocity, self._motion_gate)
        self._processed_actions[:] = paired_torques(scalar)
        self._requested_torque_delta[:] = self._processed_actions - self._previous_processed_actions

        # Independent per-environment phase: reset environments infer immediately.
        due = self._gait_tick == 0
        self._gait_tick[:] = (self._gait_tick + 1) % self._gait_interval
        if not torch.any(due):
            return
        gait_obs = torch.cat(
            (
                self._asset.data.root_ang_vel_b,
                self._asset.data.projected_gravity_b,
                self._env.command_manager.get_command(self.cfg.command_name),
                self._asset.data.joint_pos[:, self._policy_joint_ids]
                - self._asset.data.default_joint_pos[:, self._policy_joint_ids],
                self._asset.data.joint_vel[:, self._policy_joint_ids]
                - self._asset.data.default_joint_vel[:, self._policy_joint_ids],
                self._previous_gait_actions,
            ),
            dim=-1,
        )
        gait_obs = gait_obs[due]
        if self.cfg.gait_observation_noise:
            bounds = gait_obs.new_tensor([.35]*3 + [.05]*3 + [0.]*3 + [.03]*29 + [1.75]*29 + [0.]*29)
            gait_obs = gait_obs + (2 * torch.rand_like(gait_obs) - 1) * bounds
        if gait_obs.shape[-1] != 96:
            raise RuntimeError(
                f"Frozen gait policy expects 96 observations, got {gait_obs.shape[-1]}"
            )

        with torch.inference_mode():
            gait_actions = self._frozen_policy(gait_obs)
        if gait_actions.shape[-1] != len(self._policy_joint_ids):
            raise RuntimeError(
                f"Frozen gait policy expects 29 outputs, got action shape "
                f"{tuple(gait_actions.shape)}"
            )
        self._gait_actions[due] = gait_actions
        self._previous_gait_actions[due] = gait_actions
        self._gait_position_targets[due] = (
            self._asset.data.default_joint_pos[due][:, self._policy_joint_ids]
            + self.cfg.gait_action_scale * gait_actions
        )

    def apply_actions(self):
        # Reconstruct the pair at the actuator boundary: limiting must never
        # introduce unequal torques, even during release or reversal.
        scalar = self._processed_actions[:, :1].clamp(-self.cfg.torque_limit, self.cfg.torque_limit)
        self._processed_actions[:] = paired_torques(scalar)
        self._asset.set_joint_position_target(
            self._gait_position_targets, joint_ids=self._policy_joint_ids
        )
        self._asset.set_joint_position_target(
            self._extra_position_targets, joint_ids=self._extra_position_joint_ids
        )
        self._asset.set_joint_effort_target(
            self._processed_actions, joint_ids=self._assist_joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._pulse.reset(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._previous_processed_actions[env_ids] = 0.0
        self._requested_torque_delta[env_ids] = 0.0
        self._filtered_assist_joint_velocity[env_ids] = 0.0
        self._motion_gate[env_ids] = 0.0
        self._gait_tick[env_ids] = 0
        self._gait_actions[env_ids] = 0.0
        self._previous_gait_actions[env_ids] = 0.0
        self._gait_position_targets[env_ids] = self._asset.data.default_joint_pos[env_ids][
            :, self._policy_joint_ids
        ]
        self._extra_position_targets[env_ids] = 0.0


@configclass
class FrozenGaitAssistTorqueActionCfg(ActionTermCfg):
    """One normalized amplitude action -> a latched single-lobe [T,-T] trajectory."""

    class_type: type[ActionTerm] = FrozenGaitAssistTorqueAction
    asset_name: str = "robot"
    frozen_policy_path: str = ""
    policy_joint_names: list[str] = POLICY_JOINT_NAMES
    assist_joint_names: list[str] = ASSIST_JOINT_NAMES
    extra_position_joint_names: list[str] = EXTRA_POSITION_JOINT_NAMES
    command_name: str = "base_velocity"
    pulse_duration: float = 0.4
    pulse_release_duration: float = 0.2
    pulse_confirm_time: float = 0.03
    gait_action_scale: float = 0.25
    gait_policy_period: float = 0.02
    gait_observation_noise: bool = False
    torque_limit: float = 10.0
    torque_rate_limit: float = 80.0
    motion_speed_deadzone: float = 0.15
    motion_speed_full: float = 0.8
    motion_filter_time_constant: float = 0.05


def pulse_controller_state(env, action_name="assist_torque"):
    term = env.action_manager.get_term(action_name)
    return torch.cat((term._pulse.observation(), term._filtered_assist_joint_velocity), dim=-1)

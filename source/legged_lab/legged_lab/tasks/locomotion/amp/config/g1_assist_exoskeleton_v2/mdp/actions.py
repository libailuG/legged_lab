"""Hierarchical action term for the G1 assist-exoskeleton v2 task."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


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
        if cfg.assist_torque_rate_limit <= 0.0:
            raise ValueError("assist_torque_rate_limit must be greater than zero")

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
        if resolved_extra_names != cfg.extra_position_joint_names:
            raise RuntimeError(
                f"Extra position joint order mismatch: expected "
                f"{cfg.extra_position_joint_names}, got {resolved_extra_names}"
            )

        self._frozen_policy = torch.jit.load(cfg.frozen_policy_path, map_location=self.device)
        self._frozen_policy.eval()

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._previous_processed_actions = torch.zeros_like(self._raw_actions)
        self._requested_torque_delta = torch.zeros_like(self._raw_actions)
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
        return 2

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
    def gait_actions(self) -> torch.Tensor:
        """Frozen gait policy's latest 29 normalized position actions."""
        return self._gait_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target_torque = torch.clamp(actions, -1.0, 1.0) * self.cfg.assist_torque_limit
        self._previous_processed_actions[:] = self._processed_actions
        self._requested_torque_delta[:] = target_torque - self._processed_actions
        maximum_torque_delta = self.cfg.assist_torque_rate_limit * self._env.step_dt
        self._processed_actions.add_(
            torch.clamp(
                target_torque - self._processed_actions,
                min=-maximum_torque_delta,
                max=maximum_torque_delta,
            )
        )

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
        self._gait_actions[:] = gait_actions
        self._previous_gait_actions[:] = gait_actions
        self._gait_position_targets[:] = (
            self._asset.data.default_joint_pos[:, self._policy_joint_ids]
            + self.cfg.gait_action_scale * gait_actions
        )

    def apply_actions(self):
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
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._previous_processed_actions[env_ids] = 0.0
        self._requested_torque_delta[env_ids] = 0.0
        self._gait_actions[env_ids] = 0.0
        self._previous_gait_actions[env_ids] = 0.0
        self._gait_position_targets[env_ids] = self._asset.data.default_joint_pos[env_ids][
            :, self._policy_joint_ids
        ]
        self._extra_position_targets[env_ids] = 0.0


@configclass
class FrozenGaitAssistTorqueActionCfg(ActionTermCfg):
    """Configuration for the v2 assist action and mechanism hold targets."""

    class_type: type[ActionTerm] = FrozenGaitAssistTorqueAction
    asset_name: str = "robot"
    frozen_policy_path: str = ""
    policy_joint_names: list[str] = POLICY_JOINT_NAMES
    assist_joint_names: list[str] = ASSIST_JOINT_NAMES
    extra_position_joint_names: list[str] = EXTRA_POSITION_JOINT_NAMES
    command_name: str = "base_velocity"
    gait_action_scale: float = 0.25
    assist_torque_limit: float = 8.0
    assist_torque_rate_limit: float = 40.0

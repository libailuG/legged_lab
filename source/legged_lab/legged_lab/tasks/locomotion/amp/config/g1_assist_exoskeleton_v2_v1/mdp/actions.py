"""Frozen gait plus a virtual bilateral hip-assistance transmission."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ...g1_assist_exoskeleton_v2.mdp.actions import (
    ASSIST_JOINT_NAMES,
    EXTRA_POSITION_JOINT_NAMES,
    POLICY_JOINT_NAMES,
)


HIP_PITCH_JOINT_NAMES = ["left_hip_pitch_joint", "right_hip_pitch_joint"]


class FrozenGaitTransmittedAssistTorqueAction(ActionTerm):
    """Apply RL torque to the exoskeleton motors and transmit it to the real hips.

    The USD has no closed-loop thigh attachment.  The same bounded RL torque is
    therefore added as feed-forward effort to each robot hip pitch joint.  This
    provides the load-sharing path that the v1 assistance/burden rewards require.
    """

    cfg: FrozenGaitTransmittedAssistTorqueActionCfg
    _asset: Articulation

    def __init__(self, cfg: FrozenGaitTransmittedAssistTorqueActionCfg, env):
        super().__init__(cfg, env)
        if not os.path.isfile(cfg.frozen_policy_path):
            raise FileNotFoundError(f"Frozen gait policy was not found: {cfg.frozen_policy_path}")

        self._policy_joint_ids, policy_names = self._asset.find_joints(
            cfg.policy_joint_names, preserve_order=True
        )
        self._hip_joint_ids, hip_names = self._asset.find_joints(
            cfg.hip_joint_names, preserve_order=True
        )
        self._assist_joint_ids, assist_names = self._asset.find_joints(
            cfg.assist_joint_names, preserve_order=True
        )
        self._extra_position_joint_ids, extra_names = self._asset.find_joints(
            cfg.extra_position_joint_names, preserve_order=True
        )
        for actual, expected, label in (
            (policy_names, cfg.policy_joint_names, "gait"),
            (hip_names, cfg.hip_joint_names, "hip"),
            (assist_names, cfg.assist_joint_names, "assist"),
            (extra_names, cfg.extra_position_joint_names, "mechanism"),
        ):
            if actual != expected:
                raise RuntimeError(f"{label} joint order mismatch: expected {expected}, got {actual}")

        self._frozen_policy = torch.jit.load(cfg.frozen_policy_path, map_location=self.device)
        self._frozen_policy.eval()
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._gait_actions = torch.zeros(
            self.num_envs, len(self._policy_joint_ids), device=self.device
        )
        self._previous_gait_actions = torch.zeros_like(self._gait_actions)
        self._gait_position_targets = self._asset.data.default_joint_pos[
            :, self._policy_joint_ids
        ].clone()
        self._previous_gait_position_targets = self._gait_position_targets.clone()
        self._hip_velocity_targets = torch.zeros(self.num_envs, 2, device=self.device)
        self._extra_position_targets = torch.zeros(
            self.num_envs, len(self._extra_position_joint_ids), device=self.device
        )
        self._warmup_steps = max(0, math.ceil(cfg.warmup_time_s / env.step_dt))
        self._warmup_steps_remaining = torch.full(
            (self.num_envs,), self._warmup_steps, dtype=torch.long, device=self.device
        )
        self._warmup_active_this_step = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Requested and transmitted assist torque in N.m."""
        return self._processed_actions

    @property
    def normalized_previous_action(self) -> torch.Tensor:
        # Report the action actually sent to the mechanism.  This remains zero
        # during history warmup instead of exposing the ignored policy sample.
        return self._processed_actions / self.cfg.assist_torque_limit

    @property
    def hip_position_targets(self) -> torch.Tensor:
        return self._gait_position_targets[:, self._hip_policy_columns]

    @property
    def hip_velocity_targets(self) -> torch.Tensor:
        return self._hip_velocity_targets

    @property
    def reward_gate(self) -> torch.Tensor:
        return (~self._warmup_active_this_step).float()

    @property
    def primary_hip_torque(self) -> torch.Tensor:
        """Estimated gait-actuator torque before the transmitted assist contribution."""
        return self._asset.data.computed_torque[:, self._hip_joint_ids] - self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._warmup_active_this_step[:] = self._warmup_steps_remaining > 0
        requested = torch.clamp(actions, -1.0, 1.0) * self.cfg.assist_torque_limit
        self._processed_actions[:] = torch.where(
            self._warmup_active_this_step.unsqueeze(-1), 0.0, requested
        )
        self._warmup_steps_remaining.sub_(1).clamp_(min=0)

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
            raise RuntimeError(f"Frozen gait policy expects 96 observations, got {gait_obs.shape[-1]}")
        with torch.inference_mode():
            gait_actions = self._frozen_policy(gait_obs)
        self._gait_actions[:] = gait_actions
        self._previous_gait_actions[:] = gait_actions
        self._previous_gait_position_targets[:] = self._gait_position_targets
        self._gait_position_targets[:] = (
            self._asset.data.default_joint_pos[:, self._policy_joint_ids]
            + self.cfg.gait_action_scale * gait_actions
        )
        self._hip_velocity_targets[:] = (
            self._gait_position_targets[:, self._hip_policy_columns]
            - self._previous_gait_position_targets[:, self._hip_policy_columns]
        ) / self._env.step_dt

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._gait_position_targets, joint_ids=self._policy_joint_ids
        )
        self._asset.set_joint_position_target(
            self._extra_position_targets, joint_ids=self._extra_position_joint_ids
        )
        # Virtual thigh transmission: add assist feed-forward effort at the true hips.
        self._asset.set_joint_effort_target(
            self._processed_actions, joint_ids=self._hip_joint_ids
        )
        # Keep the physical exoskeleton motor/link dynamics active as well.
        self._asset.set_joint_effort_target(
            self._processed_actions, joint_ids=self._assist_joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._gait_actions[env_ids] = 0.0
        self._previous_gait_actions[env_ids] = 0.0
        default = self._asset.data.default_joint_pos[env_ids][:, self._policy_joint_ids]
        self._gait_position_targets[env_ids] = default
        self._previous_gait_position_targets[env_ids] = default
        self._hip_velocity_targets[env_ids] = 0.0
        self._extra_position_targets[env_ids] = 0.0
        self._warmup_steps_remaining[env_ids] = self._warmup_steps
        self._warmup_active_this_step[env_ids] = True

    @property
    def _hip_policy_columns(self) -> list[int]:
        return [self.cfg.policy_joint_names.index(name) for name in self.cfg.hip_joint_names]


@configclass
class FrozenGaitTransmittedAssistTorqueActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FrozenGaitTransmittedAssistTorqueAction
    asset_name: str = "robot"
    frozen_policy_path: str = ""
    policy_joint_names: list[str] = POLICY_JOINT_NAMES
    hip_joint_names: list[str] = HIP_PITCH_JOINT_NAMES
    assist_joint_names: list[str] = ASSIST_JOINT_NAMES
    extra_position_joint_names: list[str] = EXTRA_POSITION_JOINT_NAMES
    command_name: str = "base_velocity"
    gait_action_scale: float = 0.25
    assist_torque_limit: float = 8.0
    warmup_time_s: float = 0.5

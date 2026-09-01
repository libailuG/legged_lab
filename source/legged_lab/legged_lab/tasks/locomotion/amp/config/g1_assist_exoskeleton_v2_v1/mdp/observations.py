"""Deployable actor and privileged critic observations."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def previous_normalized_assist_action(
    env: ManagerBasedEnv, action_name: str = "assist_torque"
) -> torch.Tensor:
    return env.action_manager.get_term(action_name).normalized_previous_action


def privileged_assist_state(
    env: ManagerBasedEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    action_name: str = "assist_torque",
) -> torch.Tensor:
    """Return 20 bilateral features used only by the critic."""
    robot: Articulation = env.scene[hip_cfg.name]
    action = env.action_manager.get_term(action_name)
    hip_pos = robot.data.joint_pos[:, hip_cfg.joint_ids]
    hip_vel = robot.data.joint_vel[:, hip_cfg.joint_ids]
    assist_pos = robot.data.joint_pos[:, assist_cfg.joint_ids]
    assist_vel = robot.data.joint_vel[:, assist_cfg.joint_ids]
    cylinder_pos = robot.data.joint_pos[:, cylinder_cfg.joint_ids]
    target_pos = action.hip_position_targets
    target_vel = action.hip_velocity_targets
    primary_torque = action.primary_hip_torque
    assist_torque = action.processed_actions
    tracking_error = target_pos - hip_pos
    pair_error = cylinder_pos + assist_pos - hip_pos
    return torch.cat(
        (
            hip_pos,
            hip_vel,
            assist_pos,
            assist_vel,
            target_pos,
            target_vel,
            primary_torque,
            assist_torque,
            tracking_error,
            pair_error,
        ),
        dim=-1,
    )

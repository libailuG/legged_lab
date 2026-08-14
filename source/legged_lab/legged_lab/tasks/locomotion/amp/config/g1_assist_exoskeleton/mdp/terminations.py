"""Termination and reset terms for the assist-exoskeleton task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def assist_hip_angle_mismatch(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Terminate when either assist joint differs too far from its paired hip joint."""
    robot: Articulation = env.scene[hip_cfg.name]
    hip_pos = robot.data.joint_pos[:, hip_cfg.joint_ids]
    assist_pos = robot.data.joint_pos[:, assist_cfg.joint_ids]
    return torch.any(torch.abs(assist_pos - hip_pos) > threshold, dim=-1)


def reset_joints_and_sync_assist(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset all joints, then initialize each assist joint at its paired hip angle."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, robot.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, robot.device)

    joint_pos[:, assist_cfg.joint_ids] = joint_pos[:, hip_cfg.joint_ids]
    joint_vel[:, assist_cfg.joint_ids] = joint_vel[:, hip_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    velocity_limits = robot.data.soft_joint_vel_limits[env_ids]
    joint_vel.clamp_(-velocity_limits, velocity_limits)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

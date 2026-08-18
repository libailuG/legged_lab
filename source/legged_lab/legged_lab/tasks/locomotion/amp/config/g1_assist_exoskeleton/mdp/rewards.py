"""Reward terms for assist-joint torque learning."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _motion_command_magnitude(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_scale: float,
) -> torch.Tensor:
    """Return a scalar motion magnitude from planar velocity and yaw-rate commands."""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :2], dim=-1) + yaw_scale * command[:, 2].abs()


def assist_torque_alignment(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    torque_limit: float = 8.0,
    minimum_reference_torque: float = 0.25,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.05,
    yaw_scale: float = 0.3,
) -> torch.Tensor:
    """Reward useful same-direction torque only while a motion command is active.

    A score of one means the assist torque has the same sign and magnitude as
    the corresponding hip-pitch motor torque. Opposite torque receives no useful
    torque score, and torque exceeding the hip torque receives a quadratic penalty.
    The complete term is zero for standing commands so the policy has no incentive
    to follow the body's posture-holding hip torque.
    """
    robot: Articulation = env.scene[hip_cfg.name]
    hip_torque = robot.data.applied_torque[:, hip_cfg.joint_ids]
    assist_torque = robot.data.applied_torque[:, assist_cfg.joint_ids]

    hip_magnitude = hip_torque.abs()
    signed_assistance = assist_torque * torch.sign(hip_torque)
    useful_torque = torch.clamp(signed_assistance, min=0.0)
    useful_torque = torch.minimum(useful_torque, hip_magnitude)
    useful_ratio = useful_torque / hip_magnitude.clamp_min(minimum_reference_torque)

    excess = torch.clamp(assist_torque.abs() - hip_magnitude, min=0.0)
    excess_penalty = torch.square(excess / torque_limit)
    moving = _motion_command_magnitude(env, command_name, yaw_scale) > moving_threshold
    return (useful_ratio - excess_penalty).mean(dim=-1) * moving.float()


def assist_torque_zero_when_standing(
    env: ManagerBasedRLEnv,
    assist_cfg: SceneEntityCfg,
    torque_limit: float = 8.0,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.05,
    yaw_scale: float = 0.3,
) -> torch.Tensor:
    """Penalize normalized assist torque only when the command requests standing."""
    robot: Articulation = env.scene[assist_cfg.name]
    assist_torque = robot.data.applied_torque[:, assist_cfg.joint_ids]
    standing = _motion_command_magnitude(env, command_name, yaw_scale) <= moving_threshold
    normalized_torque_l2 = torch.square(assist_torque / torque_limit).mean(dim=-1)
    return normalized_torque_l2 * standing.float()


def assist_hip_angle_error_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize continuous assist-to-hip joint position mismatch."""
    robot: Articulation = env.scene[hip_cfg.name]
    error = (
        robot.data.joint_pos[:, assist_cfg.joint_ids]
        - robot.data.joint_pos[:, hip_cfg.joint_ids]
    )
    return torch.square(error).mean(dim=-1)


def assist_hip_velocity_error_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize continuous assist-to-hip joint velocity mismatch."""
    robot: Articulation = env.scene[hip_cfg.name]
    error = (
        robot.data.joint_vel[:, assist_cfg.joint_ids]
        - robot.data.joint_vel[:, hip_cfg.joint_ids]
    )
    return torch.square(error).mean(dim=-1)

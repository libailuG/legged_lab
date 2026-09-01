"""Reward terms for v2 assist-joint torque learning."""

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


def assist_torque_rate_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_rate_limit: float = 40.0,
) -> torch.Tensor:
    """Penalize the normalized actual assist-torque rate after rate limiting."""
    action = env.action_manager.get_term(action_name)
    normalized_rate = action.assist_torque_rate / torque_rate_limit
    return torch.square(normalized_rate).mean(dim=-1)


def rear_mechanism_zero_position_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    slider_position_scale: float,
    cylinder_position_scale: float,
) -> torch.Tensor:
    """Penalize normalized slider and cylinder position errors from zero."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 2:
        raise ValueError(f"Expected two rear-mechanism joints, got {joint_pos.shape[-1]}")
    if slider_position_scale <= 0.0 or cylinder_position_scale <= 0.0:
        raise ValueError("Rear-mechanism position scales must be greater than zero")
    slider_error = torch.square(joint_pos[:, 0] / slider_position_scale)
    cylinder_error = torch.square(joint_pos[:, 1] / cylinder_position_scale)
    return 0.5 * (slider_error + cylinder_error)


def assist_torque_zero_at_hip_rest_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    action_name: str = "assist_torque",
    torque_limit: float = 8.0,
    velocity_threshold: float = 0.05,
    acceleration_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize non-zero assist torque when each paired physical hip is at rest."""
    robot: Articulation = env.scene[hip_cfg.name]
    hip_vel = robot.data.joint_vel[:, hip_cfg.joint_ids]
    hip_acc = robot.data.joint_acc[:, hip_cfg.joint_ids]
    assist_torque = env.action_manager.get_term(action_name).processed_actions
    at_rest = (hip_vel.abs() < velocity_threshold) & (
        hip_acc.abs() < acceleration_threshold
    )
    normalized_torque_l2 = torch.square(assist_torque / torque_limit)
    return (normalized_torque_l2 * at_rest.float()).mean(dim=-1)


def assist_torque_hip_dynamics_tracking_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    action_name: str = "assist_torque",
    torque_limit: float = 8.0,
    velocity_gain: float = 0.5,
    acceleration_gain: float = 0.2,
) -> torch.Tensor:
    """Penalize deviation from a velocity- and acceleration-based assist reference."""
    robot: Articulation = env.scene[hip_cfg.name]
    hip_vel = robot.data.joint_vel[:, hip_cfg.joint_ids]
    hip_acc = robot.data.joint_acc[:, hip_cfg.joint_ids]
    assist_torque = env.action_manager.get_term(action_name).processed_actions
    reference_torque = torch.clamp(
        velocity_gain * hip_vel + acceleration_gain * hip_acc,
        min=-torque_limit,
        max=torque_limit,
    )
    return torch.square((assist_torque - reference_torque) / torque_limit).mean(dim=-1)


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
    """Reward useful same-direction assist torque only under motion commands."""
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
    """Penalize normalized assist torque only when standing is commanded."""
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

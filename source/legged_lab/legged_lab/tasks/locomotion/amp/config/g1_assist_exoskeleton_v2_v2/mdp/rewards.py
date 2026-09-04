"""Reward terms for v2 assist-joint torque learning."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


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
    lift_torque_rate_limit: float = 80.0,
    press_torque_rate_limit: float = 40.0,
) -> torch.Tensor:
    """Penalize the normalized actual assist-torque rate after rate limiting."""
    action = env.action_manager.get_term(action_name)
    torque_rate = action.assist_torque_rate
    directional_limit = torch.where(
        torque_rate < 0.0,
        lift_torque_rate_limit,
        press_torque_rate_limit,
    )
    normalized_rate = torque_rate / directional_limit
    return torch.square(normalized_rate).mean(dim=-1)


def assist_requested_torque_rate_excess_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_limit: float = 10.0,
    lift_torque_rate_limit: float = 80.0,
    press_torque_rate_limit: float = 40.0,
) -> torch.Tensor:
    """Penalize only the part of the requested torque step above the slew limit.

    Unlike the post-limiter rate penalty, this remains informative when the
    actuator command is saturated at the configured slew-rate limit.
    """
    if torque_limit <= 0.0 or lift_torque_rate_limit <= 0.0 or press_torque_rate_limit <= 0.0:
        raise ValueError("Torque and torque-rate limits must be greater than zero")
    action = env.action_manager.get_term(action_name)
    requested_delta = action.requested_torque_delta
    allowed_delta = torch.where(
        requested_delta < 0.0,
        lift_torque_rate_limit,
        press_torque_rate_limit,
    ) * env.step_dt
    excess = torch.clamp(requested_delta.abs() - allowed_delta, min=0.0)
    # Normalize by the per-control-step allowance.  Normalizing by the full
    # torque range made large, rate-saturated requests look artificially small.
    return torch.square(excess / allowed_delta).mean(dim=-1)


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


def hip_motor_torque_burden_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    torque_scale: float = 40.0,
) -> torch.Tensor:
    """Penalize the physical hip motors' normalized squared torque burden."""
    if torque_scale <= 0.0:
        raise ValueError("torque_scale must be greater than zero")
    robot: Articulation = env.scene[hip_cfg.name]
    hip_torque = robot.data.applied_torque[:, hip_cfg.joint_ids]
    return torch.square(hip_torque / torque_scale).mean(dim=-1)


def hip_motor_mechanical_power_l1(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    power_scale: float = 100.0,
) -> torch.Tensor:
    """Penalize absolute physical hip-motor mechanical power."""
    if power_scale <= 0.0:
        raise ValueError("power_scale must be greater than zero")
    robot: Articulation = env.scene[hip_cfg.name]
    hip_torque = robot.data.applied_torque[:, hip_cfg.joint_ids]
    hip_vel = robot.data.joint_vel[:, hip_cfg.joint_ids]
    return torch.abs(hip_torque * hip_vel).mean(dim=-1) / power_scale


class FilteredHipDynamicsReward(ManagerTermBase):
    """Stateful assist tracking/rest reward with low-pass-filtered hip acceleration."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered_acceleration = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._filtered_acceleration[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        mode: str,
        hip_cfg: SceneEntityCfg,
        action_name: str = "assist_torque",
        torque_limit: float = 10.0,
        lift_torque_limit: float = 10.0,
        press_torque_limit: float = 4.0,
        velocity_gain: float = 0.5,
        acceleration_gain: float = 0.2,
        dynamics_scale: float = 1.5,
        acceleration_filter_time_constant: float = 0.08,
        rest_velocity_scale: float = 0.5,
        rest_acceleration_scale: float = 3.0,
    ) -> torch.Tensor:
        if torque_limit <= 0.0:
            raise ValueError("torque_limit must be greater than zero")
        if lift_torque_limit <= 0.0 or press_torque_limit <= 0.0:
            raise ValueError("lift_torque_limit and press_torque_limit must be greater than zero")
        if acceleration_filter_time_constant <= 0.0:
            raise ValueError("acceleration_filter_time_constant must be greater than zero")
        if dynamics_scale <= 0.0:
            raise ValueError("dynamics_scale must be greater than zero")

        robot: Articulation = env.scene[hip_cfg.name]
        hip_vel = robot.data.joint_vel[:, hip_cfg.joint_ids]
        hip_acc = robot.data.joint_acc[:, hip_cfg.joint_ids]
        action = env.action_manager.get_term(action_name)
        assist_torque = action.processed_actions

        filter_alpha = 1.0 - math.exp(
            -env.step_dt / acceleration_filter_time_constant
        )
        self._filtered_acceleration.lerp_(hip_acc, filter_alpha)

        if mode == "tracking":
            dynamics_signal = (
                velocity_gain * hip_vel
                + acceleration_gain * self._filtered_acceleration
            )
            normalized_reference = torch.tanh(dynamics_signal / dynamics_scale)
            reference_limit = torch.where(
                normalized_reference < 0.0,
                lift_torque_limit,
                press_torque_limit,
            )
            reference_torque = (
                reference_limit
                * action.motion_gate
                * normalized_reference
            )
            return torch.square(
                (assist_torque - reference_torque) / torque_limit
            ).mean(dim=-1)

        if mode == "continuous_rest":
            if rest_velocity_scale <= 0.0 or rest_acceleration_scale <= 0.0:
                raise ValueError("rest motion scales must be greater than zero")
            normalized_motion_l2 = torch.square(
                hip_vel / rest_velocity_scale
            ) + torch.square(
                self._filtered_acceleration / rest_acceleration_scale
            )
            rest_weight = 1.0 / (1.0 + normalized_motion_l2)
            normalized_torque_l2 = torch.square(assist_torque / torque_limit)
            return (normalized_torque_l2 * rest_weight).mean(dim=-1)

        raise ValueError(f"Unsupported filtered hip-dynamics reward mode: {mode}")


def assist_torque_alignment(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    torque_limit: float = 10.0,
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
    torque_limit: float = 10.0,
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


def assist_hip_angle_tracking_exp(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward assist-to-hip angle agreement with a peak value of one at zero error."""
    if std <= 0.0:
        raise ValueError("Angle tracking standard deviation must be greater than zero")
    robot: Articulation = env.scene[hip_cfg.name]
    error = (
        robot.data.joint_pos[:, assist_cfg.joint_ids]
        - robot.data.joint_pos[:, hip_cfg.joint_ids]
    )
    return torch.exp(-torch.square(error).mean(dim=-1) / std**2)


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

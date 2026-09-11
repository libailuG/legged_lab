"""Observation terms for the v2 assist-only PPO."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def privileged_foot_support(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 10.0,
) -> torch.Tensor:
    """Critic-only [left Fz, right Fz, left contact, right contact].

    Fz is the foot-link net normal contact force's world-Z component in N.
    The threshold detects contact, not whether the leg can accept full assist.
    """
    sensor = env.scene[sensor_cfg.name]
    force_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contact = (force_z > contact_threshold).to(force_z.dtype)
    return torch.cat((force_z, contact), dim=-1)


def commanded_assist_torque(
    env: ManagerBasedEnv, action_name: str = "assist_torque"
) -> torch.Tensor:
    """Return the current commanded assist torque in N.m."""
    return env.action_manager.get_term(action_name).processed_actions


def assist_hip_angle_difference(
    env: ManagerBasedEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return assist-joint angle minus its paired physical hip angle in radians."""
    robot: Articulation = env.scene[hip_cfg.name]
    return (
        robot.data.joint_pos[:, assist_cfg.joint_ids]
        - robot.data.joint_pos[:, hip_cfg.joint_ids]
    )


def applied_joint_torque(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the applied torque of the selected joints in N.m."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.applied_torque[:, asset_cfg.joint_ids]


def joint_acceleration(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return finite-difference acceleration of the selected joints in rad/s^2."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_acc[:, asset_cfg.joint_ids]

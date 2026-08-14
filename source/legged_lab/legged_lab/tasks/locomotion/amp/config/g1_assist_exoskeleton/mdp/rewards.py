"""Reward terms for assist-joint torque learning."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def assist_torque_alignment(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    torque_limit: float = 8.0,
    minimum_reference_torque: float = 0.25,
) -> torch.Tensor:
    """Reward useful same-direction torque and penalize exceeding hip torque.

    A score of one means the assist torque has the same sign and magnitude as
    the corresponding hip-pitch motor torque. Opposite torque receives no useful
    torque score, and torque exceeding the hip torque receives a quadratic penalty.
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
    return (useful_ratio - excess_penalty).mean(dim=-1)

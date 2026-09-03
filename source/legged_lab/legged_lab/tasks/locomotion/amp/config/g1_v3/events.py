"""Reset helpers for the 29-DoF policy running on the 33-joint v3 asset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def reset_body_joints_from_ref(
    env: ManagerBasedAnimationEnv,
    env_ids: torch.Tensor,
    animation: str,
    asset_cfg: SceneEntityCfg,
    height_offset: float = 0.1,
) -> None:
    """Reset root and only the 29 G1 body joints from AMP reference data."""

    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    offset = torch.tensor(
        [0.0, 0.0, height_offset], device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    position = (
        animation_term.get_root_pos_w(env_ids)[:, 0, :]
        + env.scene.env_origins[env_ids, :]
        + offset
    )
    orientation = animation_term.get_root_quat(env_ids)[:, 0, :]
    root_pose = torch.cat((position, orientation), dim=-1)
    root_velocity = torch.cat(
        (
            animation_term.get_root_vel_w(env_ids)[:, 0, :],
            animation_term.get_root_ang_vel_w(env_ids)[:, 0, :],
        ),
        dim=-1,
    )
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

    joint_position = animation_term.get_dof_pos(env_ids)[:, 0, :]
    joint_velocity = animation_term.get_dof_vel(env_ids)[:, 0, :]
    robot.write_joint_state_to_sim(
        joint_position,
        joint_velocity,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )

"""Opt-in standing reinforcement for assist-v1 train and play configurations."""

import torch

from isaaclab.managers import RewardTermCfg


def _standing_mask(env):
    # Only true zero commands; small walking commands keep their usual rewards.
    return torch.all(torch.abs(env.command_manager.get_command("base_velocity")) < 1.0e-6, dim=1)


def standing_stability(env):
    """Reward a quiet, upright base only while all velocity commands are zero."""
    data = env.scene["robot"].data
    error = (
        torch.sum(data.root_lin_vel_b.square(), dim=1) / 0.15**2
        + torch.sum(data.root_ang_vel_b.square(), dim=1) / 0.3**2
        + torch.sum(data.projected_gravity_b[:, :2].square(), dim=1) / 0.2**2
    )
    upright = torch.clamp(-data.projected_gravity_b[:, 2], 0.0, 1.0)
    return _standing_mask(env) * upright * torch.exp(-error)


def standing_joint_motion(env):
    """Penalize joint motion at zero commands to discourage stepping in place."""
    return _standing_mask(env) * torch.sum(env.scene["robot"].data.joint_vel.square(), dim=1)


def enable_standing_training(cfg):
    """Apply before environment creation; preserves all existing moving rewards."""
    if not hasattr(cfg, "standing_training"):
        raise ValueError("--standing-training is supported only by G1 assist-v1 train/Play tasks")
    cfg.standing_training = True
    cfg.commands.base_velocity.rel_standing_envs = 0.35
    cfg.rewards.standing_stability = RewardTermCfg(func=standing_stability, weight=3.0)
    cfg.rewards.standing_joint_motion = RewardTermCfg(func=standing_joint_motion, weight=-0.02)
    print("[INFO] Standing reinforcement enabled: sampling=35%, stability=3.0, joint_motion=-0.02")

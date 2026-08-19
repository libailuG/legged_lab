"""Observation terms for the independent v1 assist-only PPO."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv


def commanded_assist_torque(
    env: ManagerBasedEnv, action_name: str = "assist_torque"
) -> torch.Tensor:
    """Return the current commanded assist torque in N.m."""
    return env.action_manager.get_term(action_name).processed_actions

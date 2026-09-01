"""Template-Assist-Direct-v1 rewards adapted to bilateral 50-Hz locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def _action(env: ManagerBasedRLEnv, action_name: str):
    return env.action_manager.get_term(action_name)


def _gate(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    return _action(env, action_name).reward_gate


def maximum_assistance(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_limit: float = 8.0,
) -> torch.Tensor:
    action = _action(env, action_name)
    primary = action.primary_hip_torque
    assist = action.processed_actions
    target = torch.clamp(primary + assist, -torque_limit, torque_limit)
    target_error = torch.abs(assist - target) / torque_limit
    return torch.clamp(1.0 - target_error, -1.0, 1.0).mean(-1) * action.reward_gate


def primary_hip_burden(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_limit: float = 8.0,
) -> torch.Tensor:
    action = _action(env, action_name)
    value = torch.clamp(torch.abs(action.primary_hip_torque) / torque_limit, max=2.0)
    return value.mean(-1) * action.reward_gate


def hip_position_tracking_error_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    action_name: str = "assist_torque",
) -> torch.Tensor:
    robot: Articulation = env.scene[hip_cfg.name]
    action = _action(env, action_name)
    error = action.hip_position_targets - robot.data.joint_pos[:, hip_cfg.joint_ids]
    return torch.square(error).mean(-1) * action.reward_gate


def hip_velocity_tracking_error_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    action_name: str = "assist_torque",
) -> torch.Tensor:
    robot: Articulation = env.scene[hip_cfg.name]
    action = _action(env, action_name)
    error = action.hip_velocity_targets - robot.data.joint_vel[:, hip_cfg.joint_ids]
    return torch.square(error).mean(-1) * action.reward_gate


def physical_pair_soft_limit(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    soft_limit: float,
    penalty_width: float,
    action_name: str = "assist_torque",
) -> torch.Tensor:
    robot: Articulation = env.scene[hip_cfg.name]
    error = (
        robot.data.joint_pos[:, cylinder_cfg.joint_ids]
        + robot.data.joint_pos[:, assist_cfg.joint_ids]
        - robot.data.joint_pos[:, hip_cfg.joint_ids]
    )
    excess = torch.clamp(torch.abs(error) - soft_limit, min=0.0)
    return torch.square(excess / penalty_width).mean(-1) * _gate(env, action_name)


def counter_torque(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_limit: float = 8.0,
) -> torch.Tensor:
    action = _action(env, action_name)
    value = torch.clamp(
        -(action.primary_hip_torque * action.processed_actions) / torque_limit**2,
        min=0.0,
    )
    return value.mean(-1) * action.reward_gate


def counter_torque_direction(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    primary_threshold: float = 0.2,
    assist_threshold: float = 0.5,
) -> torch.Tensor:
    action = _action(env, action_name)
    primary = action.primary_hip_torque
    assist = action.processed_actions
    fighting = (
        (primary * assist < 0.0)
        & (torch.abs(primary) > primary_threshold)
        & (torch.abs(assist) > assist_threshold)
    )
    return fighting.float().mean(-1) * action.reward_gate


def assist_effort_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_limit: float = 8.0,
) -> torch.Tensor:
    action = _action(env, action_name)
    return torch.square(action.processed_actions / torque_limit).mean(-1) * action.reward_gate


def assist_saturation(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_limit: float = 8.0,
    threshold: float = 0.9,
) -> torch.Tensor:
    action = _action(env, action_name)
    excess = torch.clamp(torch.abs(action.processed_actions) / torque_limit - threshold, min=0.0)
    return torch.square(excess / (1.0 - threshold)).mean(-1) * action.reward_gate


def assist_zero_when_standing(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    command_name: str = "base_velocity",
    torque_limit: float = 8.0,
    moving_threshold: float = 0.05,
    yaw_scale: float = 0.3,
) -> torch.Tensor:
    action = _action(env, action_name)
    command = env.command_manager.get_command(command_name)
    magnitude = torch.linalg.vector_norm(command[:, :2], dim=-1) + yaw_scale * command[:, 2].abs()
    standing = magnitude <= moving_threshold
    effort = torch.square(action.processed_actions / torque_limit).mean(-1)
    return effort * standing.float() * action.reward_gate


def gated_termination(
    env: ManagerBasedRLEnv, action_name: str = "assist_torque"
) -> torch.Tensor:
    return env.termination_manager.terminated.float() * _gate(env, action_name)


class AssistTorqueTemporalPenalty(ManagerTermBase):
    """Stateful torque delta, chatter, jerk, or requested-rate penalty."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous = torch.zeros(env.num_envs, 2, device=env.device)
        self._previous_delta = torch.zeros_like(self._previous)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous[env_ids] = 0.0
        self._previous_delta[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        mode: str,
        action_name: str = "assist_torque",
        torque_limit: float = 8.0,
        max_requested_delta: float = 0.8,
    ) -> torch.Tensor:
        action = _action(env, action_name)
        torque = action.processed_actions
        delta = torque - self._previous
        if mode == "delta_l2":
            value = torch.square(delta).mean(-1)
        elif mode == "delta_l1":
            value = torch.abs(delta).mean(-1)
        elif mode == "jerk":
            value = torch.square(delta - self._previous_delta).mean(-1)
        elif mode == "requested_rate":
            excess = torch.clamp(torch.abs(delta) - max_requested_delta, min=0.0)
            value = torch.square(excess / torque_limit).mean(-1)
        else:
            raise ValueError(f"Unsupported temporal penalty mode: {mode}")
        self._previous[:] = torque
        self._previous_delta[:] = delta
        return value * action.reward_gate

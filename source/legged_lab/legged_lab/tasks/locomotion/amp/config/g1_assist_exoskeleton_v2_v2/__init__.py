"""Gym registrations for the asymmetric G1 assist-exoskeleton v2-v2 tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_assist_exoskeleton_env_cfg:G1AssistExoskeletonV2V2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1AssistExoskeletonV2V2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.g1_assist_exoskeleton_env_cfg:G1AssistExoskeletonV2V2EnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1AssistExoskeletonV2V2PPORunnerCfg"
        ),
    },
)

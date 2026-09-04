"""Gym registrations for the independent G1 assist AMP v1 tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="LeggedLab-Isaac-AMP-G1-assist-v1",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_assist_amp_env_cfg:G1AssistV1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1AssistV1RslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-assist-Play-v1",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_assist_amp_env_cfg:G1AssistV1AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G1AssistV1RslRlOnPolicyRunnerAmpCfg"
        ),
    },
)

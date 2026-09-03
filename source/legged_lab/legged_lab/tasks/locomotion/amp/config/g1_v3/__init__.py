"""Gym registrations for the G1 AMP v3 PID tasks."""

import gymnasium as gym

from . import agents

from legged_lab.envs import ManagerBasedAmpEnv


gym.register(
    id="LeggedLab-Isaac-AMP-G1-v3",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1V3RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-G1-Play-v3",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpEnvCfg_PLAY_V3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1V3RslRlOnPolicyRunnerAmpCfg",
    },
)

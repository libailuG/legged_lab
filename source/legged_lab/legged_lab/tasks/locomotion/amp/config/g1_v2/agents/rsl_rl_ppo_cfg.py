"""RSL-RL configuration copied from G1 AMP v0 for the PID experiment."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RslRlOnPolicyRunnerAmpCfg,
)


@configclass
class G1V2RslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Use the v0 PPO/AMP settings with a separate experiment directory."""

    experiment_name = "g1_amp_v2"

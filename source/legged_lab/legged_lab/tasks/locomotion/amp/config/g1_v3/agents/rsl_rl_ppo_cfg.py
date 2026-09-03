"""RSL-RL configuration copied from G1 AMP v2 for the v3 experiment."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import (
    G1RslRlOnPolicyRunnerAmpCfg,
)


@configclass
class G1V3RslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Use the v2 PPO/AMP settings with a separate experiment directory."""

    experiment_name = "g1_amp_v3"

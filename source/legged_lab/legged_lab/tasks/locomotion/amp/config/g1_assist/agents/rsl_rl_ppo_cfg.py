"""RSL-RL configuration for G1 assist AMP tasks."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import G1RslRlOnPolicyRunnerAmpCfg


@configclass
class G1AssistRslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Use a separate experiment directory from the original G1 task."""

    experiment_name = "g1_assist_amp"

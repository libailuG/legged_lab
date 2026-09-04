"""RSL-RL configuration for the independent G1 assist AMP v1 tasks."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import G1RslRlOnPolicyRunnerAmpCfg


@configclass
class G1AssistV1RslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Use a log directory independent from both the base G1 and assist v0 tasks."""

    experiment_name = "g1_assist_v1_amp"

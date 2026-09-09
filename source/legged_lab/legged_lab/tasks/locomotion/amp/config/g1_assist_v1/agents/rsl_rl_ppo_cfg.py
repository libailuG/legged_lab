"""RSL-RL configuration for the independent G1 assist AMP v1 tasks."""

import copy

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.agents.rsl_rl_ppo_cfg import G1RslRlOnPolicyRunnerAmpCfg


@configclass
class G1AssistV1RslRlOnPolicyRunnerAmpCfg(G1RslRlOnPolicyRunnerAmpCfg):
    """Use a log directory independent from both the base G1 and assist v0 tasks."""

    experiment_name = "g1_assist_v1_amp"

    # Keep exploration controlled during long policy-transfer runs.  Deep-copy
    # the inherited object so these v1-specific defaults do not mutate the
    # base G1 AMP task or other derived task configurations.
    algorithm = copy.deepcopy(G1RslRlOnPolicyRunnerAmpCfg().algorithm)
    algorithm.entropy_coef = 0.0
    algorithm.min_noise_std = 0.05
    algorithm.max_noise_std = 0.5

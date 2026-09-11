"""PPO for v7: 450 actor inputs and latched peak/duration actions."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BoundedNoisePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with hard bounds on state-independent action noise."""

    min_noise_std: float = 0.15
    max_noise_std: float = 0.5


@configclass
class G1AssistExoskeletonV2V7PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 192
    max_iterations = 10000
    save_interval = 100
    experiment_name = "g1_assist_exoskeleton_v2_v7_ppo"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = None  # Preserve raw output for bounds cost; pulse generator clips physical parameters.

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        # Two exploration standard deviations for peak and duration.
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 64, 16],
        critic_hidden_dims=[256, 64, 16],
        activation="elu",
    )
    algorithm = BoundedNoisePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.999,
        lam=0.995,
        desired_kl=0.01,
        max_grad_norm=0.5,
        normalize_advantage_per_mini_batch=True,
        min_noise_std=0.15,
        max_noise_std=0.5,
    )

"""AMP environment configurations for the independent G1 assist v1 task."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1AmpEnvCfg, G1AmpEnvCfg_PLAY

from .robot_cfg import UNITREE_G1_29DOF_ASSIST_V1_CFG


def _configure_inner_loop_timing(cfg: G1AmpEnvCfg) -> None:
    """Run physics and the actuator inner loop at 1 kHz.

    The decimation is increased from 4 to 20 so the policy period remains
    unchanged at 20 ms.
    """

    cfg.sim.dt = 0.001
    cfg.decimation = 20
    cfg.sim.render_interval = cfg.decimation
    if cfg.scene.contact_forces is not None:
        cfg.scene.contact_forces.update_period = cfg.sim.dt


@configclass
class G1AssistV1AmpEnvCfg(G1AmpEnvCfg):
    """Training configuration for the independently editable G1 assist v1 task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_ASSIST_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _configure_inner_loop_timing(self)


@configclass
class G1AssistV1AmpEnvCfg_PLAY(G1AmpEnvCfg_PLAY):
    """Play configuration for the independently editable G1 assist v1 task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_ASSIST_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _configure_inner_loop_timing(self)

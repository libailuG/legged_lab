"""G1 AMP v2: v0 behavior with 1 kHz explicit PID joint control."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpEnvCfg,
    G1AmpEnvCfg_PLAY,
)

from .robot_cfg import UNITREE_G1_29DOF_PID_CFG


def _configure_pid_timing(cfg: G1AmpEnvCfg) -> None:
    """Keep the 20 ms policy period while running PID and physics at 1 ms."""

    cfg.decimation = 20
    cfg.sim.dt = 0.001
    cfg.sim.render_interval = cfg.decimation
    if cfg.scene.contact_forces is not None:
        cfg.scene.contact_forces.update_period = cfg.sim.dt


@configclass
class G1AmpEnvCfgV2(G1AmpEnvCfg):
    """Training configuration copied from v0 with PID actuators."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_PID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _configure_pid_timing(self)


@configclass
class G1AmpEnvCfg_PLAY_V2(G1AmpEnvCfg_PLAY):
    """Play configuration corresponding to :class:`G1AmpEnvCfgV2`."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_PID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _configure_pid_timing(self)

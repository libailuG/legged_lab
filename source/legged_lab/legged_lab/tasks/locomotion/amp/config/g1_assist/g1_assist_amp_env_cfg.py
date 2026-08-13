"""AMP environment configurations for the mass-adjusted G1 assist robot."""

from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1AmpEnvCfg, G1AmpEnvCfg_PLAY

from .robot_cfg import UNITREE_G1_29DOF_ASSIST_CFG


@configclass
class G1AssistAmpEnvCfg(G1AmpEnvCfg):
    """Training configuration using the generated G1 assist USD asset."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_ASSIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1AssistAmpEnvCfg_PLAY(G1AmpEnvCfg_PLAY):
    """Play configuration using the generated G1 assist USD asset."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_ASSIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

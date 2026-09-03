"""G1 AMP v3: explicit PID control with the assist-exoskeleton v2 USD."""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import (
    G1AmpEnvCfg,
    G1AmpEnvCfg_PLAY,
)
from .events import reset_body_joints_from_ref
from .robot_cfg import G1_BODY_JOINT_NAMES, UNITREE_G1_29DOF_PID_V3_CFG


def _body_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot", joint_names=G1_BODY_JOINT_NAMES, preserve_order=True
    )


def _configure_pid_timing(cfg: G1AmpEnvCfg) -> None:
    """Keep the 20 ms policy period while running PID and physics at 1 ms."""

    cfg.decimation = 20
    cfg.sim.dt = 0.001
    cfg.sim.render_interval = cfg.decimation
    if cfg.scene.contact_forces is not None:
        cfg.scene.contact_forces.update_period = cfg.sim.dt


def _configure_v3_robot(cfg: G1AmpEnvCfg) -> None:
    """Install the v3 PID robot with its configured exoskeleton USD."""

    cfg.scene.robot = UNITREE_G1_29DOF_PID_V3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # The selected USD has four extra exoskeleton/mechanism joints. Keep the
    # original AMP policy, critic and discriminator strictly on the 29 G1 body
    # joints so their dimensions and motion-data order remain unchanged.
    cfg.actions.joint_pos.joint_names = G1_BODY_JOINT_NAMES
    cfg.observations.policy.joint_pos.params["asset_cfg"] = _body_joint_cfg()
    cfg.observations.policy.joint_vel.params["asset_cfg"] = _body_joint_cfg()
    cfg.observations.critic.joint_pos.params["asset_cfg"] = _body_joint_cfg()
    cfg.observations.critic.joint_vel.params["asset_cfg"] = _body_joint_cfg()
    cfg.observations.disc.joint_pos.params["asset_cfg"] = _body_joint_cfg()
    cfg.observations.disc.joint_vel.params["asset_cfg"] = _body_joint_cfg()

    cfg.rewards.dof_torques_l2.params["asset_cfg"] = _body_joint_cfg()
    cfg.rewards.dof_acc_l2.params["asset_cfg"] = _body_joint_cfg()
    cfg.events.scale_actuator_gains.params["asset_cfg"] = _body_joint_cfg()
    cfg.events.scale_joint_parameters.params["asset_cfg"] = _body_joint_cfg()
    cfg.events.reset_from_ref = EventTerm(
        func=reset_body_joints_from_ref,
        mode="reset",
        params={
            "animation": "animation",
            "asset_cfg": _body_joint_cfg(),
            "height_offset": 0.1,
        },
    )
    _configure_pid_timing(cfg)


@configclass
class G1AmpEnvCfgV3(G1AmpEnvCfg):
    """Training configuration using assist-v0 gains and the exoskeleton USD."""

    def __post_init__(self):
        super().__post_init__()
        _configure_v3_robot(self)


@configclass
class G1AmpEnvCfg_PLAY_V3(G1AmpEnvCfg_PLAY):
    """Play configuration corresponding to :class:`G1AmpEnvCfgV3`."""

    def __post_init__(self):
        super().__post_init__()
        _configure_v3_robot(self)

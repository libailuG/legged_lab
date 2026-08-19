"""Independent environment configuration for G1 assist-exoskeleton v1."""

import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.config.g1_assist.g1_assist_amp_env_cfg import (
    G1AssistAmpEnvCfg,
)

from . import mdp
from .robot_cfg import UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V1_CFG


FROZEN_GAIT_POLICY_PATH = os.path.abspath(
    os.path.join(
        LEGGED_LAB_ROOT_DIR,
        "..",
        "..",
        "..",
        "logs",
        "rsl_rl",
        "g1_assist_amp",
        "2026-08-13_13-45-16",
        "exported",
        "policy.pt",
    )
)
ASSIST_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["left_hip_pitch_assist_joint", "right_hip_pitch_assist_joint"],
    preserve_order=True,
)
HIP_PITCH_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"],
    preserve_order=True,
)


@configclass
class AssistActionsCfg:
    """The v1 learning policy controls only two exoskeleton torques."""

    assist_torque = mdp.FrozenGaitAssistTorqueActionCfg(
        asset_name="robot",
        frozen_policy_path=FROZEN_GAIT_POLICY_PATH,
        assist_torque_limit=8.0,
    )


@configclass
class AssistObservationGroup(ObsGroup):
    """Twenty-five 50-Hz samples: 0.5 s of noisy velocity and torque."""

    assist_joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": ASSIST_JOINT_CFG},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    assist_torque = ObsTerm(
        func=mdp.commanded_assist_torque,
        params={"action_name": "assist_torque"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
    )

    def __post_init__(self):
        self.history_length = 25
        self.flatten_history_dim = True
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class AssistObservationsCfg:
    """Actor and critic receive the independent v1 100-dimensional history."""

    policy: AssistObservationGroup = AssistObservationGroup()
    critic: AssistObservationGroup = AssistObservationGroup()


@configclass
class AssistRewardsCfg:
    """V1 assist rewards including standing and joint-mismatch behavior."""

    assist_output_smoothness = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    assist_torque_alignment = RewTerm(
        func=mdp.assist_torque_alignment,
        weight=1.0,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
            "torque_limit": 8.0,
            "command_name": "base_velocity",
            "moving_threshold": 0.05,
            "yaw_scale": 0.3,
        },
    )
    assist_torque_zero_when_standing = RewTerm(
        func=mdp.assist_torque_zero_when_standing,
        weight=-1.0,
        params={
            "assist_cfg": ASSIST_JOINT_CFG,
            "torque_limit": 8.0,
            "command_name": "base_velocity",
            "moving_threshold": 0.05,
            "yaw_scale": 0.3,
        },
    )
    assist_hip_angle_error = RewTerm(
        func=mdp.assist_hip_angle_error_l2,
        weight=-5.0,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
        },
    )
    assist_hip_velocity_error = RewTerm(
        func=mdp.assist_hip_velocity_error_l2,
        weight=-0.05,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class AssistTerminationsCfg:
    """Original G1 fall conditions plus a 4-degree exoskeleton mismatch."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )
    assist_hip_angle_mismatch = DoneTerm(
        func=mdp.assist_hip_angle_mismatch,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
            "threshold": math.radians(4.0),
        },
    )


@configclass
class G1AssistExoskeletonV1EnvCfg(G1AssistAmpEnvCfg):
    """Train the independent v1 two-action PPO with a frozen gait policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions = AssistActionsCfg()
        self.observations = AssistObservationsCfg()
        self.rewards = AssistRewardsCfg()
        self.terminations = AssistTerminationsCfg()

        self.events.reset_from_ref = None
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_and_sync_assist,
            mode="reset",
            params={
                "position_range": (0.8, 1.2),
                "velocity_range": (0.0, 0.0),
                "hip_cfg": HIP_PITCH_JOINT_CFG,
                "assist_cfg": ASSIST_JOINT_CFG,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.sim.dt = 0.001
        self.decimation = 20
        self.episode_length_s = 20.0
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1AssistExoskeletonV1EnvCfg_PLAY(G1AssistExoskeletonV1EnvCfg):
    """Play configuration for the independent v1 assist policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

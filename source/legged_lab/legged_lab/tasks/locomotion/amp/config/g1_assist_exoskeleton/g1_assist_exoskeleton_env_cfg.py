"""Environment configuration for learning G1 exoskeleton assistance torques."""

import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.config.g1_assist.g1_assist_amp_env_cfg import (
    G1AssistAmpEnvCfg,
)

from . import mdp
from .robot_cfg import UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG


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
    """The learning policy controls only two exoskeleton torques."""

    assist_torque = mdp.FrozenGaitAssistTorqueActionCfg(
        asset_name="robot",
        frozen_policy_path=FROZEN_GAIT_POLICY_PATH,
        assist_torque_limit=8.0,
    )


@configclass
class AssistObservationGroup(ObsGroup):
    """Twenty-five 50-Hz samples: 0.5 s of angle, velocity and torque."""

    assist_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": ASSIST_JOINT_CFG})
    assist_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": ASSIST_JOINT_CFG})
    assist_torque = ObsTerm(
        func=mdp.commanded_assist_torque,
        params={"action_name": "assist_torque"},
    )

    def __post_init__(self):
        self.history_length = 25
        self.flatten_history_dim = True
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class AssistObservationsCfg:
    """Actor and critic both receive the requested 150-dimensional history."""

    policy: AssistObservationGroup = AssistObservationGroup()
    critic: AssistObservationGroup = AssistObservationGroup()


@configclass
class AssistRewardsCfg:
    """Only the three requested reward objectives are enabled."""

    assist_output_smoothness = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    assist_torque_alignment = RewTerm(
        func=mdp.assist_torque_alignment,
        weight=1.0,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
            "torque_limit": 8.0,
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
class G1AssistExoskeletonEnvCfg(G1AssistAmpEnvCfg):
    """Train a two-action PPO while a frozen policy walks the other 29 joints."""

    def __post_init__(self):
        # Build the established G1-assist scene/events/commands first, then
        # replace every learning-facing component with independent configs.
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions = AssistActionsCfg()
        self.observations = AssistObservationsCfg()
        self.rewards = AssistRewardsCfg()
        self.terminations = AssistTerminationsCfg()

        # A generic PPO environment has no AMP animation manager.
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
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1AssistExoskeletonEnvCfg_PLAY(G1AssistExoskeletonEnvCfg):
    """Play configuration for the learned two-joint assist policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

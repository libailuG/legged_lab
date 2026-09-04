"""Environment configuration for asymmetric G1 assist-exoskeleton v2-v2."""

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
from .robot_cfg import UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG


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
REAR_MECHANISM_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "pelvis_rear_upper_box_assist_joint",
        "pelvis_rear_cylinder_assist_joint",
    ],
    preserve_order=True,
)


@configclass
class AssistActionsCfg:
    """The policy uses strong lift and softer press torque authority."""

    assist_torque = mdp.FrozenGaitAssistTorqueActionCfg(
        asset_name="robot",
        frozen_policy_path=FROZEN_GAIT_POLICY_PATH,
        lift_torque_limit=10.0,
        press_torque_limit=4.0,
        lift_torque_rate_limit=80.0,
        press_torque_rate_limit=40.0,
        motion_speed_deadzone=0.15,
        motion_speed_full=0.8,
        motion_filter_time_constant=0.05,
    )


@configclass
class AssistPolicyObservationGroup(ObsGroup):
    """Deployable 0.25-s history of assist angle, velocity, and commanded torque."""

    assist_joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": ASSIST_JOINT_CFG},
        noise=Unoise(n_min=-math.radians(0.5), n_max=math.radians(0.5)),
    )

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
class AssistCriticObservationGroup(AssistPolicyObservationGroup):
    """Privileged robot, hip, base, and rear-mechanism state for the critic."""

    hip_joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": HIP_PITCH_JOINT_CFG},
    )
    hip_joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": HIP_PITCH_JOINT_CFG},
    )
    hip_joint_acc = ObsTerm(
        func=mdp.joint_acceleration,
        params={"asset_cfg": HIP_PITCH_JOINT_CFG},
    )
    assist_hip_angle_difference = ObsTerm(
        func=mdp.assist_hip_angle_difference,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
        },
    )
    hip_applied_torque = ObsTerm(
        func=mdp.applied_joint_torque,
        params={"asset_cfg": HIP_PITCH_JOINT_CFG},
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    rear_mechanism_joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": REAR_MECHANISM_JOINT_CFG},
    )
    rear_mechanism_joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": REAR_MECHANISM_JOINT_CFG},
    )

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class AssistObservationsCfg:
    """Deployable policy observations and asymmetric privileged critic observations."""

    policy: AssistPolicyObservationGroup = AssistPolicyObservationGroup()
    critic: AssistCriticObservationGroup = AssistCriticObservationGroup()


@configclass
class AssistRewardsCfg:
    """V2 rewards for smooth assistance that reduces physical hip burden."""

    assist_output_smoothness = RewTerm(
        func=mdp.assist_torque_rate_l2,
        weight=-0.2,
        params={
            "action_name": "assist_torque",
            "lift_torque_rate_limit": 80.0,
            "press_torque_rate_limit": 40.0,
        },
    )
    assist_requested_rate_excess = RewTerm(
        func=mdp.assist_requested_torque_rate_excess_l2,
        weight=-0.005,
        params={
            "action_name": "assist_torque",
            "torque_limit": 10.0,
            "lift_torque_rate_limit": 80.0,
            "press_torque_rate_limit": 40.0,
        },
    )
    # Temporarily disabled while retaining the reward implementations for later use.
    assist_torque_alignment = None
    assist_torque_zero_when_standing = None
    assist_hip_angle_error = RewTerm(
        func=mdp.assist_hip_angle_error_l2,
        weight=-5.0,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
        },
    )
    assist_hip_angle_tracking = None
    assist_hip_velocity_error = RewTerm(
        func=mdp.assist_hip_velocity_error_l2,
        weight=-0.05,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
        },
    )
    rear_mechanism_zero_position_error = RewTerm(
        func=mdp.rear_mechanism_zero_position_error_l2,
        weight=-1.0,
        params={
            "asset_cfg": REAR_MECHANISM_JOINT_CFG,
            # Normalize the slider (m) and cylinder (rad) by their joint limits.
            "slider_position_scale": 0.05,
            "cylinder_position_scale": 0.2,
        },
    )
    assist_torque_zero_at_hip_rest = RewTerm(
        func=mdp.FilteredHipDynamicsReward,
        weight=-2.0,
        params={
            "mode": "continuous_rest",
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "action_name": "assist_torque",
            "torque_limit": 10.0,
            "acceleration_filter_time_constant": 0.08,
            "rest_velocity_scale": 0.5,
            "rest_acceleration_scale": 3.0,
        },
    )
    assist_torque_hip_dynamics_tracking = RewTerm(
        func=mdp.FilteredHipDynamicsReward,
        weight=-8.0,
        params={
            "mode": "tracking",
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "action_name": "assist_torque",
            "torque_limit": 10.0,
            "lift_torque_limit": 10.0,
            "press_torque_limit": 4.0,
            "velocity_gain": 2.0,
            "acceleration_gain": 0.04,
            "dynamics_scale": 1.5,
            "acceleration_filter_time_constant": 0.08,
        },
    )
    hip_motor_torque_burden = None
    hip_motor_mechanical_power = None
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class AssistTerminationsCfg:
    """Original G1 fall conditions plus a 20-degree exoskeleton mismatch."""

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
            "threshold": math.radians(20.0),
        },
    )


@configclass
class G1AssistExoskeletonV2V2EnvCfg(G1AssistAmpEnvCfg):
    """Train v2-v2 with -10 Nm lift and +4 Nm press authority."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions = AssistActionsCfg()
        self.observations = AssistObservationsCfg()
        self.rewards = AssistRewardsCfg()
        self.terminations = AssistTerminationsCfg()

        self.events.reset_from_ref = None
        # Keep the mechanism PD gains deterministic and identical to Sim2Sim.
        # The inherited domain randomization continues to cover only the 29 gait
        # joints and the two learned hip-assist joints.
        self.events.scale_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=mdp.POLICY_JOINT_NAMES + mdp.ASSIST_JOINT_NAMES,
        )
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
        self.decimation = 10
        self.episode_length_s = 20.0
        # More standing episodes teach zero assist from exoskeleton histories alone.
        self.commands.base_velocity.rel_standing_envs = 0.35
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1AssistExoskeletonV2V2EnvCfg_PLAY(G1AssistExoskeletonV2V2EnvCfg):
    """Play configuration for the asymmetric v2-v2 assist policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

"""Reward-v1 task built as an independent copy of the v2 exoskeleton task."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from ..g1_assist_exoskeleton_v2.g1_assist_exoskeleton_env_cfg import (
    FROZEN_GAIT_POLICY_PATH,
)
from ..g1_assist_exoskeleton_v2.robot_cfg import (
    UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG,
)
from legged_lab.tasks.locomotion.amp.config.g1_assist.g1_assist_amp_env_cfg import (
    G1AssistAmpEnvCfg,
)

from . import mdp


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
CYLINDER_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=["pelvis_rear_cylinder_assist_joint"], preserve_order=True
)


@configclass
class AssistActionsCfg:
    assist_torque = mdp.FrozenGaitTransmittedAssistTorqueActionCfg(
        asset_name="robot",
        frozen_policy_path=FROZEN_GAIT_POLICY_PATH,
        assist_torque_limit=8.0,
        warmup_time_s=0.5,
    )


@configclass
class AssistPolicyObservationGroup(ObsGroup):
    """Deployable 0.5-s local history: q, qd, and previous action."""

    assist_joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": ASSIST_JOINT_CFG},
        noise=Gnoise(mean=0.0, std=math.radians(0.5)),
    )
    assist_joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": ASSIST_JOINT_CFG},
        noise=Gnoise(mean=0.0, std=math.radians(1.0)),
    )
    previous_action = ObsTerm(func=mdp.previous_normalized_assist_action)

    def __post_init__(self):
        self.history_length = 25
        self.flatten_history_dim = True
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class AssistCriticObservationGroup(ObsGroup):
    """Privileged 20-feature bilateral state over the same 0.5-s window."""

    privileged_state = ObsTerm(
        func=mdp.privileged_assist_state,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
            "cylinder_cfg": CYLINDER_JOINT_CFG,
        },
    )

    def __post_init__(self):
        self.history_length = 25
        self.flatten_history_dim = True
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class AssistObservationsCfg:
    policy: AssistPolicyObservationGroup = AssistPolicyObservationGroup()
    critic: AssistCriticObservationGroup = AssistCriticObservationGroup()


@configclass
class AssistRewardsCfg:
    """v1 rewards converted from 100-Hz direct-step scales to ManagerBased rates."""

    maximum_assistance = RewTerm(func=mdp.maximum_assistance, weight=20.0)
    primary_hip_burden = RewTerm(func=mdp.primary_hip_burden, weight=-5.0)
    hip_position_tracking = RewTerm(
        func=mdp.hip_position_tracking_error_l2,
        weight=-200.0,
        params={"hip_cfg": HIP_PITCH_JOINT_CFG},
    )
    hip_velocity_tracking = RewTerm(
        func=mdp.hip_velocity_tracking_error_l2,
        weight=-1.0,
        params={"hip_cfg": HIP_PITCH_JOINT_CFG},
    )
    physical_pair_soft_limit = RewTerm(
        func=mdp.physical_pair_soft_limit,
        weight=-5.0,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
            "cylinder_cfg": CYLINDER_JOINT_CFG,
            "soft_limit": math.radians(3.0),
            "penalty_width": math.radians(1.0),
        },
    )
    counter_torque = RewTerm(func=mdp.counter_torque, weight=-40.0)
    counter_torque_direction = RewTerm(func=mdp.counter_torque_direction, weight=-5.0)
    assist_effort = RewTerm(func=mdp.assist_effort_l2, weight=-1.0)
    assist_saturation = RewTerm(func=mdp.assist_saturation, weight=-5.0)
    # 50-Hz continuous-rate equivalents of the template's 100-Hz differences.
    assist_torque_delta_l2 = RewTerm(
        func=mdp.AssistTorqueTemporalPenalty,
        weight=-0.015,
        params={"mode": "delta_l2"},
    )
    assist_torque_delta_l1 = RewTerm(
        func=mdp.AssistTorqueTemporalPenalty,
        weight=-0.2,
        params={"mode": "delta_l1"},
    )
    assist_torque_jerk = RewTerm(
        func=mdp.AssistTorqueTemporalPenalty,
        weight=-0.0625,
        params={"mode": "jerk"},
    )
    requested_torque_rate = RewTerm(
        func=mdp.AssistTorqueTemporalPenalty,
        weight=-1.25,
        params={"mode": "requested_rate", "max_requested_delta": 0.8},
    )
    # Locomotion-specific safeguard retained from v2.
    assist_zero_when_standing = RewTerm(func=mdp.assist_zero_when_standing, weight=-1.0)
    # Event reward: -1000 * 0.02 s = -20 once per non-timeout termination.
    termination_penalty = RewTerm(func=mdp.gated_termination, weight=-1000.0)


@configclass
class AssistTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    )
    physical_pair_angle_mismatch = DoneTerm(
        func=mdp.physical_pair_angle_mismatch,
        params={
            "hip_cfg": HIP_PITCH_JOINT_CFG,
            "assist_cfg": ASSIST_JOINT_CFG,
            "cylinder_cfg": CYLINDER_JOINT_CFG,
            "threshold": math.radians(10.0),
        },
    )


@configclass
class G1AssistExoskeletonV2V1EnvCfg(G1AssistAmpEnvCfg):
    """Two-action bilateral task with v1 maximum-assistance rewards."""

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
        self.events.scale_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=mdp.POLICY_JOINT_NAMES + mdp.ASSIST_JOINT_NAMES
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_and_sync_physical_pair,
            mode="reset",
            params={
                "position_range": (0.8, 1.2),
                "velocity_range": (0.0, 0.0),
                "hip_cfg": HIP_PITCH_JOINT_CFG,
                "assist_cfg": ASSIST_JOINT_CFG,
                "cylinder_cfg": CYLINDER_JOINT_CFG,
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
class G1AssistExoskeletonV2V1EnvCfg_PLAY(G1AssistExoskeletonV2V1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

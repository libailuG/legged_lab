"""G1 v3 robot configuration with assist-v0 gains and an exoskeleton USD."""

from copy import deepcopy

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.config.g1_assist.robot_cfg import (
    UNITREE_G1_29DOF_ASSIST_CFG,
)

from .pid_actuator import IdealPIDActuatorCfg


G1_29DOF_ASSIST_EXOSKELETON_V2_USD_PATH = (
    f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof_assist/usd/"
    "g1_29dof_assist_exoskeleton_v2/g1_29dof_assist_exoskeleton_v2.usd"
)
INTEGRAL_EFFORT_LIMIT_NM = 10.0
G1_BODY_JOINT_NAMES = list(UNITREE_G1_29DOF_ASSIST_CFG.joint_sdk_names)


UNITREE_G1_29DOF_PID_V3_CFG = deepcopy(UNITREE_G1_29DOF_ASSIST_CFG)
UNITREE_G1_29DOF_PID_V3_CFG.spawn.usd_path = G1_29DOF_ASSIST_EXOSKELETON_V2_USD_PATH
UNITREE_G1_29DOF_PID_V3_CFG.spawn.articulation_props.enabled_self_collisions = True
UNITREE_G1_29DOF_PID_V3_CFG.actuators = {
    "assist_hip_pitch_yaw_waist_yaw": IdealPIDActuatorCfg(
        joint_names_expr=[
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_yaw_joint",
        ],
        effort_limit=200.0,
        effort_limit_sim=200.0,
        velocity_limit=32.0,
        velocity_limit_sim=32.0,
        stiffness={".*_hip_.*": 180.0, "waist_yaw_joint": 360.0},
        damping={".*_hip_.*": 3.0, "waist_yaw_joint": 7.5},
        integral_gain={".*_hip_.*": 18.0, "waist_yaw_joint": 36.0},
        integral_effort_limit=INTEGRAL_EFFORT_LIMIT_NM,
        integration_dt=0.001,
        armature=0.01,
    ),
    "assist_hip_roll_knee": IdealPIDActuatorCfg(
        joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
        effort_limit=316.0,
        effort_limit_sim=316.0,
        velocity_limit=20.0,
        velocity_limit_sim=20.0,
        stiffness={".*_hip_roll_.*": 180.0, ".*_knee_.*": 270.0},
        damping={".*_hip_roll_.*": 3.0, ".*_knee_.*": 6.0},
        integral_gain={".*_hip_roll_.*": 18.0, ".*_knee_.*": 27.0},
        integral_effort_limit=INTEGRAL_EFFORT_LIMIT_NM,
        integration_dt=0.001,
        armature=0.01,
    ),
    "assist_ankle_waist": IdealPIDActuatorCfg(
        joint_names_expr=[".*_ankle_.*", "waist_roll_joint", "waist_pitch_joint"],
        effort_limit=57.0,
        effort_limit_sim=57.0,
        velocity_limit=37.0,
        velocity_limit_sim=37.0,
        stiffness=72.0,
        damping={".*_ankle_.*": 3.0, "waist_.*_joint": 7.5},
        integral_gain=7.2,
        integral_effort_limit=INTEGRAL_EFFORT_LIMIT_NM,
        integration_dt=0.001,
        armature=0.01,
    ),
    "assist_shoulder_elbow_wrist_roll": IdealPIDActuatorCfg(
        joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll.*"],
        effort_limit=38.0,
        effort_limit_sim=38.0,
        velocity_limit=37.0,
        velocity_limit_sim=37.0,
        stiffness=56.0,
        damping=1.3,
        integral_gain=5.6,
        integral_effort_limit=INTEGRAL_EFFORT_LIMIT_NM,
        integration_dt=0.001,
        armature=0.01,
    ),
    "assist_wrist_pitch_yaw": IdealPIDActuatorCfg(
        joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
        effort_limit=8.0,
        effort_limit_sim=8.0,
        velocity_limit=22.0,
        velocity_limit_sim=22.0,
        stiffness=56.0,
        damping=1.3,
        integral_gain=5.6,
        integral_effort_limit=INTEGRAL_EFFORT_LIMIT_NM,
        integration_dt=0.001,
        armature=0.01,
    ),
    "exoskeleton_free": IdealPIDActuatorCfg(
        joint_names_expr=[
            "left_hip_pitch_assist_joint",
            "right_hip_pitch_assist_joint",
        ],
        effort_limit=8.0,
        effort_limit_sim=8.0,
        velocity_limit=100.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
        integral_gain=0.0,
        integral_effort_limit=0.0,
        integration_dt=0.001,
        armature=0.0,
    ),
    "rear_mechanism_pd": IdealPIDActuatorCfg(
        joint_names_expr=[
            "pelvis_rear_upper_box_assist_joint",
            "pelvis_rear_cylinder_assist_joint",
        ],
        effort_limit={
            "pelvis_rear_upper_box_assist_joint": 300.0,
            "pelvis_rear_cylinder_assist_joint": 200.0,
        },
        effort_limit_sim={
            "pelvis_rear_upper_box_assist_joint": 300.0,
            "pelvis_rear_cylinder_assist_joint": 200.0,
        },
        velocity_limit={
            "pelvis_rear_upper_box_assist_joint": 30.0,
            "pelvis_rear_cylinder_assist_joint": 10.0,
        },
        velocity_limit_sim={
            "pelvis_rear_upper_box_assist_joint": 30.0,
            "pelvis_rear_cylinder_assist_joint": 10.0,
        },
        stiffness=10.0,
        damping=1.0,
        integral_gain=0.0,
        integral_effort_limit=0.0,
        integration_dt=0.001,
        armature=0.0,
    ),
}

UNITREE_G1_29DOF_PID_V3_CFG.init_state.joint_pos.update(
    {
        "left_hip_pitch_assist_joint": -0.1,
        "right_hip_pitch_assist_joint": -0.1,
        "pelvis_rear_upper_box_assist_joint": 0.0,
        "pelvis_rear_cylinder_assist_joint": 0.0,
    }
)

"""Robot configuration for the v2 exoskeleton and its passive PD mechanism."""

from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.config.g1_assist.robot_cfg import (
    UNITREE_G1_29DOF_ASSIST_CFG,
)


G1_29DOF_ASSIST_EXOSKELETON_V2_USD_PATH = (
    f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof_assist/usd/"
    "g1_29dof_assist_exoskeleton_v2/g1_29dof_assist_exoskeleton_v2.usd"
)

UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG = deepcopy(UNITREE_G1_29DOF_ASSIST_CFG)
UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.spawn.usd_path = (
    G1_29DOF_ASSIST_EXOSKELETON_V2_USD_PATH
)
UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.spawn.articulation_props.enabled_self_collisions = True

UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.actuators[
    "assist_hip_pitch_yaw_waist_yaw"
].joint_names_expr = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_yaw_joint",
]

UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.actuators[
    "exoskeleton_torque"
] = ImplicitActuatorCfg(
    joint_names_expr=["left_hip_pitch_assist_joint", "right_hip_pitch_assist_joint"],
    effort_limit_sim=8.0,
    velocity_limit_sim=100.0,
    stiffness=0.0,
    damping=0.0,
    armature=0.0,
)

UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.actuators[
    "box_assist_position"
] = ImplicitActuatorCfg(
    joint_names_expr=["pelvis_rear_upper_box_assist_joint"],
    effort_limit_sim=300.0,
    velocity_limit_sim=30.0,
    stiffness=10000.0,
    damping=20.0,
    armature=0.0,
)

UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.actuators[
    "cylinder_assist_position"
] = ImplicitActuatorCfg(
    joint_names_expr=["pelvis_rear_cylinder_assist_joint"],
    effort_limit_sim=200.0,
    velocity_limit_sim=10.0,
    stiffness=4.0,
    damping=2.0,
    armature=0.0,
)

UNITREE_G1_29DOF_ASSIST_EXOSKELETON_V2_CFG.init_state.joint_pos.update(
    {
        "left_hip_pitch_assist_joint": -0.1,
        "right_hip_pitch_assist_joint": -0.1,
        "pelvis_rear_upper_box_assist_joint": 0.0,
        "pelvis_rear_cylinder_assist_joint": 0.0,
    }
)

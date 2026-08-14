"""Independent robot configuration for torque-controlled assist joints."""

from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.tasks.locomotion.amp.config.g1_assist.robot_cfg import (
    UNITREE_G1_29DOF_ASSIST_CFG,
)


G1_29DOF_ASSIST_EXOSKELETON_USD_PATH = (
    f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof_assist/usd/"
    "g1_29dof_assist_exoskeleton/g1_29dof_assist_exoskeleton.usd"
)

UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG = deepcopy(UNITREE_G1_29DOF_ASSIST_CFG)
UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG.spawn.usd_path = G1_29DOF_ASSIST_EXOSKELETON_USD_PATH
UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG.spawn.articulation_props.enabled_self_collisions = True

# Prevent the existing hip-pitch wildcard from also claiming the two assist joints.
UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG.actuators[
    "assist_hip_pitch_yaw_waist_yaw"
].joint_names_expr = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_yaw_joint",
]

# Pure feed-forward torque control for the exoskeleton. The PPO action term
# supplies effort targets in [-8, 8] N.m; there is no position/velocity drive.
UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG.actuators["exoskeleton_torque"] = ImplicitActuatorCfg(
    joint_names_expr=["left_hip_pitch_assist_joint", "right_hip_pitch_assist_joint"],
    effort_limit_sim=8.0,
    velocity_limit_sim=100.0,
    stiffness=0.0,
    damping=0.0,
    armature=0.0,
)

# Start the assist links aligned with the corresponding hip-pitch links.
UNITREE_G1_29DOF_ASSIST_EXOSKELETON_CFG.init_state.joint_pos.update(
    {
        "left_hip_pitch_assist_joint": -0.1,
        "right_hip_pitch_assist_joint": -0.1,
    }
)

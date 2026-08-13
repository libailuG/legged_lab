"""Robot asset configuration for the mass-adjusted G1 assist model."""

from copy import deepcopy

from isaaclab.actuators import ImplicitActuatorCfg

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG


G1_29DOF_ASSIST_USD_PATH = (
    f"{LEGGED_LAB_ROOT_DIR}/data/Robots/Unitree/g1_29dof_assist/"
    "usd/g1_29dof_assist/g1_29dof_assist.usd"
)

# Keep this asset independent from the existing G1 task configuration.
UNITREE_G1_29DOF_ASSIST_CFG = deepcopy(UNITREE_G1_29DOF_CFG)
UNITREE_G1_29DOF_ASSIST_CFG.spawn.usd_path = G1_29DOF_ASSIST_USD_PATH

# The assist robot is about 2.27 times heavier than the original G1. Increase
# lower-body/waist authority while keeping the upper body comparatively soft.
UNITREE_G1_29DOF_ASSIST_CFG.actuators = {
    "assist_hip_pitch_yaw_waist_yaw": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
        effort_limit_sim=200.0,
        velocity_limit_sim=32.0,
        stiffness={
            ".*_hip_.*": 180.0,
            "waist_yaw_joint": 360.0,
        },
        damping={
            ".*_hip_.*": 3.0,
            "waist_yaw_joint": 7.5,
        },
        armature=0.01,
    ),
    "assist_hip_roll_knee": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
        effort_limit_sim=316.0,
        velocity_limit_sim=20.0,
        stiffness={
            ".*_hip_roll_.*": 180.0,
            ".*_knee_.*": 270.0,
        },
        damping={
            ".*_hip_roll_.*": 3.0,
            ".*_knee_.*": 6.0,
        },
        armature=0.01,
    ),
    "assist_ankle_waist": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_.*", "waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=57.0,
        velocity_limit_sim=37.0,
        stiffness=72.0,
        damping={
            ".*_ankle_.*": 3.0,
            "waist_.*_joint": 7.5,
        },
        armature=0.01,
    ),
    "assist_shoulder_elbow_wrist_roll": ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_roll.*"],
        effort_limit_sim=38.0,
        velocity_limit_sim=37.0,
        stiffness=56.0,
        damping=1.3,
        armature=0.01,
    ),
    "assist_wrist_pitch_yaw": ImplicitActuatorCfg(
        joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
        effort_limit_sim=8.0,
        velocity_limit_sim=22.0,
        stiffness=56.0,
        damping=1.3,
        armature=0.01,
    ),
}

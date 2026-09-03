"""G1 robot configuration copied from v0 with explicit PID actuators."""

from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG

from .pid_actuator import IdealPIDActuatorCfg


UNITREE_G1_29DOF_PID_CFG = UNITREE_G1_29DOF_CFG.replace(
    actuators={
        "N7520-14.3": IdealPIDActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit=88,
            velocity_limit=32.0,
            velocity_limit_sim=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            integral_gain={".*_hip_.*": 10.0, "waist_yaw_joint": 20.0},
            integral_error_limit=0.5,
            integration_dt=0.001,
            armature=0.01,
        ),
        "N7520-22.5": IdealPIDActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit=139,
            velocity_limit=20.0,
            velocity_limit_sim=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
            integral_gain={".*_hip_roll_.*": 10.0, ".*_knee_.*": 15.0},
            integral_error_limit=0.5,
            integration_dt=0.001,
            armature=0.01,
        ),
        "N5020-16": IdealPIDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit=25,
            velocity_limit=37,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            integral_gain=4.0,
            integral_error_limit=0.5,
            integration_dt=0.001,
            armature=0.01,
        ),
        "W4010-25": IdealPIDActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit=5,
            velocity_limit=22,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=1.0,
            integral_gain=4.0,
            integral_error_limit=0.5,
            integration_dt=0.001,
            armature=0.01,
        ),
    }
)

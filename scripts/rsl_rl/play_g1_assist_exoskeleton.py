# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play the 29-DoF G1 assist policy on the 31-DoF exoskeleton asset.

The trained policy still observes and controls the original 29 robot joints. The
two exoskeleton joints are passive: they are excluded from the action/observation
spaces and assigned an actuator with zero stiffness, damping, and effort limit.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


DEFAULT_TASK = "LeggedLab-Isaac-AMP-G1-assist-Play-v0"
DEFAULT_CHECKPOINT = (
    "/home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_amp/"
    "2026-08-13_13-45-16/model_29400.pt"
)

parser = argparse.ArgumentParser(
    description="Play model_29400.pt on the G1 assist exoskeleton USD with passive assist joints."
)
parser.add_argument("--video", action="store_true", default=False, help="Record a video during play.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in simulation steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of Fabric.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Registered base play task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent configuration registry entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Try to run in real time.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many policy steps; by default run until the simulator closes.",
)
parser.add_argument(
    "--torque_report_interval",
    type=int,
    default=200,
    help="Print the maximum absolute assist-joint actuator torque every N steps; 0 disables reports.",
)
cli_args.add_rsl_rl_args(parser)
parser.set_defaults(checkpoint=DEFAULT_CHECKPOINT)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""The remaining imports require Isaac Sim to be running."""

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import legged_lab.tasks  # noqa: F401
from legged_lab import LEGGED_LAB_ROOT_DIR


EXOSKELETON_USD_PATH = os.path.join(
    LEGGED_LAB_ROOT_DIR,
    "data",
    "Robots",
    "Unitree",
    "g1_29dof_assist",
    "usd",
    "g1_29dof_assist_exoskeleton",
    "g1_29dof_assist_exoskeleton.usd",
)

# This is the articulation/action order used when model_29400.pt was trained.
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
ASSIST_JOINT_NAMES = ["left_hip_pitch_assist_joint", "right_hip_pitch_assist_joint"]


def configure_exoskeleton(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Replace the asset and retain the checkpoint's original 29-DoF interface."""
    if not os.path.isfile(EXOSKELETON_USD_PATH):
        raise FileNotFoundError(f"Exoskeleton USD was not found: {EXOSKELETON_USD_PATH}")

    env_cfg.scene.robot.spawn.usd_path = EXOSKELETON_USD_PATH
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True

    # The original wildcard also matches '*_hip_pitch_assist_joint'. Replace it
    # with exact controlled-joint names, then give the two new joints a passive
    # zero-effort actuator so their commanded/applied actuator torque is zero.
    hip_actuator = env_cfg.scene.robot.actuators["assist_hip_pitch_yaw_waist_yaw"]
    hip_actuator.joint_names_expr = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_yaw_joint",
    ]
    env_cfg.scene.robot.actuators["passive_exoskeleton_joints"] = ImplicitActuatorCfg(
        joint_names_expr=ASSIST_JOINT_NAMES,
        effort_limit_sim=0.0,
        velocity_limit_sim=100.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
    )
    for joint_name in ASSIST_JOINT_NAMES:
        env_cfg.scene.robot.init_state.joint_pos[joint_name] = 0.0

    # Keep the action vector at 29 dimensions and preserve the exact ordering
    # used by the checkpoint.
    env_cfg.actions.joint_pos.joint_names = POLICY_JOINT_NAMES
    env_cfg.actions.joint_pos.preserve_order = True

    # The actor, critic and AMP discriminator must retain their checkpoint-time
    # dimensions. Reference/demo observations already contain 29 motion joints.
    for group_name in ("policy", "critic", "disc"):
        obs_group = getattr(env_cfg.observations, group_name)
        for term_name in ("joint_pos", "joint_vel"):
            obs_term = getattr(obs_group, term_name)
            obs_term.params = {
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True
                )
            }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Run the trained assist policy with the passive exoskeleton asset."""
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise TypeError("This script requires a ManagerBasedRLEnvCfg task.")
    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint was not found: {args_cli.checkpoint}")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    configure_exoskeleton(env_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Exoskeleton USD: {EXOSKELETON_USD_PATH}")
    print(f"[INFO] Checkpoint: {resume_path}")
    print("[INFO] Policy interface: 29 controlled joints; 2 passive zero-torque assist joints")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_exoskeleton"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    robot = env.unwrapped.scene["robot"]
    assist_joint_ids, assist_joint_names = robot.find_joints(ASSIST_JOINT_NAMES, preserve_order=True)
    if assist_joint_names != ASSIST_JOINT_NAMES:
        raise RuntimeError(
            f"Unexpected assist joint resolution: expected {ASSIST_JOINT_NAMES}, got {assist_joint_names}"
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPRunner":
        from rsl_rl.runners import AMPRunner

        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading model checkpoint: {resume_path}")
    runner.load(resume_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            if actions.shape[-1] != len(POLICY_JOINT_NAMES):
                raise RuntimeError(f"Expected 29 policy actions, received shape {tuple(actions.shape)}")
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        timestep += 1
        if args_cli.torque_report_interval > 0 and (
            timestep == 1 or timestep % args_cli.torque_report_interval == 0
        ):
            max_assist_torque = robot.data.applied_torque[:, assist_joint_ids].abs().max().item()
            print(f"[INFO] Step {timestep}: max |assist actuator torque| = {max_assist_torque:.3e} N.m")
            if max_assist_torque > 1.0e-6:
                raise RuntimeError("An assist joint produced non-zero actuator torque.")

        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break
        if args_cli.video and timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

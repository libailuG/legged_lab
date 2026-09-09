"""Paired no-push walk/stop and continuous-walk checkpoint evaluation."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--check-only', action='store_true', help='20-step script validation, not a benchmark')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
import legged_lab.tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import AMPRunner
from legged_lab.tasks.locomotion.amp.config.g1_assist_v1.standing import (
    enable_standing_training, standing_stability, standing_joint_motion,
)

task = 'LeggedLab-Isaac-AMP-G1-assist-v1'
cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
cfg.episode_length_s = 30.0
cfg.events.push_robot = None
cfg.events.reset_from_ref = None
cfg.commands.base_velocity.heading_command = False
cfg.commands.base_velocity.rel_standing_envs = 0.0
cfg.commands.base_velocity.resampling_time_range = (1000., 1000.)
cfg.commands.base_velocity.ranges.lin_vel_x = (.7, .7)
cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
cfg.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
enable_standing_training(cfg)
cfg.commands.base_velocity.rel_standing_envs = 0.0
agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
agent.device = args.device
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
runner = AMPRunner(env, agent.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint, load_optimizer=False, map_location=args.device)
policy = runner.get_inference_policy(device=args.device)
raw = env.unwrapped
assert abs(raw.step_dt-.02)<1e-8 and abs(cfg.sim.dt-.001)<1e-8
command = raw.command_manager.get_term('base_velocity')
results = {}
with torch.inference_mode():
    for scenario in ('walk_then_stop', 'continuous_walk'):
        env.seed(args.seed)
        env.reset()
        fallen = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        samples = {k: [] for k in ('walking', 'post_stop', 'late_stop')}
        for step in range(20 if args.check_only else 1250):
            stopping = scenario == 'walk_then_stop' and step >= (10 if args.check_only else 250)
            command.vel_command_b[:] = 0.
            command.vel_command_b[:, 0] = 0. if stopping else .7
            command.is_standing_env[:] = stopping
            obs = env.get_observations()
            if not stopping:
                assert torch.count_nonzero(standing_stability(raw)) == 0
                assert torch.count_nonzero(standing_joint_motion(raw)) == 0
            _, _, dones, _ = env.step(policy(obs))
            fallen |= dones.bool()
            d = raw.scene['robot'].data
            valid = ~fallen
            values = torch.stack((
                torch.linalg.vector_norm(d.root_lin_vel_w[:, :2], dim=1),
                torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1., 1.))*180./torch.pi,
                d.joint_vel.square().mean(dim=1).sqrt(),
                (d.root_lin_vel_b[:, 0] - (0. if stopping else .7)).abs(),
            ), dim=1)
            assert torch.isfinite(values).all(), 'Nonfinite evaluation state'
            phase = 'post_stop' if stopping else 'walking'
            # Exclude reset trajectories; expose survivor sample counts and fall rate.
            if valid.any():
                row = torch.cat((values[valid].sum(dim=0), valid.sum().reshape(1))).cpu()
                samples[phase].append(row)
                if stopping and step >= 1000:
                    samples['late_stop'].append(row)
        result = {'fall_rate': fallen.float().mean().item(), 'fallen_envs': fallen.sum().item()}
        for phase, rows in samples.items():
            if rows:
                sums = torch.stack(rows).sum(dim=0)
                result[phase] = dict(zip(
                    ('horizontal_speed_m_s', 'tilt_deg', 'joint_velocity_rms_rad_s', 'vx_abs_error_m_s'),
                    (sums[:4]/sums[4]).tolist()))
                result[phase]['survivor_samples'] = int(sums[4])
        results[scenario] = result
report = {'checkpoint': args.checkpoint, 'iteration': runner.current_learning_iteration,
          'check_only': args.check_only,
          'num_envs': args.num_envs, 'seed': args.seed, 'pushes': False,
          'reset_from_reference': False, 'policy_dt': raw.step_dt,
          'protocol': '0.7m/s 5s then zero 20s; control: 0.7m/s 25s; late_stop: final 5s',
          'metrics_population': 'not-yet-fallen environments; fall rate counts any termination',
          'results': results}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
env.close()
app.close()

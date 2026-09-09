"""Check MuJoCo-side controller/history against training-side tensor semantics."""

import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sim = load_module("v3_sim2sim", ROOT / "scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v3.py")
train = load_module("v3_paired", ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/"
                    "g1_assist_exoskeleton_v2_v3/mdp/paired_assist.py")


class Sim2SimTests(unittest.TestCase):
    def test_gait_holds_targets_and_resets_each_environment_phase(self):
        path = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist_exoskeleton_v2_v3/mdp/actions.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "FrozenGaitAssistTorqueAction")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("process_actions", "reset")]
        for method in methods:
            method.returns = None
            for arg in method.args.args:
                arg.annotation = None
        import math
        scope = dict(torch=torch, math=math, shared_motion_gate=train.shared_motion_gate,
                     paired_torques=train.paired_torques, slew_paired_scalar=train.slew_paired_scalar)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
        cfg = SimpleNamespace(motion_filter_time_constant=.05, motion_speed_deadzone=.15,
                              motion_speed_full=.8, torque_limit=10., torque_rate_limit=80.,
                              command_name="base_velocity", gait_action_scale=.25, gait_observation_noise=False)
        data = SimpleNamespace(root_ang_vel_b=torch.zeros(2, 3), projected_gravity_b=torch.zeros(2, 3),
                               joint_pos=torch.zeros(2, 31), joint_vel=torch.ones(2, 31),
                               default_joint_pos=torch.zeros(2, 31), default_joint_vel=torch.zeros(2, 31))
        obj = SimpleNamespace(cfg=cfg, _asset=SimpleNamespace(data=data), _policy_joint_ids=list(range(29)),
                              _assist_joint_ids=[29,30], _gait_interval=2, _gait_tick=torch.zeros(2, dtype=torch.long))
        obj._env = SimpleNamespace(step_dt=.01, command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(2,3)))
        for name in ("_raw_actions", "_motion_gate"):
            setattr(obj, name, torch.zeros(2,1))
        for name in ("_processed_actions", "_previous_processed_actions", "_requested_torque_delta",
                     "_filtered_assist_joint_velocity", "_extra_position_targets"):
            setattr(obj, name, torch.zeros(2,2))
        for name in ("_gait_actions", "_previous_gait_actions", "_gait_position_targets"):
            setattr(obj, name, torch.zeros(2,29))
        calls = []
        def policy(obs):
            calls.append(obs.clone())
            return torch.full((len(obs), 29), float(len(calls)))
        obj._frozen_policy = policy
        step = lambda: scope["process_actions"](obj, torch.ones(2,1))
        step()
        first = obj._gait_position_targets.clone()
        torque = obj._processed_actions.clone()
        step()
        torch.testing.assert_close(obj._gait_position_targets, first)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.any(obj._processed_actions != torque))
        step()
        scope["reset"](obj, [0])
        step()
        self.assertEqual(len(calls[-1]), 1)
        torch.testing.assert_close(calls[-1][0,-29:], torch.zeros(29))
        torch.testing.assert_close(obj._gait_position_targets[1], torch.full((29,), .5))
        step()
        self.assertEqual(len(calls[-1]), 1)
        torch.testing.assert_close(obj._gait_position_targets[0], torch.full((29,), .75))

    def test_pid_matches_v1_compute_and_reset(self):
        # Execute the actual training compute/reset methods without launching Isaac.
        path = ROOT / "source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist_v1/pid_actuator.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "IdealPIDActuator")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("compute", "reset")]
        for method in methods:
            method.returns = None
            for arg in method.args.args:
                arg.annotation = None
        scope = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
        g = dict(kp=np.array([180., 56.]), kd=np.array([3., 1.3]),
                 ki=np.array([18., 5.6]), integral_limits=np.array([10., 10.]),
                 pid_limits=np.array([200., 8.]))
        controller = sim.JointPIDController(g, .001)
        ref = SimpleNamespace(stiffness=torch.from_numpy(g["kp"]), damping=torch.from_numpy(g["kd"]),
                              integral_gain=torch.from_numpy(g["ki"]),
                              integral_effort_limit=torch.from_numpy(g["integral_limits"]),
                              _integral_effort=torch.zeros(2, dtype=torch.float64),
                              cfg=SimpleNamespace(integration_dt=.001))
        limits = torch.from_numpy(g["pid_limits"])
        ref._clip_effort = lambda effort: torch.clamp(effort, -limits, limits)
        rng = np.random.default_rng(42)
        for step in range(2000):
            target = np.full(2, .02 if step < 500 else 3. if step < 1000 else -3.)
            velocity = rng.uniform(-2., 2., 2)
            action = SimpleNamespace(joint_positions=torch.from_numpy(target), joint_velocities=torch.zeros(2),
                                     joint_efforts=torch.zeros(2))
            result = scope["compute"](ref, action, torch.zeros(2), torch.from_numpy(velocity))
            actual = controller.step(target, np.zeros(2), velocity)
            np.testing.assert_allclose(actual, result.joint_efforts.numpy(), atol=1e-12)
            np.testing.assert_allclose(controller.integral, ref._integral_effort.numpy(), atol=1e-12)
        controller.reset()
        scope["reset"](ref, [0, 1])
        np.testing.assert_array_equal(controller.integral, ref._integral_effort.numpy())

    def test_numpy_controller_matches_training_over_reversals_and_stops(self):
        cfg = dict(motion_filter_time_constant=.05, motion_speed_deadzone=.15,
                   motion_speed_full=.8, torque_limit=10., torque_rate_limit=80.)
        controller = sim.PairedAssistController(cfg, .01)
        filtered = torch.zeros(1, 2, dtype=torch.float64)
        previous = torch.zeros(1, 1, dtype=torch.float64)
        rng = np.random.default_rng(7)
        for step in range(1000):
            action = (-2. if step % 160 < 80 else 2.) if step < 800 else 0.
            velocity = rng.uniform(-2., 2., 2) if step < 800 else np.zeros(2)
            filtered.lerp_(torch.from_numpy(velocity).reshape(1, 2), 1-np.exp(-.01/.05))
            gate = train.shared_motion_gate(filtered, .15, .8)
            target = float(np.clip(action, -1., 1.)) * 10. * gate
            previous = train.slew_paired_scalar(target, previous, 10., 80., .01)
            expected = train.paired_torques(previous).numpy()[0]
            actual = controller.step(action, velocity)
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(actual, [0., 0.])
        controller.reset()
        self.assertEqual(controller.scalar, 0.)
        np.testing.assert_array_equal(controller.filtered_velocity, [0., 0.])

    def test_history_padding_and_term_major_order(self):
        history = sim.AssistHistory()
        history.reset([1., 2.], [3., 4.])
        obs = history.observation()
        self.assertEqual(obs.shape, (150,))
        np.testing.assert_array_equal(obs[:50].reshape(25, 2), np.tile([1.,2.], (25,1)))
        np.testing.assert_array_equal(obs[50:100].reshape(25, 2), np.tile([3.,4.], (25,1)))
        np.testing.assert_array_equal(obs[100:], np.zeros(50))
        for n in range(30):
            history.append([n,n+1], [n+100,n+101], [n+200,-n-200])
        obs = history.observation().reshape(3,25,2)
        np.testing.assert_array_equal(obs[0,:,0], np.arange(5,30))
        np.testing.assert_array_equal(obs[1,:,0], np.arange(105,130))
        np.testing.assert_array_equal(obs[2,:,0], np.arange(205,230))
        np.testing.assert_array_equal(obs[2,:,1], -np.arange(205,230))

    def test_nonfinite_action_is_rejected(self):
        controller = sim.PairedAssistController({}, .01)
        with self.assertRaises(ValueError):
            controller.step(float('nan'), np.zeros(2))

    def test_regex_parameter_resolution(self):
        names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint']
        actual = sim.resolve_parameter({'.*_hip_.*':180., 'waist_yaw_joint':360.}, names)
        np.testing.assert_array_equal(actual, [180.,180.,360.])
        with self.assertRaises(ValueError):
            sim.resolve_parameter({'.*':1., 'left_.*':2.}, names)


if __name__ == '__main__':
    unittest.main()

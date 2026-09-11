"""Offline network, temporal controller and C99 equivalence checks."""
import ctypes as ct
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch

from controller import AssistController

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('sim', ROOT / 'scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v7.py')
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)

Pair = ct.c_float * 2
class History(ct.Structure):
    _fields_ = [('p', Pair*25), ('v', Pair*25), ('t', Pair*25), ('pulse', (ct.c_float*12)*25), ('index',ct.c_uint)]
class Workspace(ct.Structure):
    _fields_ = [('h',ct.c_float*256), ('s',ct.c_float*450)]
class CController(ct.Structure):
    _fields_ = [('history',History), ('workspace',Workspace), ('obs',ct.c_float*450),
                ('previous',Pair), ('filtered',Pair), ('pulse',ct.c_float*10), ('ready',ct.c_uint), ('first',ct.c_uint)]


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        lib = Path(cls.temp.name) / 'policy.so'
        subprocess.run(['gcc','-std=c99','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                        str(HERE/'c/g1_assist_v7_policy.c'),'-lm','-o',str(lib)], check=True)
        cls.c = ct.CDLL(str(lib))
        cls.c.g1_assist_v7_reset.argtypes = [ct.POINTER(CController), Pair, Pair]
        cls.c.g1_assist_v7_step.argtypes = [ct.POINTER(CController), Pair, Pair, ct.c_int, Pair]
        cls.c.g1_assist_v7_policy_forward.argtypes = [ct.POINTER(ct.c_float),ct.POINTER(ct.c_float),ct.POINTER(Workspace)]
        cls.jit = torch.jit.load(str(HERE/'weights/assist_policy.pt')).eval()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_network_numpy_c_torch(self):
        policy = AssistController().policy
        rng = np.random.default_rng(123)
        workspace, out = Workspace(), (ct.c_float*2)()
        peak = 0.
        for _ in range(200):
            obs = rng.normal(0, 2, 450).astype(np.float32)
            with torch.inference_mode():
                expected = self.jit(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
            actual = policy(obs)
            self.c.g1_assist_v7_policy_forward(obs.ctypes.data_as(ct.POINTER(ct.c_float)),out,ct.byref(workspace))
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(list(out), expected, atol=2e-5, rtol=2e-5)
            peak = max(peak, float(np.max(np.abs(expected-np.array(out)))))
        print('C/Torch raw-action peak error:',peak)

    def test_control_sequence_and_history(self):
        deployment = AssistController()
        reference = sim.PairedAssistController(deployment.cfg,.01)
        history = sim.AssistHistory()
        p,v = np.zeros(2,dtype=np.float32),np.zeros(2,dtype=np.float32)
        deployment.reset(p,v)
        history.reset(p,v)
        c,out = CController(),Pair()
        self.assertEqual(self.c.g1_assist_v7_reset(ct.byref(c),Pair(*p),Pair(*v)),0)
        last = np.zeros(2)
        previous_reference = np.zeros(2)
        peak = 0.
        for n in range(1500):
            t=n*.01
            if 500<=n<800:
                p,v=np.zeros(2,dtype=np.float32),np.zeros(2,dtype=np.float32)
            else:
                p=np.array([.4*np.sin(4*t),-.4*np.sin(4*t)],dtype=np.float32)
                v=np.array([1.6*np.cos(4*t),-1.6*np.cos(4*t)],dtype=np.float32)
            if n==0: history.reset(p,v)
            else: history.append(p,v,previous_reference,reference.observation())
            with torch.inference_mode():
                raw=self.jit(torch.from_numpy(history.observation()).unsqueeze(0)).numpy()[0]
            expected=reference.step(raw,v)
            actual=deployment.step(p,v,timestamp=t)
            self.assertEqual(self.c.g1_assist_v7_step(ct.byref(c),Pair(*p),Pair(*v),1,out),0)
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
            np.testing.assert_allclose(list(out),actual,atol=1e-4,rtol=1e-4)
            self.assertEqual(float(actual.sum()),0.)
            self.assertLessEqual(float(np.abs(actual).max()),10.)
            self.assertLessEqual(float(np.abs(actual-last).max()),.80001)
            np.testing.assert_allclose(deployment.history.observation(),history.observation(),atol=1e-4)
            peak=max(peak,float(np.max(np.abs(np.array(out)-actual))))
            last=actual;previous_reference=expected
        print('C/NumPy closed-loop torque peak error:',peak)

    def test_numpy_gait_matches_torch(self):
        from numpy_gait_policy import NumpyGaitPolicy
        policy=NumpyGaitPolicy(HERE/'weights')
        jit=torch.jit.load(str(HERE/'weights/gait_policy.pt')).eval()
        obs=np.random.default_rng(11).normal(0,1,(256,96)).astype(np.float32)
        with torch.inference_mode(): expected=jit(torch.from_numpy(obs)).numpy()
        actual=policy(obs)
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        print('NumPy/Torch gait peak error:',float(np.max(np.abs(actual-expected))))

    def test_optional_gait_pid_matches_sim2sim(self):
        import mujoco as mj
        from gait_controller import GaitPIDController, DEFAULT_JOINT_POS, KP, KD, EFFORT_LIMIT
        cfg=sim.load_training_config(HERE/'weights/env.yaml')
        model=mj.MjModel.from_xml_path(str(sim.DEFAULT_MODEL))
        groups=sim.configure_model(mj,model,cfg)
        gait=groups['gait']
        np.testing.assert_allclose(gait['default'],DEFAULT_JOINT_POS)
        np.testing.assert_allclose(gait['kp'],KP)
        np.testing.assert_allclose(gait['kd'],KD)
        np.testing.assert_allclose(gait['ki'],.1*KP)
        np.testing.assert_allclose(gait['pid_limits'],EFFORT_LIMIT)
        ref=sim.JointPIDController(gait,.001)
        obj=GaitPIDController();obj.reset()
        previous=np.zeros(29,dtype=np.float32)
        rng=np.random.default_rng(9)
        for n in range(100):
            q=(DEFAULT_JOINT_POS+rng.uniform(-.01,.01,29)).astype(np.float32)
            v=rng.uniform(-.1,.1,29).astype(np.float32)
            gyro=np.zeros(3,dtype=np.float32);gravity=np.array([0,0,-1],dtype=np.float32)
            command=np.array([.7,0,0],dtype=np.float32)
            if n%20==0:
                obs=np.concatenate((gyro,gravity,command,q-DEFAULT_JOINT_POS,v,previous)).astype(np.float32)
                with torch.inference_mode(): previous=obj.policy(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
            target=DEFAULT_JOINT_POS+.25*previous
            expected=ref.step(target,q,v)
            actual=obj.step(q,v,gyro,gravity,command)
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
        obj.reset();np.testing.assert_array_equal(obj.integral,np.zeros(29));self.assertEqual(obj.tick,0)

    def test_stop_fault_and_stale_sample(self):
        obj=AssistController();obj.reset([0,0],[1,-1])
        obj.step([0,0],[1,-1],timestamp=1.)
        with self.assertRaises(ValueError): obj.step([0,0],[1,-1],timestamp=1.1)
        self.assertFalse(obj.ready)
        np.testing.assert_array_equal(obj.previous,[0,0])
        with self.assertRaises(RuntimeError): obj.step([0,0],[1,-1])
        obj.reset([0,0],[1,-1])
        with self.assertRaises(ValueError): obj.step([np.nan,0],[1,-1])
        obj.reset([0,0],[1,-1])
        np.testing.assert_array_equal(obj.step([0,0],[1,-1],enabled=False),[0,0])
        c,out=CController(),Pair(1,1)
        self.c.g1_assist_v7_reset(ct.byref(c),Pair(0,0),Pair(1,-1))
        self.assertEqual(self.c.g1_assist_v7_step(ct.byref(c),Pair(float('nan'),0),Pair(1,-1),1,out),-1)
        self.assertEqual(list(out),[0,0]);self.assertEqual(c.ready,0)
        self.assertEqual(self.c.g1_assist_v7_step(ct.byref(c),Pair(0,0),Pair(1,-1),1,out),1)


if __name__=='__main__': unittest.main()

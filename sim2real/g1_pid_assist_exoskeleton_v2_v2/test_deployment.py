"""Offline validation: exports, multi-step assist state and PID against Sim2Sim."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from controllers import AssistController, GaitPIDController, ROOT, DEFAULT_JOINT_POS
from numpy_policy import NumpyAssistPolicy

sys.path.insert(0, str(ROOT.parents[1] / "scripts/mujoco"))
import sim2sim_g1_assist_v1 as gait_reference
import sim2sim_g1_pid_assist_exoskeleton_v2_v2 as assist_reference

class DeploymentTests(unittest.TestCase):
    def test_network(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(128,150)).astype(np.float32)
        policy = torch.jit.load(str(ROOT / "weights/assist_policy.pt")).eval()
        with torch.inference_mode():
            expected = policy(torch.from_numpy(x)).numpy()
        actual = NumpyAssistPolicy(ROOT / "weights/assist_policy_v2.npz")(x)
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)

    def test_assist_sequence(self):
        c = AssistController(output_scale=1)
        q = np.array([-.1,-.1])
        v = np.zeros(2)
        c.reset(q,v)
        hist = assist_reference.V2AssistHistory(False,42)
        hist.reset(q,v)
        policy = torch.jit.load(str(ROOT / "weights/assist_policy.pt")).eval()
        previous = np.zeros(2)
        filtered = np.zeros(2)
        for i in range(600):
            t=i*.01
            q=-.1 + .4*np.sin(4*t+np.array([0,np.pi]))
            v=1.6*np.cos(4*t+np.array([0,np.pi]))
            if i: hist.append(q,v,previous)
            with torch.inference_mode():
                a=policy(torch.from_numpy(hist.observation()).unsqueeze(0)).squeeze(0).numpy()
            filtered += (1-np.exp(-.01/.05))*(v-filtered)
            target=np.clip(a,-1,1)*np.where(a<0,10.,4.)*assist_reference.joint_motion_gate(filtered)
            d=target-previous
            limit=np.where(d<0,80.,40.)*.01
            previous += np.clip(d,-limit,limit)
            actual=c.step(q,v)
            np.testing.assert_allclose(c.previous,previous,atol=3e-4,rtol=3e-4)
            np.testing.assert_allclose(actual,np.clip(previous,-8,4),atol=3e-4,rtol=3e-4)

    def test_pid(self):
        c=GaitPIDController()
        c.reset()
        integral=np.zeros(29)
        rng=np.random.default_rng(8)
        for _ in range(100):
            q=DEFAULT_JOINT_POS+rng.normal(0,.03,29)
            v=rng.normal(0,.1,29)
            got=c.step(q,v,[0,0,0],[0,0,-1],[.7,0,0])
            expected=gait_reference.pid_torque(c.target,q.astype(np.float32),v.astype(np.float32),integral)
            np.testing.assert_allclose(got,expected,atol=1e-6)

    def test_stop_and_bad_input(self):
        c=AssistController()
        c.reset([0,0],[0,0])
        np.testing.assert_array_equal(c.step([0,0],[1,1]),[0,0]) # default output scale zero
        np.testing.assert_array_equal(c.step([0,0],[0,0],enabled=False),[0,0])
        with self.assertRaises(RuntimeError): c.step([0,0],[0,0])
        c.reset([0,0],[0,0])
        with self.assertRaises(ValueError): c.step([np.nan,0],[0,0])
        self.assertFalse(c.ready)

if __name__ == "__main__":
    unittest.main()


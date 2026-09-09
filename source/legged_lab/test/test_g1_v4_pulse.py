"""Verify pulse shape under adversarial actor chatter and local phase changes."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
path=ROOT/'source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist_exoskeleton_v2_v4/mdp/pulse.py'
spec=importlib.util.spec_from_file_location('pulse',path)
pulse=importlib.util.module_from_spec(spec);spec.loader.exec_module(pulse)

class PulseTests(unittest.TestCase):
    def test_one_peak_despite_actor_chatter(self):
        c=pulse.SinglePulse(1,'cpu');out=[]
        for i in range(100):
            action=torch.tensor([[1. if i%2==0 else -1.]])
            out.append(c.step(action,torch.tensor([[-1.,1.]]),torch.ones(1,1)).item())
        x=np.abs(out);delta=np.diff(x)
        self.assertAlmostEqual(max(x),10.,places=5)
        peak=int(np.argmax(x))
        self.assertTrue(np.all(delta[:peak]>=-1e-5))
        self.assertTrue(np.all(delta[peak:]<=1e-5))
        self.assertLessEqual(max(abs(delta)),.80001)
        self.assertTrue(np.all(x[43:]==0))

    def test_stop_fades_monotonically_and_opposite_rearms(self):
        c=pulse.SinglePulse(1,'cpu');out=[]
        for i in range(130):
            v=[-1.,1.] if i<14 else [0.,0.] if i<65 else [1.,-1.]
            out.append(c.step(torch.ones(1,1),torch.tensor([v]),torch.ones(1,1)).item())
        self.assertTrue(np.all(np.diff(np.abs(out[16:50]))<=1e-5))
        self.assertLessEqual(max(abs(np.diff(out))),.80001)
        self.assertTrue(any(x<0 for x in out[:50]))
        self.assertTrue(any(x>0 for x in out[65:]))

    def test_gate_is_latched_and_reset_is_local(self):
        c=pulse.SinglePulse(2,'cpu')
        for i in range(8):
            out=c.step(torch.ones(2,1),torch.tensor([[-1.,1.],[1.,-1.]]),torch.ones(2,1))
        peak=c.state[:,2].clone()
        for _ in range(3): c.step(-torch.ones(2,1),torch.tensor([[-1.,1.],[1.,-1.]]),torch.zeros(2,1))
        torch.testing.assert_close(c.state[:,2],peak)
        before=c.state[1].clone();c.reset([0])
        torch.testing.assert_close(c.state[1],before)
        torch.testing.assert_close(c.state[0],torch.zeros(8))
        self.assertEqual(tuple(c.observation().shape),(2,8))

    def test_timing_cannot_break_rate_limit(self):
        with self.assertRaises(ValueError): pulse.SinglePulse(1,'cpu',duration=.1)
        with self.assertRaises(ValueError): pulse.SinglePulse(1,'cpu',release_duration=.05)

if __name__=='__main__': unittest.main()

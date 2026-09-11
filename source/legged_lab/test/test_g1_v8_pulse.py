"""Regression tests for false onset rejection, cadence, zero offsets and slew."""
import importlib.util
from pathlib import Path
import math
import unittest
import torch

BASE=Path(__file__).resolve().parents[1]/'legged_lab/tasks/locomotion/amp/config'
spec=importlib.util.spec_from_file_location('v8pulse',BASE/'g1_assist_exoskeleton_v2_v8/mdp/pulse.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class V8Tests(unittest.TestCase):
    def test_small_oscillations_never_start(self):
        # All phases, 0.5 to 1 degree amplitude, 0.5 to 4 Hz, 10 seconds.
        amps=torch.tensor([.5,1.,.5,1.,.5,1.])*math.pi/180
        freq=torch.tensor([.5,.5,2.,2.,4.,4.])
        c=m.SinglePulse(6,'cpu');vel=torch.zeros(6,2)
        for n in range(1000):
            phase=2*math.pi*freq*n*.01
            q=amps*phase.sin();v=amps*2*math.pi*freq*phase.cos()
            pos=torch.stack((q,-q),-1);raw=torch.stack((v,-v),-1)
            vel.lerp_(raw,1-math.exp(-.01/.03))
            out=c.step(torch.ones(6,2),vel,pos)
            self.assertFalse(c.started.any());self.assertEqual(out.abs().max().item(),0.)

    def test_first_step_and_zero_offset_invariance(self):
        c=m.SinglePulse(2,'cpu');offset=torch.tensor([[0.,0.],[.17,-.12]])
        starts=0
        for n in range(100):
            q=torch.tensor([[-.3*n*.01,0.]]).repeat(2,1)
            v=torch.tensor([[-.3,0.]]).repeat(2,1)
            out=c.step(torch.ones(2,2),v,q+offset)
            self.assertTrue(torch.allclose(out[0],out[1],atol=1e-5))
            self.assertTrue(torch.allclose(c.observation()[0],c.observation()[1],atol=1e-5))
            if c.started[0]:
                self.assertGreaterEqual(n*.003,math.radians(3));starts+=1
        self.assertEqual(starts,1)
        c.reset([0]);self.assertEqual(c.state[0].abs().sum().item(),0.)
        self.assertGreater(c.state[1].abs().sum().item(),0.)

    def test_fast_cadence_adapts_without_slew_violation(self):
        c=m.SinglePulse(1,'cpu');vel=torch.zeros(1,2);last=0.;starts=0;adapted=0
        for n in range(1400):
            # 0.35 s between opposite lifts.
            w=2*math.pi/0.7;t=n*.01
            q=torch.tensor([[.25*math.sin(w*t),-.25*math.sin(w*t)]])
            raw=torch.tensor([[.25*w*math.cos(w*t),-.25*w*math.cos(w*t)]])
            vel.lerp_(raw,1-math.exp(-.01/.03))
            out=c.step(torch.ones(1,2),vel,q).item()
            self.assertLessEqual(abs(out-last),.80002);self.assertLessEqual(abs(out),10.)
            last=out
            if c.started.item():
                starts+=1
                if c.state[0,17]>0:
                    adapted+=1
                    self.assertLess(c.state[0,8].item(),.6)
                    self.assertLess(c.state[0,2].item(),10.)
        self.assertGreater(adapted,20)
        print('Fast synthetic starts/adapted:',starts,adapted)

    def test_stopping_relocks_and_small_motion_stays_off(self):
        c=m.SinglePulse(1,'cpu')
        for n in range(180):
            t=n*.01;w=2*math.pi
            c.step(torch.ones(1,2),torch.tensor([[.3*w*math.cos(w*t),-.3*w*math.cos(w*t)]]),
                   torch.tensor([[.3*math.sin(w*t),-.3*math.sin(w*t)]]))
        for _ in range(200):c.step(torch.ones(1,2),torch.zeros(1,2),torch.zeros(1,2))
        self.assertEqual(c.previous.item(),0.);self.assertEqual(c.state[0,13].item(),0.)
        for n in range(300):
            w=4*math.pi;t=n*.01;a=math.radians(.5)
            out=c.step(torch.ones(1,2),torch.tensor([[a*w*math.cos(w*t),-a*w*math.cos(w*t)]]),
                       torch.tensor([[a*math.sin(w*t),-a*math.sin(w*t)]]))
            self.assertEqual(out.item(),0.)

    def test_peak_latched_and_no_repeat_on_constant_direction(self):
        c=m.SinglePulse(1,'cpu');started=0;last=0.
        for n in range(200):
            action=torch.ones(1,2) if n<20 else -torch.ones(1,2)
            out=c.step(action,torch.tensor([[-.4,0.]]),torch.tensor([[-.004*n,0.]])).item()
            started+=int(c.started.item())
            self.assertLessEqual(abs(out-last),.80002);last=out
            if 35<n<45:self.assertAlmostEqual(abs(out),10.,places=4)
        self.assertEqual(started,1)

if __name__=='__main__':unittest.main()

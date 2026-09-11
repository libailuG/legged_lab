"""v7 low-speed authority, plateau/release, state isolation and encoder bias checks."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist_exoskeleton_v2_v7/mdp'
spec=importlib.util.spec_from_file_location('pulse',BASE/'pulse.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('metrics',BASE/'response_metrics.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)

class V7Tests(unittest.TestCase):
    def test_slow_motion_reaches_full_peak_without_chatter(self):
        c=m.SinglePulse(1,'cpu');values=[]
        for n in range(180):
            a=torch.ones(1,2) if n<=5 else -torch.ones(1,2)
            values.append(c.step(a,torch.tensor([[-.1,.1]])).item())
        x=np.abs(values);peak=int(np.argmax(x))
        self.assertAlmostEqual(max(x),10,places=5)
        self.assertTrue(np.all(np.diff(x[:peak+1])>=-1e-5))
        self.assertTrue(np.all(np.diff(x[peak:])<=1e-5))
        self.assertLessEqual(max(abs(np.diff(values))),.80001)
        self.assertTrue(np.all(x[150:]==0))

    def test_terminal_quiet_hold_then_release(self):
        c=m.SinglePulse(1,'cpu');values=[]
        for n in range(130):
            v=torch.tensor([[-.2,.2]]) if n<35 else torch.zeros(1,2)
            values.append(c.step(torch.ones(1,2),v).item())
        self.assertAlmostEqual(abs(values[45]),10.,places=5)
        self.assertLess(abs(values[80]),10.)
        self.assertEqual(values[-1],0.)
        self.assertLessEqual(max(abs(np.diff(values))),.80001)

    def test_lowering_releases_early_and_new_direction_follows(self):
        c=m.SinglePulse(1,'cpu');out=[]
        for n in range(130):
            v=torch.tensor([[-.4,.4]]) if n<30 else torch.tensor([[.4,-.4]])
            out.append(c.step(torch.ones(1,2),v).item())
        self.assertTrue(all(v<=0 for v in out[:55]))
        self.assertTrue(any(v>0 for v in out[60:]))
        self.assertLessEqual(max(abs(np.diff(out))),.80001)

    def test_bias_is_episode_constant_actor_only_and_reset_local(self):
        tree=ast.parse((BASE/'actions.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FrozenGaitAssistTorqueAction')
        reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
        obs=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='biased_assist_position')
        reset.returns=None
        for arg in reset.args.args:arg.annotation=None
        scope={'torch':torch}
        exec(compile(ast.Module(body=[reset,obs],type_ignores=[]),str(BASE/'actions.py'),'exec'),scope)
        c=NS(_pulse=m.SinglePulse(2,'cpu'),_angle_bias=torch.zeros(2,2),cfg=NS(angle_bias_range=math.radians(10)),
             _policy_joint_ids=[0,1],_asset=NS(data=NS(default_joint_pos=torch.zeros(2,2))))
        names=['_raw_actions','_processed_actions','_previous_processed_actions','_requested_torque_delta',
               '_filtered_assist_joint_velocity','_motion_gate','_gait_actions','_previous_gait_actions',
               '_gait_position_targets','_extra_position_targets']
        for name in names:setattr(c,name,torch.zeros(2,2))
        c._gait_tick=torch.zeros(2,dtype=torch.long)
        scope['reset'](c)
        original=c._angle_bias.clone()
        q=torch.tensor([[.1,.2],[.3,.4]])
        env=NS(action_manager=NS(get_term=lambda _:c),scene={'robot':NS(data=NS(joint_pos=q))})
        asset=NS(name='robot',joint_ids=[0,1])
        for _ in range(10):torch.testing.assert_close(scope['biased_assist_position'](env,asset),q+original)
        scope['reset'](c,[0])
        torch.testing.assert_close(c._angle_bias[1],original[1])
        self.assertFalse(torch.equal(c._angle_bias[0],original[0]))
        self.assertLessEqual(float(c._angle_bias.abs().max()),math.radians(10))
        torch.testing.assert_close(q,torch.tensor([[.1,.2],[.3,.4]]))

    def test_completed_pulse_feedback_includes_zero_output(self):
        # Exercise the actual reward class without loading Isaac.
        tree=ast.parse((BASE/'rewards.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CompletedPulsePeakReward')
        pair_spec=importlib.util.spec_from_file_location('paired',BASE/'paired_assist.py')
        pair=importlib.util.module_from_spec(pair_spec);pair_spec.loader.exec_module(pair)
        class ManagerBase:
            def __init__(self,*args):pass
        scope={'torch':torch,'ManagerTermBase':ManagerBase,'ResponseMetrics':metrics.ResponseMetrics,'update_support':pair.update_support}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(BASE/'rewards.py'),'exec'),scope)
        def evaluate(peak, supported):
            pulse=m.SinglePulse(1,'cpu')
            robot=NS(data=NS(joint_pos=torch.tensor([[-.3,0.]]),default_joint_pos=torch.zeros(1,2),
                             joint_vel=torch.tensor([[-.2,.2]])))
            sensor=NS(data=NS(net_forces_w=torch.zeros(1,2,3)))
            if supported:sensor.data.net_forces_w[0,1,2]=250.
            env=NS(num_envs=1,device='cpu',step_dt=.01,extras={},scene={'robot':robot,'feet':sensor},
                   action_manager=NS(get_term=lambda _:NS(_pulse=pulse)))
            reward=scope['CompletedPulsePeakReward'](None,env)
            hip=NS(name='robot',joint_ids=[0,1]);foot=NS(name='feet',body_ids=[0,1])
            total=0.
            for _ in range(130):
                pulse.step(torch.tensor([[peak/5.-1.,0.]]),robot.data.joint_vel)
                total+=float(reward(env,hip,foot).item())*env.step_dt
            return total,env.extras['log'],reward
        # Lift=0.3 -> smoothstep(0.5)=0.5 -> target=8.0 Nm.
        zero,log,reward=evaluate(0.,True)
        matched,_,_=evaluate(8.0,True)
        unsupported,_,_=evaluate(0.,False)
        self.assertGreater(zero,.20)
        self.assertLess(matched,1e-8)
        self.assertEqual(unsupported,0.)
        self.assertEqual(float(log['Pulse/zero_peak_fraction']),1.)
        self.assertEqual(float(log['Pulse/supported_completion_fraction']),1.)
        reward.reset([0]);self.assertEqual(float(reward.stats.sum()),0.)

    def test_response_is_faster_than_v6(self):
        spec=importlib.util.spec_from_file_location('v6',BASE.parents[1]/'g1_assist_exoskeleton_v2_v6/mdp/pulse.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        timings=[]
        for module,tau in ((old,.05),(m,.03)):
            pulse=module.SinglePulse(1,'cpu');filtered=torch.zeros(1,2)
            metric=metrics.ResponseMetrics(1,'cpu',.01);values=[]
            start=None;half=None
            for n in range(180):
                velocity=torch.tensor([[-.2,.2]])
                filtered.lerp_(velocity,1-math.exp(-.01/tau))
                out=pulse.step(torch.ones(1,2),filtered).item();values.append(out)
                log=metric.update(velocity,pulse)
                if pulse.started.item() and start is None:start=n*.01
                if abs(out)>=5.-1e-6 and half is None:half=n*.01
            self.assertAlmostEqual(log['Response/mean_trigger_delay_s'].item(),start,places=5)
            self.assertAlmostEqual(log['Response/mean_half_peak_delay_s'].item(),half,places=5)
            self.assertLessEqual(max(abs(np.diff(values))),.80001)
            self.assertLessEqual(max(abs(np.array(values))),10.)
            timings.append((start,half))
            metric.reset();self.assertEqual(metric.stats.sum().item(),0.)
        self.assertLess(timings[1][0],timings[0][0])
        self.assertLess(timings[1][1],timings[0][1])
        print('v6/v7 synthetic trigger and half-peak delays (s):',timings)

    def test_reverse_still_requires_sixty_ms(self):
        c=m.SinglePulse(1,'cpu')
        for _ in range(30):c.step(torch.ones(1,2),torch.tensor([[-.2,.2]]))
        for _ in range(5):
            c.step(torch.ones(1,2),torch.tensor([[0.,-.2]]))
            self.assertEqual(c.state[0,3].item(),0.)
        c.step(torch.ones(1,2),torch.tensor([[0.,-.2]]))
        self.assertGreater(c.state[0,3].item(),0.)
        self.assertEqual(c.release_reason.item(),2.)

    def test_zero_peak_has_no_half_peak_event(self):
        c=m.SinglePulse(2,'cpu');metric=metrics.ResponseMetrics(2,'cpu',.01)
        for _ in range(150):
            v=torch.tensor([[-.2,.2],[-.2,.2]])
            c.step(-torch.ones(2,2),v);log=metric.update(v,c)
        self.assertEqual(log['Response/half_peak_events_per_env'].item(),0.)
        self.assertGreater(metric.stats[1,0].item(),0.)
        metric.reset([0]);self.assertEqual(metric.stats[0].sum().item(),0.)
        self.assertGreater(metric.stats[1,0].item(),0.)

    def test_rejects_invalid_ramp(self):
        with self.assertRaises(ValueError):m.SinglePulse(1,'cpu',rise_time=.1)
        with self.assertRaises(ValueError):m.SinglePulse(1,'cpu',duration_min=.2)

if __name__=='__main__':unittest.main()

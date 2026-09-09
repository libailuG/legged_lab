"""v6 latched peak/duration, smooth rise-hold-release, local velocity only."""
import math
import torch


class SinglePulse:
    def __init__(self, num_envs, device, dt=.01, rise_time=.25, release_duration=.25,
                 duration_min=.6, duration_max=1.4, torque_limit=10., rate_limit=80.,
                 confirm_time=.06, quiet_time=.3, quiet_min_elapsed=.5):
        values=(dt,rise_time,release_duration,duration_min,duration_max,torque_limit,rate_limit,confirm_time,quiet_time,quiet_min_elapsed)
        if not all(math.isfinite(x) and x>0 for x in values): raise ValueError('Positive finite pulse parameters required')
        if min(rise_time,release_duration)<math.pi*torque_limit/(2*rate_limit):
            raise ValueError('Ramp duration violates slew limit')
        if duration_min<rise_time+release_duration or duration_max<duration_min:
            raise ValueError('Invalid duration range')
        self.dt,self.rise,self.release=dt,rise_time,release_duration
        self.minimum,self.maximum=duration_min,duration_max
        self.limit,self.confirm=torque_limit,confirm_time
        self.quiet_time,self.quiet_min_elapsed=quiet_time,quiet_min_elapsed
        # elapsed, direction, peak, release elapsed, release origin,
        # consumed direction, candidate direction, candidate age, total duration, quiet age
        self.state=torch.zeros(num_envs,10,device=device)
        self.previous=torch.zeros(num_envs,1,device=device)
        self.started=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.finished=torch.zeros_like(self.started)
        self.completed_peak=torch.zeros(num_envs,device=device)
        # Diagnostics only: never included in actor observations or control decisions.
        for name in ("actual_peak", "release_reason", "completed_actual_peak",
                     "completed_duration", "completed_release_reason"):
            setattr(self, name, torch.zeros(num_envs, device=device))

    def reset(self,ids=slice(None)):
        self.state[ids]=0;self.previous[ids]=0
        self.started[ids]=False;self.finished[ids]=False;self.completed_peak[ids]=0
        for name in ("actual_peak", "release_reason", "completed_actual_peak",
                     "completed_duration", "completed_release_reason"):
            getattr(self, name)[ids] = 0

    def observation(self):
        scales=self.state.new_tensor([self.maximum,1,self.limit,self.release,self.limit,1,1,self.confirm,self.maximum,self.quiet_time])
        return self.state/scales

    def step(self,action,velocity):
        s=self.state
        delta=velocity[:,0]-velocity[:,1]
        direction=torch.where((delta<-.12)&(velocity[:,0]<-.08),-1.,
                              torch.where((delta>.12)&(velocity[:,1]<-.08),1.,0.))
        s[:,7]=torch.where(direction==s[:,6],(s[:,7]+self.dt).clamp(max=self.confirm),self.dt)
        s[:,6]=direction
        confirmed=s[:,7]>=self.confirm-1e-6
        quiet=velocity.abs().amax(dim=-1)<.04
        s[:,9]=torch.where(quiet,(s[:,9]+self.dt).clamp(max=self.quiet_time),0.)
        active=s[:,1]!=0
        releasing=s[:,3]>0
        # Extension of the currently assisted leg starts release; stillness alone
        # permits terminal support, with a finite quiet timeout and total duration.
        lowering=torch.where(s[:,1]<0,velocity[:,0],velocity[:,1])>.12
        # Reuse candidate confirmation for opposite flexion; lowering is additionally
        # filtered upstream (50 ms), and can release without knowing foot contact.
        reverse=confirmed&(direction!=0)&(direction!=s[:,1])
        quiet_end=(s[:,9]>=self.quiet_time-1e-6)&(s[:,0]>=self.quiet_min_elapsed)
        timed_end=s[:,0]>=s[:,8]-self.release-1e-6
        abort=active&~releasing&(lowering|reverse|quiet_end|timed_end)
        # Assign one cause, priority: lowering > opposite lift > quiet > timeout.
        reason=torch.where(lowering,1,torch.where(reverse,2,torch.where(quiet_end,3,4)))
        self.release_reason[abort]=reason[abort].to(s.dtype)
        s[abort,3]=self.dt;s[abort,4]=self.previous[abort,0].abs()
        releasing=s[:,3]>0
        s[~active&confirmed&(direction==0),5]=0
        start=~active&confirmed&(direction!=0)&(direction!=s[:,5])
        self.started[:]=start
        self.actual_peak[start]=0;self.release_reason[start]=0
        s[start,0]=0;s[start,1]=direction[start]
        s[start,2]=.5*(action[start,0].clamp(-1,1)+1)*self.limit
        s[start,8]=self.minimum+.5*(action[start,1].clamp(-1,1)+1)*(self.maximum-self.minimum)
        s[start,5]=direction[start];s[start,3:5]=0
        active=s[:,1]!=0
        s[active,0]+=self.dt
        rise=(s[:,0]/self.rise).clamp(0,1)
        rp=(s[:,3]/self.release).clamp(0,1)
        magnitude=torch.where(releasing,s[:,4]*torch.cos(.5*math.pi*rp).square(),
                              s[:,2]*torch.sin(.5*math.pi*rise).square())
        out=torch.where(active,s[:,1]*magnitude,0.).unsqueeze(-1)
        done=active&releasing&(s[:,3]>=self.release-1e-6)
        self.finished[:]=done
        self.completed_peak[:]=torch.where(done,s[:,2],0.)
        self.actual_peak[:]=torch.maximum(self.actual_peak,out[:,0].abs())
        self.completed_actual_peak[:]=torch.where(done,self.actual_peak,0.)
        self.completed_duration[:]=torch.where(done,s[:,0],0.)
        self.completed_release_reason[:]=torch.where(done,self.release_reason,0.)
        out[done]=0;s[done,:5]=0;s[done,8]=0
        s[releasing&~done,3]+=self.dt
        self.previous[:]=out
        return out

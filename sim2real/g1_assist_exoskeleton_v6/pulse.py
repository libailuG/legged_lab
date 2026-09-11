"""NumPy float32 v6 rise-hold-release state machine; local exoskeleton velocity only."""
import math
import numpy as np


class SinglePulse:
    def __init__(self, dt=.01, rise_time=.25, release_duration=.25, duration_min=.6,
                 duration_max=1.4, torque_limit=10., rate_limit=80., confirm_time=.06):
        values=(dt,rise_time,release_duration,duration_min,duration_max,torque_limit,rate_limit,confirm_time)
        if not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Positive finite pulse parameters required')
        if min(rise_time,release_duration)<math.pi*torque_limit/(2*rate_limit):raise ValueError('Ramp violates slew limit')
        if duration_min<rise_time+release_duration or duration_max<duration_min:raise ValueError('Invalid duration range')
        self.dt,self.rise,self.release,self.minimum,self.maximum,self.limit,self.confirm=map(np.float32,
            (dt,rise_time,release_duration,duration_min,duration_max,torque_limit,confirm_time))
        self.scale=np.array([duration_max,1,torque_limit,release_duration,torque_limit,1,1,confirm_time,duration_max,.3],dtype=np.float32)
        self.reset()

    def reset(self):
        self.state=np.zeros(10,dtype=np.float32)
        self.previous=np.float32(0)

    def observation(self):return self.state/self.scale

    def step(self,action,velocity):
        s=self.state
        delta=velocity[0]-velocity[1]
        direction=-1. if delta<-.12 and velocity[0]<-.08 else 1. if delta>.12 and velocity[1]<-.08 else 0.
        s[7]=min(s[7]+self.dt,self.confirm) if direction==s[6] else self.dt
        s[6]=direction
        confirmed=s[7]>=self.confirm-1e-6
        s[9]=min(s[9]+self.dt,np.float32(.3)) if np.max(np.abs(velocity))<.04 else 0
        active=s[1]!=0
        releasing=s[3]>0
        lowering=(velocity[0] if s[1]<0 else velocity[1])>.12
        reverse=confirmed and direction!=0 and direction!=s[1]
        quiet_end=s[9]>=.3-1e-6 and s[0]>=.5
        timed_end=s[0]>=s[8]-self.release-1e-6
        if active and not releasing and (lowering or reverse or quiet_end or timed_end):
            s[3]=self.dt;s[4]=abs(self.previous)
        releasing=s[3]>0
        if not active and confirmed and direction==0:s[5]=0
        if not active and confirmed and direction!=0 and direction!=s[5]:
            s[0]=0;s[1]=direction
            a=np.clip(np.asarray(action,dtype=np.float32),-1,1)
            s[2]=np.float32(.5)*(a[0]+np.float32(1))*self.limit
            s[8]=self.minimum+np.float32(.5)*(a[1]+np.float32(1))*(self.maximum-self.minimum)
            s[5]=direction;s[3:5]=0
        active=s[1]!=0
        if active:s[0]+=self.dt
        rise=np.float32(np.clip(s[0]/self.rise,0,1))
        rp=np.float32(np.clip(s[3]/self.release,0,1))
        wave=np.cos(np.float32(.5*math.pi)*rp) if releasing else np.sin(np.float32(.5*math.pi)*rise)
        magnitude=(s[4] if releasing else s[2])*wave*wave
        out=np.float32(s[1]*magnitude if active else 0)
        done=active and releasing and s[3]>=self.release-1e-6
        if done:out=np.float32(0);s[:5]=0;s[8]=0
        if releasing and not done:s[3]+=self.dt
        self.previous=out
        return out

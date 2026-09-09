"""NumPy float32 counterpart of v4's training SinglePulse (one environment)."""
import math
import numpy as np


class SinglePulse:
    def __init__(self, dt=.01, duration=.4, release_duration=.2, torque_limit=10., rate_limit=80., confirm_time=.03):
        values = [dt,duration,release_duration,torque_limit,rate_limit,confirm_time]
        if not all(math.isfinite(x) and x>0 for x in values):
            raise ValueError('Pulse parameters must be positive and finite')
        if duration < math.pi*torque_limit/rate_limit or release_duration < math.pi*torque_limit/(2*rate_limit):
            raise ValueError('Pulse timing violates slew limit')
        self.dt,self.duration,self.release_duration=map(np.float32,(dt,duration,release_duration))
        self.limit,self.confirm_time=map(np.float32,(torque_limit,confirm_time))
        self.scale=np.array([duration,1,torque_limit,release_duration,torque_limit,1,1,confirm_time],dtype=np.float32)
        self.reset()

    def reset(self):
        self.state=np.zeros(8,dtype=np.float32)
        self.previous=np.float32(0)

    def observation(self):
        return self.state/self.scale

    def step(self, action, velocity, onset_gate):
        s=self.state
        delta=velocity[0]-velocity[1]
        direction=-1. if delta<-.3 and velocity[0]<-.2 else 1. if delta>.3 and velocity[1]<-.2 else 0.
        s[7]=min(s[7]+self.dt,self.confirm_time) if direction==s[6] else self.dt
        s[6]=direction
        confirmed=s[7]>=self.confirm_time-1e-6
        active=s[1]!=0
        releasing=s[3]>0
        if active and not releasing and confirmed and direction!=s[1]:
            s[3]=self.dt;s[4]=abs(self.previous)
        releasing=s[3]>0
        if not active and confirmed and direction==0:s[5]=0
        if not active and confirmed and direction!=0 and direction!=s[5]:
            s[0]=0;s[1]=direction
            s[2]=np.float32(.5)*(np.float32(np.clip(action,-1,1))+np.float32(1))*self.limit*np.float32(onset_gate)
            s[5]=direction;s[3:5]=0
        active=s[1]!=0
        if active and not releasing:s[0]+=self.dt
        phase=np.float32(np.clip(s[0]/self.duration,0,1))
        rp=np.float32(np.clip(s[3]/self.release_duration,0,1))
        if releasing:
            magnitude=s[4]*np.cos(np.float32(.5*math.pi)*rp)**np.float32(2)
        else:
            magnitude=s[2]*np.sin(np.float32(math.pi)*phase)**np.float32(2)
        output=np.float32(s[1]*magnitude if active else 0)
        done=active and ((not releasing and s[0]>=self.duration-1e-6) or (releasing and s[3]>=self.release_duration-1e-6))
        if done:output=np.float32(0);s[:5]=0
        if releasing and not done:s[3]+=self.dt
        self.previous=output
        return output

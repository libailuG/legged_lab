"""NumPy float32 v8 rise-hold-release state machine; local exoskeleton velocity only."""
import math
import numpy as np


class SinglePulse:
    def __init__(self, dt=.01, rise_time=.20, release_duration=.25, duration_min=.6,
                 duration_max=1.4, torque_limit=10., rate_limit=80., confirm_time=.03):
        values=(dt,rise_time,release_duration,duration_min,duration_max,torque_limit,rate_limit,confirm_time)
        if not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Positive finite pulse parameters required')
        if min(rise_time,release_duration)<math.pi*torque_limit/(2*rate_limit):raise ValueError('Ramp violates slew limit')
        if duration_min<rise_time+release_duration or duration_max<duration_min:raise ValueError('Invalid duration range')
        self.dt,self.rise,self.release,self.minimum,self.maximum,self.limit,self.confirm=map(np.float32,
            (dt,rise_time,release_duration,duration_min,duration_max,torque_limit,confirm_time))
        self.candidate_window=np.float32(max(confirm_time,.06))
        self.rate=np.float32(rate_limit)
        self.scale=np.array([duration_max,1,torque_limit,release_duration,torque_limit,1,1,self.candidate_window,duration_max,.3,.1,.1,1.5,1,1.2,rise_time,release_duration,1],dtype=np.float32)
        self.reset()

    def reset(self):
        self.state=np.zeros(18,dtype=np.float32)
        self.previous=np.float32(0)
        self.displacement=np.zeros(2,dtype=np.float32)

    def observation(self):
        obs=self.state.copy();obs[10:12]=self.displacement
        return obs/self.scale

    def step(self,action,velocity,position):
        s=self.state
        delta=velocity[0]-velocity[1]
        direction=-1. if delta<-.12 and velocity[0]<-.08 else 1. if delta>.12 and velocity[1]<-.08 else 0.
        if direction!=s[6]:s[10:12]=position
        self.displacement[:]=s[10:12]-position
        s[12]=min(s[12]+self.dt,np.float32(1.5))
        if s[12]>=1.5-1e-6:s[13]=0;s[17]=0
        walking=s[13]!=0
        threshold=np.float32(math.radians(2.5 if walking else 3.))
        travel=self.displacement[0 if direction<0 else 1]
        s[7]=min(s[7]+self.dt,self.candidate_window) if direction==s[6] else self.dt
        s[6]=direction
        confirmed=s[7]>=self.confirm-1e-6
        qualified=confirmed and direction!=0 and travel>=threshold
        event=qualified and direction!=s[13]
        interval=s[12]
        if event and walking and interval>=np.float32(.2) and interval<=np.float32(1.2):
            s[14]=np.float32(.5)*s[14]+np.float32(.5)*interval if s[17]>0 else interval
            s[17]=1
        if event:s[12]=0;s[13]=direction
        s[9]=min(s[9]+self.dt,np.float32(.3)) if np.max(np.abs(velocity))<.04 else 0
        active=s[1]!=0
        releasing=s[3]>0
        lowering=(velocity[0] if s[1]<0 else velocity[1])>.12
        reverse=s[7]>=.06-1e-6 and direction!=0 and direction!=s[1]
        quiet_end=s[9]>=.3-1e-6 and s[0]>=.5
        timed_end=s[0]>=s[8]-s[16]-1e-6
        if active and not releasing and (lowering or reverse or quiet_end or timed_end):
            tail=max(np.float32(math.pi)*abs(self.previous)/(np.float32(2)*self.rate)+self.dt,np.float32(2)*self.dt)
            s[16]=min(s[16],tail)
            s[3]=self.dt;s[4]=abs(self.previous)
        releasing=s[3]>0
        if not active and confirmed and direction==0:s[5]=0
        if not active and qualified and direction!=s[5]:
            s[0]=0;s[1]=direction
            a=np.clip(np.asarray(action,dtype=np.float32),-1,1)
            s[2]=np.float32(.5)*(a[0]+np.float32(1))*self.limit
            s[8]=self.minimum+np.float32(.5)*(a[1]+np.float32(1))*(self.maximum-self.minimum)
            if s[17]>0:s[8]=min(s[8],np.clip(np.float32(.85)*s[14],np.float32(.3),self.maximum))
            s[15]=min(self.rise,np.float32(.45)*s[8])
            s[16]=min(self.release,np.float32(.45)*s[8])
            feasible=np.float32(2)*self.rate*min(s[15],s[16])/np.float32(math.pi)
            s[2]=min(s[2],feasible)
            s[5]=direction;s[3:5]=0
        active=s[1]!=0
        if active:s[0]+=self.dt
        rise=np.float32(np.clip(s[0]/max(s[15],self.dt),0,1))
        rp=np.float32(np.clip(s[3]/max(s[16],self.dt),0,1))
        wave=np.cos(np.float32(.5*math.pi)*rp) if releasing else np.sin(np.float32(.5*math.pi)*rise)
        magnitude=(s[4] if releasing else s[2])*wave*wave
        out=np.float32(s[1]*magnitude if active else 0)
        done=active and releasing and s[3]>=s[16]-1e-6
        if done:out=np.float32(0);s[:5]=0;s[8]=0
        if releasing and not done:s[3]+=self.dt
        self.previous=out
        return out

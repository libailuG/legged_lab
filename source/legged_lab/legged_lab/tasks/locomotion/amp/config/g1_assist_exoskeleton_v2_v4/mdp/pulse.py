"""Single-lobe paired assist trajectory driven only by local exoskeleton velocity."""
import math
import torch


class SinglePulse:
    def __init__(self, num_envs, device, dt=.01, duration=.4, release_duration=.2,
                 torque_limit=10., rate_limit=80., confirm_time=.03):
        if dt <= 0 or duration < math.pi*torque_limit/rate_limit or release_duration < math.pi*torque_limit/(2*rate_limit):
            raise ValueError('Pulse timing must respect torque slew limit')
        self.dt, self.duration, self.release_duration = dt, duration, release_duration
        self.limit, self.confirm_time = torque_limit, confirm_time
        self.state = torch.zeros(num_envs, 8, device=device)
        # elapsed, active direction, peak, release elapsed, release start,
        # consumed direction, candidate direction, candidate age
        self.previous = torch.zeros(num_envs, 1, device=device)

    def reset(self, ids=slice(None)):
        self.state[ids] = 0
        self.previous[ids] = 0

    def observation(self):
        scale = self.state.new_tensor([self.duration,1,self.limit,self.release_duration,
                                       self.limit,1,1,self.confirm_time])
        return self.state / scale

    def step(self, action, velocity, onset_gate):
        s = self.state
        # Negative hip velocity means flexion; choose the faster flexing leg.
        delta = velocity[:,0]-velocity[:,1]
        direction = torch.where((delta < -.3) & (velocity[:,0] < -.2), -1.,
                                torch.where((delta > .3) & (velocity[:,1] < -.2), 1., 0.))
        same = direction == s[:,6]
        s[:,7] = torch.where(same, (s[:,7]+self.dt).clamp(max=self.confirm_time), self.dt)
        s[:,6] = direction
        confirmed = s[:,7] >= self.confirm_time-1e-6
        active = s[:,1] != 0
        releasing = s[:,3] > 0
        # A confirmed stop or opposite flexion ends a pulse with a monotone fade.
        abort = active & ~releasing & confirmed & (direction != s[:,1])
        s[abort,3] = self.dt
        s[abort,4] = self.previous[abort,0].abs()
        releasing = s[:,3] > 0
        # A quiet interval rearms the next local movement, never retriggers mid-swing.
        s[~active & confirmed & (direction == 0),5] = 0
        start = ~active & confirmed & (direction != 0) & (direction != s[:,5])
        s[start,0] = 0
        s[start,1] = direction[start]
        # The policy selects only pulse strength; direction is local-encoder based.
        s[start,2] = (.5 * (action[start,0].clamp(-1,1) + 1.)) * self.limit * onset_gate[start,0]
        s[start,5] = direction[start]
        s[start,3:5] = 0
        active = s[:,1] != 0
        s[active & ~releasing,0] += self.dt
        phase = (s[:,0]/self.duration).clamp(0,1)
        envelope = torch.sin(math.pi*phase).square()
        release_phase = (s[:,3]/self.release_duration).clamp(0,1)
        magnitude = torch.where(releasing, s[:,4]*torch.cos(.5*math.pi*release_phase).square(),
                                s[:,2]*envelope)
        output = torch.where(active, s[:,1]*magnitude, 0.).unsqueeze(-1)
        done = active & ((~releasing & (s[:,0] >= self.duration-1e-6)) |
                         (releasing & (s[:,3] >= self.release_duration-1e-6)))
        output[done] = 0
        s[done,0:5] = 0
        s[releasing & ~done,3] += self.dt
        self.previous[:] = output
        return output

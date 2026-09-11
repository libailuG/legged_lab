"""Training-only response timings from unfiltered hip velocity; no actor inputs."""
import torch


class ResponseMetrics:
    def __init__(self, num_envs, device, dt):
        self.dt = dt
        self.age = torch.zeros(num_envs, device=device)
        self.direction = torch.zeros_like(self.age)
        self.onset_delay = torch.zeros_like(self.age)
        self.elapsed = torch.zeros_like(self.age)
        self.pending = torch.zeros_like(self.age, dtype=torch.bool)
        # matched starts, summed trigger delay, half-peak events, summed half delay,
        # all starts; zero peaks remain starts but cannot yield a half-peak event.
        self.stats = torch.zeros(num_envs, 5, device=device)

    def reset(self, ids=slice(None)):
        for name in ('age', 'direction', 'onset_delay', 'elapsed', 'pending', 'stats'):
            getattr(self, name)[ids] = 0

    def update(self, velocity, pulse):
        delta = velocity[:, 0] - velocity[:, 1]
        direction = torch.where((delta < -.12) & (velocity[:, 0] < -.08), -1.,
                               torch.where((delta > .12) & (velocity[:, 1] < -.08), 1., 0.))
        # Time since first consecutive qualifying sample, not physiological intent.
        self.age[:] = torch.where((direction != 0) & (direction == self.direction),
                                  self.age + self.dt, 0.)
        self.direction[:] = direction
        self.elapsed += self.dt
        start = pulse.started
        matched = start & (direction != 0) & (direction == pulse.state[:, 1])
        self.pending[start] = False
        self.pending[matched] = pulse.state[matched, 2] > .1
        self.onset_delay[matched] = self.age[matched]
        self.elapsed[start] = 0.
        self.stats[:, 0] += matched
        self.stats[:, 1] += torch.where(matched, self.age, 0.)
        self.stats[:, 4] += start
        reached = (self.pending & (pulse.state[:, 1] != 0)
                   & (pulse.previous[:, 0].abs() >= .5 * pulse.state[:, 2] - 1e-6))
        self.stats[:, 2] += reached
        self.stats[:, 3] += torch.where(reached, self.onset_delay + self.elapsed, 0.)
        self.pending[reached | pulse.finished] = False
        totals = self.stats.sum(0)
        return {
            'Response/mean_trigger_delay_s': totals[1] / totals[0].clamp_min(1.),
            'Response/mean_half_peak_delay_s': totals[3] / totals[2].clamp_min(1.),
            'Response/matched_start_fraction': totals[0] / totals[4].clamp_min(1.),
            'Response/half_peak_events_per_env': self.stats[:, 2].mean(),
            'Response/matched_starts_per_env': self.stats[:, 0].mean(),
        }

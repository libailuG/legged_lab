"""100 Hz paired assistance core; no hardware communication or G1 state input."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

try:
    from .numpy_policy import NumpyAssistPolicy, V5AssistHistory
    from .pulse import SinglePulse
except ImportError:
    from numpy_policy import NumpyAssistPolicy, V5AssistHistory
    from pulse import SinglePulse


def vector(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be two finite values [left, right]')
    return result


class AssistController:
    """reset(q,dq), then step(q,dq) at 100 Hz. Returns joint torque [T,-T].

    Input must already be calibrated into training joint coordinates. Stop/errors
    latch the controller off until reset. The caller must transmit zero/disable
    on exceptions and provide a hardware-side communication watchdog.
    """
    def __init__(self, weights=None):
        directory = Path(weights or Path(__file__).resolve().parent / 'weights')
        manifest = json.loads((directory / 'manifest.json').read_text())
        weights_path = directory / 'assist_policy.npz'
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != manifest['npz_sha256']:
            raise ValueError('Policy weights hash mismatch')
        self.policy = NumpyAssistPolicy(weights_path)
        self.cfg = manifest['control']
        c = self.cfg
        if not all(math.isfinite(v) for v in c.values()) or c['dt'] != .01 or c['torque_limit'] != 10:
            raise ValueError('Expected finite v5 parameters / 100 Hz / +/-10 Nm')
        if c['torque_rate_limit'] <= 0 or c['motion_filter_time_constant'] <= 0 or not 0 <= c['motion_speed_deadzone'] < c['motion_speed_full']:
            raise ValueError('Invalid motion gate or rate limit')
        self.pulse = SinglePulse(c["dt"], c["pulse_rise_time"], c["pulse_release_duration"],
                                 c["pulse_duration_min"], c["pulse_duration_max"],
                                 c["torque_limit"], c["torque_rate_limit"], c["pulse_confirm_time"])
        self.history = V5AssistHistory()
        self.stop()

    def stop(self):
        self.ready = False
        self.first = True
        self.scalar = 0.
        self.filtered_velocity = np.zeros(2,dtype=np.float32)
        self.pulse.reset()
        self.previous = np.zeros(2, dtype=np.float32)
        self.raw_action = np.zeros(2,dtype=np.float32)
        self.gate = 0.
        self.last_timestamp = None
        self.history.reset(np.zeros(2), np.zeros(2))
        return self.previous.copy()

    def reset(self, position, velocity):
        self.stop()
        self.history.reset(vector(position, 'position rad'), vector(velocity, 'velocity rad/s'))
        self.ready = True

    def step(self, position, velocity, *, enabled=True, timestamp=None):
        if not enabled:
            return self.stop()
        if not self.ready:
            raise RuntimeError('Controller stopped; reset required')
        try:
            p, v = vector(position, 'position rad'), vector(velocity, 'velocity rad/s')
            if timestamp is not None:
                timestamp = float(timestamp)
                if not math.isfinite(timestamp):
                    raise ValueError('Invalid sample timestamp')
                if self.last_timestamp is not None and not .005 <= timestamp-self.last_timestamp <= .015:
                    raise ValueError('Nonmonotonic, missing or mistimed 100 Hz sample; reset required')
            elif self.last_timestamp is not None:
                raise ValueError('Sample timestamps cannot be omitted after timestamped operation')
            self.last_timestamp = timestamp
            if self.first:
                # Fill first observation with the actual first control sample.
                self.history.reset(p, v)
            else:
                self.history.append(p, v, self.previous, np.concatenate((self.pulse.observation(), self.filtered_velocity)))
            self.first = False
            output = np.asarray(self.policy(self.history.observation()))
            if output.shape != (2,) or not np.isfinite(output).all():
                raise ValueError('Invalid policy output')
            self.raw_action = output.astype(np.float32,copy=True)
            c = self.cfg
            self.filtered_velocity += (1-math.exp(-c['dt']/c['motion_filter_time_constant'])) * (v-self.filtered_velocity)
            phase = float(np.clip((np.max(np.abs(self.filtered_velocity))-c['motion_speed_deadzone']) /
                                  (c['motion_speed_full']-c['motion_speed_deadzone']), 0., 1.))
            self.gate = phase * phase * (3-2*phase)
            self.scalar = float(self.pulse.step(self.raw_action, self.filtered_velocity))
            self.previous = np.array([self.scalar, -self.scalar], dtype=np.float32)
            return self.previous.copy()
        except Exception:
            self.stop()
            raise

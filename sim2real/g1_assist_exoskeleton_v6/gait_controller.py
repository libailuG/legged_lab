"""Optional G1 robot bench controller; never required by the human exoskeleton actor."""
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent

def vector(value, size, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected {size} finite values")
    return value

POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_JOINT_POS = np.array(
    [
        -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.3, 0.3, 0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0.0, 0.0,
        0.0, 0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# Values mirror g1_assist_v1/robot_cfg.py; kept local for independent editing.
KP = np.array(
    [
        180.0, 180.0, 360.0, 180.0, 180.0, 72.0, 180.0, 180.0, 72.0,
        270.0, 270.0, 56.0, 56.0, 72.0, 72.0, 56.0, 56.0, 72.0, 72.0,
        56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
    ],
    dtype=np.float64,
)
KD = np.array(
    [
        3.0, 3.0, 7.5, 3.0, 3.0, 7.5, 3.0, 3.0, 7.5,
        6.0, 6.0, 1.3, 1.3, 3.0, 3.0, 1.3, 1.3, 3.0, 3.0,
        1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3,
    ],
    dtype=np.float64,
)
EFFORT_LIMIT = np.array(
    [
        200.0, 200.0, 200.0, 316.0, 316.0, 57.0, 200.0, 200.0, 57.0,
        316.0, 316.0, 38.0, 38.0, 57.0, 57.0, 38.0, 38.0, 57.0, 57.0,
        38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 8.0, 8.0, 8.0, 8.0,
    ],
    dtype=np.float64,
)


class GaitPIDController:
    """1 kHz step; 50 Hz policy. Input vectors follow POLICY_JOINT_NAMES.

    gyro and gravity are expressed in the pelvis frame. Gravity is a unit
    direction, not acceleration in m/s^2. The returned values are torques (Nm).
    """
    def __init__(self, policy=None):
        import torch
        self.torch = torch
        self.policy = torch.jit.load(str(policy or ROOT / "weights/gait_policy.pt"), map_location="cpu").eval()
        self.ready = False
        self.integral = np.zeros(29)
        self.last_action = np.zeros(29)
        self.target = DEFAULT_JOINT_POS.copy()
        self.tick = 0

    def stop(self):
        self.ready = False
        self.integral.fill(0)
        self.last_action.fill(0)
        self.target[:] = DEFAULT_JOINT_POS
        self.tick = 0
        return np.zeros(29)

    def reset(self):
        self.stop()
        self.ready = True

    def step(self, position, velocity, angular_velocity, projected_gravity, command, enabled=True):
        if not enabled:
            return self.stop()
        if not self.ready:
            raise RuntimeError("Controller stopped: call reset before enabling")
        try:
            q = vector(position, 29, "joint position")
            dq = vector(velocity, 29, "joint velocity")
            gyro = vector(angular_velocity, 3, "pelvis angular velocity")
            gravity = vector(projected_gravity, 3, "projected gravity")
            cmd = vector(command, 3, "vx,vy,yaw")
            if self.tick % 20 == 0:
                obs = np.concatenate((gyro, gravity, cmd, q - DEFAULT_JOINT_POS, dq, self.last_action)).astype(np.float32)
                with self.torch.inference_mode():
                    action = self.policy(self.torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                self.last_action[:] = vector(action, 29, "gait output")
                self.target[:] = DEFAULT_JOINT_POS + 0.25 * self.last_action
            error = self.target - q
            candidate_i = np.clip(self.integral + 0.1 * KP * error * 0.001, -10, 10)
            candidate = KP * error - KD * dq + candidate_i
            clipped = np.clip(candidate, -EFFORT_LIMIT, EFFORT_LIMIT)
            accept = ~((candidate != clipped) & (error * candidate > 0))
            self.integral[accept] = candidate_i[accept]
            self.tick += 1
            return np.clip(KP * error - KD * dq + self.integral, -EFFORT_LIMIT, EFFORT_LIMIT)
        except Exception:
            self.stop()
            raise

def rear_mechanism_pd(position, velocity):
    """Slider m and cylinder rad -> slider N and cylinder Nm."""
    p = vector(position, 2, "rear position")
    v = vector(velocity, 2, "rear velocity")
    return np.clip(-np.array([10000.,4.]) * p - np.array([20.,2.]) * v, [-300,-200], [300,200])


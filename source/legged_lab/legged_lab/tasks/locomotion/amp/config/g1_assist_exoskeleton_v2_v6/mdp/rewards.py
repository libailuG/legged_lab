"""Training rewards for paired assistance; privileged signals never enter action processing."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .paired_assist import (
    bilateral_rest_cost, support_aware_reference, unsupported_press_cost, update_support,
)


def _motion_command_magnitude(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_scale: float,
) -> torch.Tensor:
    """Return a scalar motion magnitude from planar velocity and yaw-rate commands."""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :2], dim=-1) + yaw_scale * command[:, 2].abs()


def assist_torque_rate_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_rate_limit: float = 80.0,
) -> torch.Tensor:
    """Penalize the normalized actual assist-torque rate after rate limiting."""
    action = env.action_manager.get_term(action_name)
    torque_rate = action.assist_torque_rate
    normalized_rate = torque_rate / torque_rate_limit
    return torch.square(normalized_rate).mean(dim=-1)


def assist_requested_torque_rate_excess_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "assist_torque",
    torque_rate_limit: float = 80.0,
) -> torch.Tensor:
    """Penalize only the part of the requested torque step above the slew limit.

    Unlike the post-limiter rate penalty, this remains informative when the
    actuator command is saturated at the configured slew-rate limit.
    """
    if torque_rate_limit <= 0.0:
        raise ValueError("Torque-rate limit must be greater than zero")
    action = env.action_manager.get_term(action_name)
    requested_delta = action.requested_torque_delta
    allowed_delta = torque_rate_limit * env.step_dt
    excess = torch.clamp(requested_delta.abs() - allowed_delta, min=0.0)
    # Normalize by the per-control-step allowance.  Normalizing by the full
    # torque range made large, rate-saturated requests look artificially small.
    return torch.square(excess / allowed_delta).mean(dim=-1)


def rear_mechanism_zero_position_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    slider_position_scale: float,
    cylinder_position_scale: float,
) -> torch.Tensor:
    """Penalize normalized slider and cylinder position errors from zero."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 2:
        raise ValueError(f"Expected two rear-mechanism joints, got {joint_pos.shape[-1]}")
    if slider_position_scale <= 0.0 or cylinder_position_scale <= 0.0:
        raise ValueError("Rear-mechanism position scales must be greater than zero")
    slider_error = torch.square(joint_pos[:, 0] / slider_position_scale)
    cylinder_error = torch.square(joint_pos[:, 1] / cylinder_position_scale)
    return 0.5 * (slider_error + cylinder_error)


def rear_mechanism_velocity_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
    slider_velocity_scale: float, cylinder_velocity_scale: float,
) -> torch.Tensor:
    """Penalize actual rear-mechanism motion, not just displacement from zero."""
    if slider_velocity_scale <= 0 or cylinder_velocity_scale <= 0:
        raise ValueError("Rear-mechanism velocity scales must be positive")
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    if velocity.shape[-1] != 2:
        raise ValueError("Expected slider and cylinder joints in that order")
    return 0.5 * ((velocity[:, 0] / slider_velocity_scale).square()
                  + (velocity[:, 1] / cylinder_velocity_scale).square())


def hip_motor_torque_burden_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    torque_scale: float = 40.0,
) -> torch.Tensor:
    """Penalize the physical hip motors' normalized squared torque burden."""
    if torque_scale <= 0.0:
        raise ValueError("torque_scale must be greater than zero")
    robot: Articulation = env.scene[hip_cfg.name]
    hip_torque = robot.data.applied_torque[:, hip_cfg.joint_ids]
    return torch.square(hip_torque / torque_scale).mean(dim=-1)


def hip_motor_mechanical_power_l1(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    power_scale: float = 100.0,
) -> torch.Tensor:
    """Penalize absolute physical hip-motor mechanical power."""
    if power_scale <= 0.0:
        raise ValueError("power_scale must be greater than zero")
    robot: Articulation = env.scene[hip_cfg.name]
    hip_torque = robot.data.applied_torque[:, hip_cfg.joint_ids]
    hip_vel = robot.data.joint_vel[:, hip_cfg.joint_ids]
    return torch.abs(hip_torque * hip_vel).mean(dim=-1) / power_scale


class SupportAwareAssistReward(ManagerTermBase):
    """Privileged paired reference, unsupported press, and bilateral-rest costs.

    Every configured term owns its own identically updated support/filter state;
    one call per reward step, with episode-local state cleared on reset.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered_acceleration = torch.zeros(env.num_envs, 2, device=env.device)
        self._contact = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)
        self._contact_age = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._filtered_acceleration[env_ids] = 0.0
        self._contact[env_ids] = False
        self._contact_age[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        mode: str,
        hip_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        action_name: str = "assist_torque",
        torque_limit: float = 10.0,
        velocity_gain: float = 2.0,
        acceleration_gain: float = 0.04,
        dynamics_scale: float = 1.5,
        acceleration_filter_time_constant: float = 0.08,
        lift_speed_deadzone: float = 0.15,
        lift_speed_full: float = 0.8,
        support_force_off: float = 10.0,
        support_force_on: float = 20.0,
        support_force_full: float = 200.0,
        support_rise_time: float = 0.05,
        rest_velocity_scale: float = 0.5,
        rest_acceleration_scale: float = 3.0,
    ) -> torch.Tensor:
        if torque_limit <= 0 or acceleration_filter_time_constant <= 0:
            raise ValueError("Torque limit and filter time constant must be positive")
        if rest_velocity_scale <= 0 or rest_acceleration_scale <= 0:
            raise ValueError("Rest motion scales must be positive")
        robot = env.scene[hip_cfg.name]
        hip_vel = robot.data.joint_vel[:, hip_cfg.joint_ids]
        hip_acc = robot.data.joint_acc[:, hip_cfg.joint_ids]
        torque = env.action_manager.get_term(action_name).processed_actions
        alpha = 1.0 - math.exp(-env.step_dt / acceleration_filter_time_constant)
        self._filtered_acceleration.lerp_(hip_acc, alpha)
        sensor = env.scene[sensor_cfg.name]
        force_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
        contact, age, support = update_support(
            force_z, self._contact, self._contact_age, env.step_dt,
            support_force_off, support_force_on, support_force_full, support_rise_time,
        )
        self._contact[:] = contact
        self._contact_age[:] = age

        if mode in ("posture_support", "standing_rest"):
            # Training-only posture proxy, not an inverse-dynamics torque estimate.
            q = robot.data.joint_pos[:, hip_cfg.joint_ids]
            neutral = robot.data.default_joint_pos[:, hip_cfg.joint_ids]
            lift = (neutral - q).clamp_min(0.)
            posture = ((lift - 0.10) / 0.40).clamp(0., 1.)
            posture = posture.square() * (3. - 2. * posture)
            if mode == "standing_rest":
                # Quiet elevated legs are not ordinary upright standing.
                quiet = 1. / (1. + (hip_vel / 0.25).square().amax(dim=-1))
                upright = 1. - posture.amax(dim=-1)
                return (torque / torque_limit).square().mean(dim=-1) * quiet * upright
            eligible = (1. - support) * support.flip(dims=(-1,))
            lowering = (hip_vel.clamp_min(0.) / .25).clamp(0., 1.)
            demand = posture * eligible * (1. - lowering)
            reference = torque_limit * (demand[:, 1:2] - demand[:, 0:1])
            reference = torch.cat((reference, -reference), dim=-1)
            # Evaluate low-speed elevated postures without encouraging arbitrary
            # torque elsewhere; existing costs cover unsafe/unnecessary effort.
            weight = (posture * eligible).amax(dim=-1)
            return ((torque - reference) / torque_limit).square().mean(dim=-1) * weight

        if mode == "supported_work":
            # Positive flexion work only when the opposite foot bears load.
            work = (-torque).clamp_min(0.) * (-hip_vel).clamp(0., 2.) / (torque_limit * 2.)
            eligible = (1. - support) * support.flip(dims=(-1,))
            return (work * eligible).mean(dim=-1)
        if mode == "lowering_resistance":
            # Penalize lift torque resisting extension, even when pressing the other foot is legal.
            return ((-torque).clamp_min(0.) * hip_vel.clamp(0., 2.) / (torque_limit * 2.)).mean(dim=-1)
        if mode == "tracking":
            reference = support_aware_reference(
                hip_vel, self._filtered_acceleration, support, torque_limit,
                velocity_gain, acceleration_gain, dynamics_scale,
                lift_speed_deadzone, lift_speed_full,
            )
            return ((torque - reference) / torque_limit).square().mean(dim=-1)
        if mode == "unsupported_press":
            return unsupported_press_cost(torque, support, torque_limit)
        if mode == "bilateral_rest":
            return bilateral_rest_cost(
                torque, hip_vel, self._filtered_acceleration,
                torque_limit, rest_velocity_scale, rest_acceleration_scale,
            )
        raise ValueError(f"Unsupported paired-assist reward mode: {mode}")


def assist_torque_alignment(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    torque_limit: float = 10.0,
    minimum_reference_torque: float = 0.25,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.05,
    yaw_scale: float = 0.3,
) -> torch.Tensor:
    """Reward useful same-direction assist torque only under motion commands."""
    robot: Articulation = env.scene[hip_cfg.name]
    hip_torque = robot.data.applied_torque[:, hip_cfg.joint_ids]
    assist_torque = robot.data.applied_torque[:, assist_cfg.joint_ids]

    hip_magnitude = hip_torque.abs()
    signed_assistance = assist_torque * torch.sign(hip_torque)
    useful_torque = torch.clamp(signed_assistance, min=0.0)
    useful_torque = torch.minimum(useful_torque, hip_magnitude)
    useful_ratio = useful_torque / hip_magnitude.clamp_min(minimum_reference_torque)

    excess = torch.clamp(assist_torque.abs() - hip_magnitude, min=0.0)
    excess_penalty = torch.square(excess / torque_limit)
    moving = _motion_command_magnitude(env, command_name, yaw_scale) > moving_threshold
    return (useful_ratio - excess_penalty).mean(dim=-1) * moving.float()


def assist_torque_zero_when_standing(
    env: ManagerBasedRLEnv,
    assist_cfg: SceneEntityCfg,
    torque_limit: float = 10.0,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.05,
    yaw_scale: float = 0.3,
) -> torch.Tensor:
    """Penalize normalized assist torque only when standing is commanded."""
    robot: Articulation = env.scene[assist_cfg.name]
    assist_torque = robot.data.applied_torque[:, assist_cfg.joint_ids]
    standing = _motion_command_magnitude(env, command_name, yaw_scale) <= moving_threshold
    normalized_torque_l2 = torch.square(assist_torque / torque_limit).mean(dim=-1)
    return normalized_torque_l2 * standing.float()


def assist_hip_angle_error_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize continuous assist-to-hip joint position mismatch."""
    robot: Articulation = env.scene[hip_cfg.name]
    error = (
        robot.data.joint_pos[:, assist_cfg.joint_ids]
        - robot.data.joint_pos[:, hip_cfg.joint_ids]
    )
    return torch.square(error).mean(dim=-1)


def assist_hip_angle_tracking_exp(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward assist-to-hip angle agreement with a peak value of one at zero error."""
    if std <= 0.0:
        raise ValueError("Angle tracking standard deviation must be greater than zero")
    robot: Articulation = env.scene[hip_cfg.name]
    error = (
        robot.data.joint_pos[:, assist_cfg.joint_ids]
        - robot.data.joint_pos[:, hip_cfg.joint_ids]
    )
    return torch.exp(-torch.square(error).mean(dim=-1) / std**2)


def assist_hip_velocity_error_l2(
    env: ManagerBasedRLEnv,
    hip_cfg: SceneEntityCfg,
    assist_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize continuous assist-to-hip joint velocity mismatch."""
    robot: Articulation = env.scene[hip_cfg.name]
    error = (
        robot.data.joint_vel[:, assist_cfg.joint_ids]
        - robot.data.joint_vel[:, hip_cfg.joint_ids]
    )
    return torch.square(error).mean(dim=-1)


class CompletedPulsePeakReward(ManagerTermBase):
    """Give the onset decision a per-pulse target, including zero-output pulses.

    Target uses privileged support only for training. It is an assistance
    heuristic, not a human torque estimate. Reward is dt-normalized because
    RewardManager multiplies terms by step_dt.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.acc = torch.zeros(env.num_envs, 3, device=env.device)
        # direction, accumulated eligible weight, weighted desired peak
        self.contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.age = torch.zeros(env.num_envs, 2, device=env.device)
        self.stats = torch.zeros(env.num_envs, 12, device=env.device)
        # starts, zero starts, sum peaks, completions, supported completions,
        # actual peak sum, duration sum, eligible seconds, release counts (4 causes)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.acc[ids] = 0
        self.contact[ids] = False
        self.age[ids] = 0
        self.stats[ids] = 0

    def __call__(self, env, hip_cfg, sensor_cfg, action_name="assist_torque", torque_limit=10.):
        action = env.action_manager.get_term(action_name)
        pulse = action._pulse
        starts, finishes = pulse.started, pulse.finished
        self.acc[starts] = 0
        self.acc[starts, 0] = pulse.state[starts, 1]
        self.stats[:, 0] += starts
        self.stats[:, 1] += starts & (pulse.state[:, 2] <= .1)
        self.stats[:, 2] += torch.where(starts, pulse.state[:, 2], 0.)
        robot = env.scene[hip_cfg.name]
        velocity = robot.data.joint_vel[:, hip_cfg.joint_ids]
        q = robot.data.joint_pos[:, hip_cfg.joint_ids]
        neutral = robot.data.default_joint_pos[:, hip_cfg.joint_ids]
        posture = ((neutral-q-.10)/.40).clamp(0., 1.)
        posture = posture.square()*(3.-2.*posture)
        force = env.scene[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
        contact, age, support = update_support(force, self.contact, self.age, env.step_dt,
                                                10., 20., 200., .05)
        self.contact[:] = contact
        self.age[:] = age
        # Evaluate only the selected flexing/elevated leg against its loaded partner.
        side = (self.acc[:, 0] > 0).long().unsqueeze(-1)
        moving_or_elevated = torch.maximum((-velocity/.25).clamp(0., 1.), posture)
        not_lowering = 1. - (velocity/.25).clamp(0., 1.)
        eligibility = ((1.-support)*support.flip(-1)*moving_or_elevated*not_lowering).gather(1, side).squeeze(-1)
        active = (pulse.state[:, 1] != 0) | finishes
        weight = eligibility * active
        target = (torque_limit*(.60+.40*posture)).gather(1, side).squeeze(-1)
        self.acc[:, 1] += weight*env.step_dt
        self.acc[:, 2] += weight*target*env.step_dt
        valid = finishes & (self.acc[:, 1] >= .05)
        desired_peak = self.acc[:, 2]/self.acc[:, 1].clamp_min(1e-6)
        cost = torch.where(valid, ((pulse.completed_peak-desired_peak)/torque_limit).square(), 0.)
        self.stats[:, 3] += finishes
        self.stats[:, 4] += valid
        self.stats[:, 5] += torch.where(finishes, pulse.completed_actual_peak, 0.)
        self.stats[:, 6] += torch.where(finishes, pulse.completed_duration, 0.)
        self.stats[:, 7] += torch.where(finishes, self.acc[:, 1], 0.)
        for reason in range(1, 5):
            self.stats[:, 7 + reason] += finishes & (pulse.completed_release_reason == reason)
        totals = self.stats.sum(dim=0)
        log = env.extras.setdefault("log", {})
        log["Pulse/starts_per_env"] = self.stats[:, 0].mean()
        log["Pulse/zero_peak_fraction"] = totals[1]/totals[0].clamp_min(1.)
        log["Pulse/mean_onset_peak_nm"] = totals[2]/totals[0].clamp_min(1.)
        log["Pulse/supported_completion_fraction"] = totals[4]/totals[3].clamp_min(1.)
        log["Pulse/mean_actual_peak_nm"] = totals[5]/totals[3].clamp_min(1.)
        log["Pulse/mean_duration_s"] = totals[6]/totals[3].clamp_min(1.)
        log["Pulse/mean_eligible_support_s"] = totals[7]/totals[3].clamp_min(1.)
        for reason, name in enumerate(("lowering", "opposite_lift", "quiet", "timeout"), 1):
            log[f"Pulse/release_{name}_fraction"] = totals[7+reason]/totals[3].clamp_min(1.)
        return cost/env.step_dt


def assist_action_out_of_bounds(env, action_name="assist_torque"):
    """Training surrogate: discourage Gaussian means escaping the usable action range."""
    raw = env.action_manager.get_term(action_name).raw_actions
    return (raw.abs()-1.).clamp_min(0.).square().mean(dim=-1)

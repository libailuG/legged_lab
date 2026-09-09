"""Tensor helpers for deployable paired control and training-only reference design.

The control helpers accept only actor output and local exoskeleton motion.
Support information is used exclusively by the reward helpers below.
"""

import torch


def smooth_gate(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if not 0.0 <= low < high:
        raise ValueError("Expected 0 <= low < high")
    phase = ((value - low) / (high - low)).clamp(0.0, 1.0)
    return phase.square() * (3.0 - 2.0 * phase)


def shared_motion_gate(velocity: torch.Tensor, deadzone: float, full_speed: float) -> torch.Tensor:
    """One gate for BOTH actuators; a stationary support leg must not disable its motor."""
    return smooth_gate(velocity.abs().amax(dim=-1, keepdim=True), deadzone, full_speed)


def paired_torques(scalar: torch.Tensor) -> torch.Tensor:
    """A positive scalar presses left/lifts right; negative lifts left/presses right."""
    return torch.cat((scalar, -scalar), dim=-1)


def slew_paired_scalar(
    target: torch.Tensor, previous: torch.Tensor, torque_limit: float, rate_limit: float, dt: float
) -> torch.Tensor:
    """Rate-limit one scalar, explicitly visiting zero before changing pair direction."""
    if torque_limit <= 0.0 or rate_limit <= 0.0 or dt <= 0.0:
        raise ValueError("Torque limit, rate limit and dt must be positive")
    target = target.clamp(-torque_limit, torque_limit)
    reversing = target * previous < 0.0
    effective_target = torch.where(reversing, torch.zeros_like(target), target)
    return (previous + (effective_target - previous).clamp(-rate_limit * dt, rate_limit * dt)).clamp(
        -torque_limit, torque_limit
    )


def update_support(
    force_z: torch.Tensor,
    was_contact: torch.Tensor,
    age: torch.Tensor,
    dt: float,
    force_off: float,
    force_on: float,
    force_full: float,
    rise_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training only: hysteresis plus load and contact-age ramps; immediate release on unloading."""
    if not 0 <= force_off < force_on < force_full or dt <= 0 or rise_time <= 0:
        raise ValueError("Invalid support thresholds or timing")
    contact = (force_z >= force_on) | (was_contact & (force_z > force_off))
    age = torch.where(contact, age + dt, torch.zeros_like(age))
    support = smooth_gate(force_z, force_off, force_full) * smooth_gate(age, 0.0, rise_time)
    return contact, age, support * contact.to(force_z.dtype)


def support_aware_reference(
    hip_velocity: torch.Tensor,
    filtered_acceleration: torch.Tensor,
    support: torch.Tensor,
    torque_limit: float,
    velocity_gain: float,
    acceleration_gain: float,
    dynamics_scale: float,
    lift_speed_deadzone: float,
    lift_speed_full: float,
) -> torch.Tensor:
    """Training only: lift the flexing/unloading leg against its weight-bearing partner.

    Positive hip velocity (leg lowering/extension) never creates lift demand.
    Both unsupported, both fully supported, or no flexion => zero reference.
    """
    if torque_limit <= 0 or dynamics_scale <= 0:
        raise ValueError("Torque limit and dynamics scale must be positive")
    signal = -(velocity_gain * hip_velocity + acceleration_gain * filtered_acceleration)
    demand = torch.tanh(signal.clamp_min(0.0) / dynamics_scale)
    demand *= smooth_gate(-hip_velocity, lift_speed_deadzone, lift_speed_full)
    eligible = demand * (1.0 - support) * support.flip(dims=(-1,))
    scalar = torque_limit * (eligible[:, 1:2] - eligible[:, 0:1])
    return paired_torques(scalar)


def unsupported_press_cost(torque: torch.Tensor, support: torch.Tensor, torque_limit: float) -> torch.Tensor:
    """Penalize positive (press) effort on a leg that is not bearing weight."""
    return ((torque.clamp_min(0.0) / torque_limit).square() * (1.0 - support)).mean(dim=-1)


def bilateral_rest_cost(
    torque: torch.Tensor, hip_velocity: torch.Tensor, filtered_acceleration: torch.Tensor,
    torque_limit: float, velocity_scale: float, acceleration_scale: float,
) -> torch.Tensor:
    """Only strong when BOTH hips are quiet; never penalize one quiet support leg in isolation."""
    motion = (hip_velocity / velocity_scale).square() + (filtered_acceleration / acceleration_scale).square()
    rest_weight = 1.0 / (1.0 + motion.amax(dim=-1))
    return (torque / torque_limit).square().mean(dim=-1) * rest_weight

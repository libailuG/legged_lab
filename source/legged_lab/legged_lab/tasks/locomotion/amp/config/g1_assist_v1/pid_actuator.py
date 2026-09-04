"""Explicit joint PID actuator used only by the G1 assist AMP v1 task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class IdealPIDActuator(IdealPDActuator):
    r"""Explicit PID actuator with torque-bounded integral and anti-windup.

    The integral torque is updated once per physics step. Integration is
    rejected when actuator saturation would be driven farther in the same
    direction.
    """

    cfg: IdealPIDActuatorCfg

    def __init__(self, cfg: IdealPIDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.integration_dt <= 0.0:
            raise ValueError(f"integration_dt must be positive, got {cfg.integration_dt}")

        self.integral_gain = self._parse_joint_parameter(cfg.integral_gain, 0.0)
        self.integral_effort_limit = self._parse_joint_parameter(
            cfg.integral_effort_limit, 0.0
        )
        self._integral_effort = torch.zeros_like(self.computed_effort)

    def reset(self, env_ids: Sequence[int]):
        self._integral_effort[env_ids] = 0.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        candidate_integral_effort = torch.clamp(
            self._integral_effort
            + self.integral_gain * error_pos * self.cfg.integration_dt,
            min=-self.integral_effort_limit,
            max=self.integral_effort_limit,
        )
        candidate_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + candidate_integral_effort
            + control_action.joint_efforts
        )
        candidate_applied_effort = self._clip_effort(candidate_effort)

        saturated = candidate_effort != candidate_applied_effort
        pushes_further_into_saturation = error_pos * candidate_effort > 0.0
        accept_integral = ~(saturated & pushes_further_into_saturation)
        self._integral_effort[:] = torch.where(
            accept_integral, candidate_integral_effort, self._integral_effort
        )

        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + self._integral_effort
            + control_action.joint_efforts
        )
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class IdealPIDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for :class:`IdealPIDActuator`."""

    class_type: type = IdealPIDActuator
    integral_gain: float | dict[str, float] = 0.0
    """Integral gain in N m / (rad s)."""
    integral_effort_limit: float | dict[str, float] = 10.0
    """Absolute integral-torque limit in N m."""
    integration_dt: float = 0.001
    """PID update period in seconds; must match the physics time step."""

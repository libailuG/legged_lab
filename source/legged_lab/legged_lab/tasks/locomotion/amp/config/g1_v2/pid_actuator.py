"""Explicit joint PID actuator used only by the G1 AMP v2 task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class IdealPIDActuator(IdealPDActuator):
    r"""Explicit PID actuator with bounded integral and conditional anti-windup.

    The integral state is updated once per physics step. When the requested
    effort saturates and the position error would push it farther into
    saturation, that step's integration is rejected.
    """

    cfg: IdealPIDActuatorCfg

    def __init__(self, cfg: IdealPIDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.integration_dt <= 0.0:
            raise ValueError(f"integration_dt must be positive, got {cfg.integration_dt}")

        self.integral_gain = self._parse_joint_parameter(cfg.integral_gain, 0.0)
        self.integral_error_limit = self._parse_joint_parameter(cfg.integral_error_limit, 0.0)
        self._integral_error = torch.zeros_like(self.computed_effort)

    def reset(self, env_ids: Sequence[int]):
        self._integral_error[env_ids] = 0.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        candidate_integral = torch.clamp(
            self._integral_error + error_pos * self.cfg.integration_dt,
            min=-self.integral_error_limit,
            max=self.integral_error_limit,
        )
        candidate_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + self.integral_gain * candidate_integral
            + control_action.joint_efforts
        )
        candidate_applied_effort = self._clip_effort(candidate_effort)

        saturated = candidate_effort != candidate_applied_effort
        pushes_further_into_saturation = error_pos * candidate_effort > 0.0
        accept_integral = ~(saturated & pushes_further_into_saturation)
        self._integral_error[:] = torch.where(
            accept_integral, candidate_integral, self._integral_error
        )

        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + self.integral_gain * self._integral_error
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
    integral_error_limit: float | dict[str, float] = 0.5
    """Absolute integral-state limit in rad s."""
    integration_dt: float = 0.001
    """PID update period in seconds; must match the physics time step."""

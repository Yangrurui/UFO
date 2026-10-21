# Adapted from XLab xlab/actuators/mjlab.py; runtime adapter omitted; see XLAB_LICENSE.
"""MJLab-native four-quadrant torque-speed envelope actuator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg

from .torque_speed_envelope import (
    clip_effort_by_envelope,
    clip_effort_by_symmetric_envelope,
    piecewise_linear_slope_changes,
    validate_effort_limit,
    validate_torque_speed_envelope,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class TorqueSpeedEnvelopeActuatorCfg(IdealPdActuatorCfg):
    """Ideal PD control clipped by four measured torque-speed curves."""

    effort_limit: float = field()
    """Initial effort ceiling intersected with every curve; runtime DR may replace it."""

    velocity_limit: float
    """Metadata speed limit covering every envelope breakpoint, in rad/s."""

    forward_motoring_velocity: tuple[float, ...]
    forward_motoring_effort: tuple[float, ...]
    reverse_motoring_velocity: tuple[float, ...]
    reverse_motoring_effort: tuple[float, ...]
    forward_braking_velocity: tuple[float, ...]
    forward_braking_effort: tuple[float, ...]
    reverse_braking_velocity: tuple[float, ...]
    reverse_braking_effort: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_effort_limit(self.effort_limit)
        validate_torque_speed_envelope(
            velocity_limit=self.velocity_limit,
            forward_motoring_velocity=self.forward_motoring_velocity,
            forward_motoring_effort=self.forward_motoring_effort,
            reverse_motoring_velocity=self.reverse_motoring_velocity,
            reverse_motoring_effort=self.reverse_motoring_effort,
            forward_braking_velocity=self.forward_braking_velocity,
            forward_braking_effort=self.forward_braking_effort,
            reverse_braking_velocity=self.reverse_braking_velocity,
            reverse_braking_effort=self.reverse_braking_effort,
        )

    def build(
        self,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> TorqueSpeedEnvelopeActuator:
        return TorqueSpeedEnvelopeActuator(self, entity, target_ids, target_names)


class TorqueSpeedEnvelopeActuator(IdealPdActuator[TorqueSpeedEnvelopeActuatorCfg]):
    """MJLab explicit PD actuator with measured motoring and braking limits."""

    def __init__(
        self,
        cfg: TorqueSpeedEnvelopeActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self.forward_motoring_velocity: torch.Tensor | None = None
        self.forward_motoring_effort: torch.Tensor | None = None
        self.reverse_motoring_velocity: torch.Tensor | None = None
        self.reverse_motoring_effort: torch.Tensor | None = None
        self.forward_braking_velocity: torch.Tensor | None = None
        self.forward_braking_effort: torch.Tensor | None = None
        self.reverse_braking_velocity: torch.Tensor | None = None
        self.reverse_braking_effort: torch.Tensor | None = None
        self._symmetric_envelope = False
        self._motoring_slope_changes: torch.Tensor | None = None
        self._braking_slope_changes: torch.Tensor | None = None
        self._joint_vel: torch.Tensor | None = None
        self.raw_torque: torch.Tensor | None = None
        self.clipped_torque: torch.Tensor | None = None

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        first_new_motor = len(self._mjs_actuators)
        super().edit_spec(spec, target_names)
        # Native safety caps must admit every legal envelope output after effort
        # randomization. The live force_limit tensor supplies the current ceiling;
        # fixing these native caps at its initial value silently defeats increases.
        native_ceiling = max(
            self.cfg.effort_limit,
            *self.cfg.forward_motoring_effort,
            *self.cfg.reverse_motoring_effort,
            *self.cfg.forward_braking_effort,
            *self.cfg.reverse_braking_effort,
        )
        if native_ceiling == 0.0:
            # A zero ceiling and four zero curves permanently disable this motor.
            # MuJoCo rejects enabled [0, 0] ranges; zero its fixed gain instead.
            # Leave the joint sum limit intact for any other motor on that joint.
            for motor in self._mjs_actuators[first_new_motor:]:
                motor.gainprm[0] = 0.0
                motor.ctrllimited = False
                motor.forcelimited = False
            return
        native_range = (-native_ceiling, native_ceiling)
        for motor in self._mjs_actuators[first_new_motor:]:
            motor.ctrlrange[:] = native_range
            motor.forcerange[:] = native_range
        for name in target_names:
            joint = spec.joint(name)
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.actfrcrange[:] = native_range

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        assert self.force_limit is not None
        tensor_options = {
            "device": self.force_limit.device,
            "dtype": self.force_limit.dtype,
        }
        self.forward_motoring_velocity = torch.tensor(
            self.cfg.forward_motoring_velocity, **tensor_options
        )
        self.forward_motoring_effort = torch.tensor(
            self.cfg.forward_motoring_effort, **tensor_options
        )
        self.reverse_motoring_velocity = torch.tensor(
            self.cfg.reverse_motoring_velocity, **tensor_options
        )
        self.reverse_motoring_effort = torch.tensor(
            self.cfg.reverse_motoring_effort, **tensor_options
        )
        self.forward_braking_velocity = torch.tensor(
            self.cfg.forward_braking_velocity, **tensor_options
        )
        self.forward_braking_effort = torch.tensor(
            self.cfg.forward_braking_effort, **tensor_options
        )
        self.reverse_braking_velocity = torch.tensor(
            self.cfg.reverse_braking_velocity, **tensor_options
        )
        self.reverse_braking_effort = torch.tensor(
            self.cfg.reverse_braking_effort, **tensor_options
        )
        self._symmetric_envelope = (
            self.cfg.forward_motoring_velocity
            == self.cfg.reverse_motoring_velocity
            and self.cfg.forward_motoring_effort
            == self.cfg.reverse_motoring_effort
            and self.cfg.forward_braking_velocity
            == self.cfg.reverse_braking_velocity
            and self.cfg.forward_braking_effort
            == self.cfg.reverse_braking_effort
        )
        if self._symmetric_envelope:
            # The reference four-curve path floors interval widths at 1e-6.
            # Reverse hinges use true slopes, so narrower native-dtype intervals
            # must retain that path instead of changing applied torque.
            widths = torch.cat((
                torch.diff(self.forward_motoring_velocity),
                torch.diff(self.forward_braking_velocity),
            ))
            self._symmetric_envelope = bool(torch.all(widths >= 1.0e-6))
        if self._symmetric_envelope:
            self._motoring_slope_changes = piecewise_linear_slope_changes(
                self.forward_motoring_velocity,
                self.forward_motoring_effort,
            )
            self._braking_slope_changes = piecewise_linear_slope_changes(
                self.forward_braking_velocity,
                self.forward_braking_effort,
            )
        self._joint_vel = torch.zeros_like(self.force_limit)

    def enable_torque_recording(self) -> None:
        """Record the actual delayed, randomized PD demand for substep rewards."""
        assert self.force_limit is not None
        self.raw_torque = torch.zeros_like(self.force_limit)
        self.clipped_torque = torch.zeros_like(self.force_limit)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        self._joint_vel = cmd.vel
        return super().compute(cmd)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        assert self.forward_motoring_velocity is not None
        assert self.forward_motoring_effort is not None
        assert self.reverse_motoring_velocity is not None
        assert self.reverse_motoring_effort is not None
        assert self.forward_braking_velocity is not None
        assert self.forward_braking_effort is not None
        assert self.reverse_braking_velocity is not None
        assert self.reverse_braking_effort is not None
        assert self.force_limit is not None
        assert self._joint_vel is not None
        if self._symmetric_envelope:
            assert self._motoring_slope_changes is not None
            assert self._braking_slope_changes is not None
            clipped = clip_effort_by_symmetric_envelope(
                effort,
                self._joint_vel,
                self.forward_motoring_velocity,
                self.forward_motoring_effort[-1],
                self._motoring_slope_changes,
                self.forward_braking_velocity,
                self.forward_braking_effort[-1],
                self._braking_slope_changes,
                self.force_limit,
            )
        else:
            clipped = clip_effort_by_envelope(
                effort,
                self._joint_vel,
                self.forward_motoring_velocity,
                self.forward_motoring_effort,
                self.reverse_motoring_velocity,
                self.reverse_motoring_effort,
                self.forward_braking_velocity,
                self.forward_braking_effort,
                self.reverse_braking_velocity,
                self.reverse_braking_effort,
                self.force_limit,
            )
        if self.raw_torque is not None:
            self.raw_torque.copy_(effort)
            self.clipped_torque.copy_(clipped)
        return clipped

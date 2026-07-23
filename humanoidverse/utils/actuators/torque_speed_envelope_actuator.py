"""PD actuator clipped by a measured four-quadrant torque-speed envelope.

Ported from InstinctMJ (feature/jump). It subclasses MJLab's IdealPdActuator
and only overrides effort clipping, so it drops into the same actuator slot as
DcMotorActuatorCfg used elsewhere in UFO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

from humanoidverse.utils.actuators.torque_speed_envelope import (
    clip_effort_by_envelope,
    validate_envelope,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity

TorqueSpeedEnvelopeCfgT = TypeVar("TorqueSpeedEnvelopeCfgT", bound="TorqueSpeedEnvelopeActuatorCfg")


@dataclass(kw_only=True)
class TorqueSpeedEnvelopeActuatorCfg(IdealPdActuatorCfg):
    """Ideal PD actuator with separate motoring and braking torque-speed limits."""

    velocity_limit: float
    """Metadata velocity limit, normally the largest zero-torque speed in the envelope."""

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
        validate_envelope("forward_motoring", self.forward_motoring_velocity, self.forward_motoring_effort)
        validate_envelope("reverse_motoring", self.reverse_motoring_velocity, self.reverse_motoring_effort)
        validate_envelope("forward_braking", self.forward_braking_velocity, self.forward_braking_effort)
        validate_envelope("reverse_braking", self.reverse_braking_velocity, self.reverse_braking_effort)
        if self.velocity_limit < max(
            self.forward_motoring_velocity[-1],
            self.reverse_motoring_velocity[-1],
            self.forward_braking_velocity[-1],
            self.reverse_braking_velocity[-1],
        ):
            raise ValueError("velocity_limit must cover the full torque-speed envelope.")

    def build(
        self, entity: "Entity", target_ids: list[int], target_names: list[str]
    ) -> "TorqueSpeedEnvelopeActuator":
        return TorqueSpeedEnvelopeActuator(self, entity, target_ids, target_names)


class TorqueSpeedEnvelopeActuator(
    IdealPdActuator[TorqueSpeedEnvelopeCfgT],
    Generic[TorqueSpeedEnvelopeCfgT],
):
    """PD actuator whose output is clipped by measured motoring/braking envelopes."""

    def __init__(
        self,
        cfg: TorqueSpeedEnvelopeCfgT,
        entity: "Entity",
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
        self._joint_vel: torch.Tensor | None = None
        self._real_effort_breakpoints: dict[str, torch.Tensor] | None = None
        self._flat_effort: float | None = None

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        self.forward_motoring_velocity = torch.tensor(self.cfg.forward_motoring_velocity, dtype=torch.float, device=device)
        self.forward_motoring_effort = torch.tensor(self.cfg.forward_motoring_effort, dtype=torch.float, device=device)
        self.reverse_motoring_velocity = torch.tensor(self.cfg.reverse_motoring_velocity, dtype=torch.float, device=device)
        self.reverse_motoring_effort = torch.tensor(self.cfg.reverse_motoring_effort, dtype=torch.float, device=device)
        self.forward_braking_velocity = torch.tensor(self.cfg.forward_braking_velocity, dtype=torch.float, device=device)
        self.forward_braking_effort = torch.tensor(self.cfg.forward_braking_effort, dtype=torch.float, device=device)
        self.reverse_braking_velocity = torch.tensor(self.cfg.reverse_braking_velocity, dtype=torch.float, device=device)
        self.reverse_braking_effort = torch.tensor(self.cfg.reverse_braking_effort, dtype=torch.float, device=device)
        self._joint_vel = torch.zeros(data.nworld, len(self._target_names), device=device)

        # Baselines for the ideal->envelope curriculum blend. alpha=0 -> flat clamp at
        # effort_limit (no velocity roll-off); alpha=1 -> real measured breakpoints.
        self._real_effort_breakpoints = {
            "forward_motoring_effort": self.forward_motoring_effort.clone(),
            "reverse_motoring_effort": self.reverse_motoring_effort.clone(),
            "forward_braking_effort": self.forward_braking_effort.clone(),
            "reverse_braking_effort": self.reverse_braking_effort.clone(),
        }
        self._flat_effort = float(self.cfg.effort_limit)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self._joint_vel is not None
        self._joint_vel[:] = cmd.vel
        return super().compute(cmd)

    def set_envelope_blend(self, alpha: float) -> None:
        """Blend effort breakpoints between the flat clamp (alpha=0) and real envelope (alpha=1)."""
        assert self._real_effort_breakpoints is not None
        assert self._flat_effort is not None
        a = float(min(max(alpha, 0.0), 1.0))
        for name, real in self._real_effort_breakpoints.items():
            flat = torch.full_like(real, self._flat_effort)
            setattr(self, name, (1.0 - a) * flat + a * real)

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
        return clip_effort_by_envelope(
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

# Adapted from XLab xlab/actuators/torque_speed_envelope.py; see XLAB_LICENSE.
"""Engine-neutral four-quadrant torque-speed envelope math."""

from __future__ import annotations

import math
from itertools import pairwise
from numbers import Real

import torch


def piecewise_linear_limit(
    speed_abs: torch.Tensor,
    velocity: torch.Tensor,
    effort: torch.Tensor,
) -> torch.Tensor:
    """Interpolate one non-negative effort limit at absolute joint speed."""
    lower_index = torch.bucketize(speed_abs.contiguous(), velocity, right=True) - 1
    lower_index = torch.clamp(lower_index, min=0, max=velocity.numel() - 1)
    upper_index = torch.clamp(lower_index + 1, max=velocity.numel() - 1)

    lower_velocity = velocity[lower_index]
    upper_velocity = velocity[upper_index]
    lower_effort = effort[lower_index]
    upper_effort = effort[upper_index]

    denominator = torch.clamp(upper_velocity - lower_velocity, min=1.0e-6)
    ratio = torch.clamp(
        (speed_abs - lower_velocity) / denominator,
        min=0.0,
        max=1.0,
    )
    return lower_effort + ratio * (upper_effort - lower_effort)


def piecewise_linear_slope_changes(
    velocity: torch.Tensor,
    effort: torch.Tensor,
) -> torch.Tensor:
    """Precompute reverse-hinge slope changes for a clamped linear curve."""
    interval_slopes = (effort[1:] - effort[:-1]) / (
        velocity[1:] - velocity[:-1]
    )
    return torch.cat(
        (
            interval_slopes[1:] - interval_slopes[:-1],
            -interval_slopes[-1:],
        )
    )


def piecewise_linear_limit_from_slope_changes(
    speed_abs: torch.Tensor,
    velocity: torch.Tensor,
    effort_at_max_speed: torch.Tensor,
    slope_changes: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a clamped curve without index searches or tail cancellation."""
    hinges = torch.relu(velocity[1:] - speed_abs.unsqueeze(-1))
    return effort_at_max_speed + torch.sum(hinges * slope_changes, dim=-1)


def clip_effort_by_symmetric_envelope(
    effort: torch.Tensor,
    joint_vel: torch.Tensor,
    motoring_velocity: torch.Tensor,
    motoring_effort_at_max_speed: torch.Tensor,
    motoring_slope_changes: torch.Tensor,
    braking_velocity: torch.Tensor,
    braking_effort_at_max_speed: torch.Tensor,
    braking_slope_changes: torch.Tensor,
    force_limit: torch.Tensor,
) -> torch.Tensor:
    """Clip effort when forward and reverse curves are exactly symmetric."""
    speed_abs = torch.abs(joint_vel)
    motoring_limit = piecewise_linear_limit_from_slope_changes(
        speed_abs,
        motoring_velocity,
        motoring_effort_at_max_speed,
        motoring_slope_changes,
    )
    braking_limit = piecewise_linear_limit_from_slope_changes(
        speed_abs,
        braking_velocity,
        braking_effort_at_max_speed,
        braking_slope_changes,
    )

    maximum_effort = torch.where(joint_vel < 0.0, braking_limit, motoring_limit)
    minimum_effort = -torch.where(joint_vel > 0.0, braking_limit, motoring_limit)
    maximum_effort = torch.minimum(maximum_effort, force_limit)
    minimum_effort = torch.maximum(minimum_effort, -force_limit)
    return torch.clamp(effort, min=minimum_effort, max=maximum_effort)


def clip_effort_by_envelope(
    effort: torch.Tensor,
    joint_vel: torch.Tensor,
    forward_motoring_velocity: torch.Tensor,
    forward_motoring_effort: torch.Tensor,
    reverse_motoring_velocity: torch.Tensor,
    reverse_motoring_effort: torch.Tensor,
    forward_braking_velocity: torch.Tensor,
    forward_braking_effort: torch.Tensor,
    reverse_braking_velocity: torch.Tensor,
    reverse_braking_effort: torch.Tensor,
    force_limit: torch.Tensor,
) -> torch.Tensor:
    """Clip requested effort by motoring/braking limits in all four quadrants."""
    speed_abs = torch.abs(joint_vel)
    forward_motoring_limit = piecewise_linear_limit(
        speed_abs,
        forward_motoring_velocity,
        forward_motoring_effort,
    )
    reverse_motoring_limit = piecewise_linear_limit(
        speed_abs,
        reverse_motoring_velocity,
        reverse_motoring_effort,
    )
    forward_braking_limit = piecewise_linear_limit(
        speed_abs,
        forward_braking_velocity,
        forward_braking_effort,
    )
    reverse_braking_limit = piecewise_linear_limit(
        speed_abs,
        reverse_braking_velocity,
        reverse_braking_effort,
    )

    maximum_effort = torch.where(
        joint_vel < 0.0,
        reverse_braking_limit,
        forward_motoring_limit,
    )
    minimum_effort = -torch.where(
        joint_vel > 0.0,
        forward_braking_limit,
        reverse_motoring_limit,
    )
    maximum_effort = torch.minimum(maximum_effort, force_limit)
    minimum_effort = torch.maximum(minimum_effort, -force_limit)
    return torch.clamp(effort, min=minimum_effort, max=maximum_effort)


def validate_envelope(
    name: str,
    velocity: tuple[float, ...],
    effort: tuple[float, ...],
) -> None:
    """Validate one speed/effort breakpoint curve before native construction."""
    if len(velocity) != len(effort):
        raise ValueError(
            f"{name}: velocity and effort envelopes must have the same length."
        )
    if len(velocity) < 2:
        raise ValueError(f"{name}: envelope must contain at least two points.")
    if any(
        isinstance(value, bool) or not isinstance(value, Real) for value in velocity
    ):
        raise TypeError(f"{name}: velocity envelope values must be real numbers.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in effort):
        raise TypeError(f"{name}: effort envelope values must be real numbers.")
    if any(not math.isfinite(float(value)) for value in (*velocity, *effort)):
        raise ValueError(f"{name}: envelope values must be finite.")
    if velocity[0] != 0.0:
        raise ValueError(f"{name}: first velocity point must be 0.0.")
    if any(high <= low for low, high in pairwise(velocity)):
        raise ValueError(f"{name}: velocity envelope must be strictly increasing.")
    if any(value < 0.0 for value in effort):
        raise ValueError(f"{name}: effort envelope must be non-negative.")


def validate_effort_limit(
    effort_limit: float,
    *,
    name: str = "effort_limit",
) -> None:
    """Validate one homogeneous actuator group's static effort ceiling."""
    if isinstance(effort_limit, bool) or not isinstance(effort_limit, Real):
        raise TypeError(f"{name} must be a scalar real number.")
    if not math.isfinite(float(effort_limit)):
        raise ValueError(f"{name} must be finite.")
    if effort_limit < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def validate_torque_speed_envelope(
    *,
    velocity_limit: float,
    forward_motoring_velocity: tuple[float, ...],
    forward_motoring_effort: tuple[float, ...],
    reverse_motoring_velocity: tuple[float, ...],
    reverse_motoring_effort: tuple[float, ...],
    forward_braking_velocity: tuple[float, ...],
    forward_braking_effort: tuple[float, ...],
    reverse_braking_velocity: tuple[float, ...],
    reverse_braking_effort: tuple[float, ...],
) -> None:
    """Validate all four curves and their declared metadata velocity limit."""
    curves = (
        (
            "forward_motoring",
            forward_motoring_velocity,
            forward_motoring_effort,
        ),
        (
            "reverse_motoring",
            reverse_motoring_velocity,
            reverse_motoring_effort,
        ),
        (
            "forward_braking",
            forward_braking_velocity,
            forward_braking_effort,
        ),
        (
            "reverse_braking",
            reverse_braking_velocity,
            reverse_braking_effort,
        ),
    )
    for name, velocity, effort in curves:
        validate_envelope(name, velocity, effort)

    if isinstance(velocity_limit, bool) or not isinstance(velocity_limit, Real):
        raise TypeError("velocity_limit must be a real number.")
    if not math.isfinite(float(velocity_limit)):
        raise ValueError("velocity_limit must be finite.")
    if velocity_limit < max(velocity[-1] for _, velocity, _ in curves):
        raise ValueError("velocity_limit must cover the full torque-speed envelope.")


__all__ = [
    "clip_effort_by_envelope",
    "clip_effort_by_symmetric_envelope",
    "piecewise_linear_limit",
    "piecewise_linear_limit_from_slope_changes",
    "piecewise_linear_slope_changes",
    "validate_effort_limit",
    "validate_envelope",
    "validate_torque_speed_envelope",
]

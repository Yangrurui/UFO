"""Four-quadrant torque-speed envelope math (ported from InstinctMJ feature/jump).

The envelope clips an already-computed PD effort by a measured, per-quadrant
torque-speed limit.  The four quadrants are:

  - forward_motoring : joint_vel > 0, torque > 0  (driving in + direction)
  - reverse_motoring : joint_vel < 0, torque < 0  (driving in - direction)
  - forward_braking  : joint_vel > 0, torque < 0  (braking a + motion)
  - reverse_braking  : joint_vel < 0, torque > 0  (braking a - motion)

Each quadrant is a piecewise-linear curve of ``|speed|`` breakpoints -> max
torque magnitude (plateau, then linear roll-off to zero at the no-load speed).
"""

from __future__ import annotations

import torch


def piecewise_linear_limit(
    speed_abs: torch.Tensor,
    velocity: torch.Tensor,
    effort: torch.Tensor,
) -> torch.Tensor:
    idx_lo = torch.bucketize(speed_abs.contiguous(), velocity, right=True) - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=velocity.numel() - 1)
    idx_hi = torch.clamp(idx_lo + 1, max=velocity.numel() - 1)

    vel_lo = velocity[idx_lo]
    vel_hi = velocity[idx_hi]
    effort_lo = effort[idx_lo]
    effort_hi = effort[idx_hi]

    denom = torch.clamp(vel_hi - vel_lo, min=1.0e-6)
    ratio = torch.clamp((speed_abs - vel_lo) / denom, min=0.0, max=1.0)
    return effort_lo + ratio * (effort_hi - effort_lo)


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
    speed_abs = torch.abs(joint_vel)
    forward_motoring_limit = piecewise_linear_limit(
        speed_abs, forward_motoring_velocity, forward_motoring_effort
    )
    reverse_motoring_limit = piecewise_linear_limit(
        speed_abs, reverse_motoring_velocity, reverse_motoring_effort
    )
    forward_braking_limit = piecewise_linear_limit(
        speed_abs, forward_braking_velocity, forward_braking_effort
    )
    reverse_braking_limit = piecewise_linear_limit(
        speed_abs, reverse_braking_velocity, reverse_braking_effort
    )

    max_effort = torch.where(joint_vel < 0.0, reverse_braking_limit, forward_motoring_limit)
    min_effort = -torch.where(joint_vel > 0.0, forward_braking_limit, reverse_motoring_limit)
    max_effort = torch.minimum(max_effort, force_limit)
    min_effort = torch.maximum(min_effort, -force_limit)
    return torch.clamp(effort, min=min_effort, max=max_effort)


def validate_envelope(name: str, velocity: tuple[float, ...], effort: tuple[float, ...]) -> None:
    if len(velocity) != len(effort):
        raise ValueError(f"{name}: velocity and effort envelopes must have the same length.")
    if len(velocity) < 2:
        raise ValueError(f"{name}: envelope must contain at least two points.")
    if velocity[0] != 0.0:
        raise ValueError(f"{name}: first velocity point must be 0.0.")
    if any(v1 <= v0 for v0, v1 in zip(velocity, velocity[1:])):
        raise ValueError(f"{name}: velocity envelope must be strictly increasing.")
    if any(t < 0.0 for t in effort):
        raise ValueError(f"{name}: effort envelope must be non-negative.")

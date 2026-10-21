"""Adam dynamics and bounded reference-reset randomization.

The event uses nominal mass/inertia, so repeated calls never compound. Reset
perturbations return new tensors and use XYZW quaternions throughout.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from mjlab.envs.mdp.dr import body_mass
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields


@requires_model_fields("body_mass", "body_inertia", recompute=body_mass.recompute)
def randomize_mass_and_inertia(
    env,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float] = (0.95, 1.05),
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Scale each selected body's nominal mass and inertia by one uniform draw.

    Draws are independent across environments and bodies. COM and principal-axis
    orientation are untouched. Register this event with ``mode="startup"``.
    """
    low, high = map(float, ranges)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("Mass scale range must be finite, positive and ordered")
    asset_cfg = asset_cfg if asset_cfg is not None else SceneEntityCfg("robot")
    ids = (
        torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    )
    asset = env.scene[asset_cfg.name]
    body_ids = asset.indexing.body_ids[asset_cfg.body_ids].to(dtype=torch.long).flatten()
    if ids.numel() == 0 or body_ids.numel() == 0:
        return
    env_grid, body_grid = torch.meshgrid(ids, body_ids, indexing="ij")

    def nominal(name):
        values = env.sim.get_default_field(name)
        if name in env.sim.per_world_default_fields:
            return values[env_grid, body_grid]
        return values[body_ids].unsqueeze(0).expand(len(ids), *values[body_ids].shape)

    mass, inertia = nominal("body_mass"), nominal("body_inertia")
    factors = torch.empty_like(mass).uniform_(low, high)
    new_mass, new_inertia = mass * factors, inertia * factors.unsqueeze(-1)
    env.sim.model.body_mass[env_grid, body_grid] = new_mass
    env.sim.model.body_inertia[env_grid, body_grid] = new_inertia


def _half_range(values: float | Sequence[float], size: int, like: torch.Tensor) -> torch.Tensor:
    values = (float(values),) * size if isinstance(values, (int, float)) else tuple(values)
    if len(values) != size or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"Expected {size} finite, nonnegative half ranges")
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def perturb_reference_state(
    root_states: torch.Tensor,
    dof_states: torch.Tensor,
    joint_limits: torch.Tensor,
    *,
    joint_position_half_range: float = 0.05,
    root_position_half_range: Sequence[float] = (0.025, 0.025, 0.005),
    root_rotation_half_range: Sequence[float] = (0.05, 0.05, 0.1),
    root_linear_velocity_half_range: Sequence[float] = (0.25, 0.25, 0.1),
    root_angular_velocity_half_range: Sequence[float] = (0.26, 0.26, 0.39),
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perturb a reference reset without mutating motion-library tensors.

    ``root_states`` is (N,13): xyz [m], XYZW quaternion, world linear [m/s]
    and angular [rad/s] velocities. ``dof_states`` is (N,J,2): q [rad], qdot
    [rad/s]. ``joint_limits`` contains ordered hard limits (J,2).

    Uniform roll/pitch/yaw increments use Rz(yaw) Ry(pitch) Rx(roll), composed
    on the LEFT of the reference orientation (world-frame perturbation).
    Reference quaternions must be nonzero; outputs are normalized. Joint
    velocities are preserved and perturbed positions are clamped to hard limits.
    """
    if root_states.ndim != 2 or root_states.shape[-1] != 13:
        raise ValueError("root_states must have shape (N,13)")
    if dof_states.ndim != 3 or dof_states.shape[0] != root_states.shape[0] or dof_states.shape[-1] != 2:
        raise ValueError("dof_states must have shape (N,J,2)")
    if joint_limits.shape != (dof_states.shape[1], 2):
        raise ValueError("joint_limits must have shape (J,2)")
    if not root_states.is_floating_point() or not dof_states.is_floating_point():
        raise ValueError("Reference states must be floating point")
    if root_states.device != dof_states.device:
        raise ValueError("Reference states must share a device")
    half_ranges = {
        "position": _half_range(root_position_half_range, 3, root_states),
        "rotation": _half_range(root_rotation_half_range, 3, root_states),
        "linear": _half_range(root_linear_velocity_half_range, 3, root_states),
        "angular": _half_range(root_angular_velocity_half_range, 3, root_states),
    }
    joint_half = _half_range(joint_position_half_range, dof_states.shape[1], dof_states)
    root, dofs = root_states.clone(), dof_states.clone()

    def uniform_like(values, half):
        return (torch.rand(values.shape, dtype=values.dtype, device=values.device, generator=generator) * 2 - 1) * half

    root[:, :3] += uniform_like(root[:, :3], half_ranges["position"])
    root[:, 7:10] += uniform_like(root[:, 7:10], half_ranges["linear"])
    root[:, 10:13] += uniform_like(root[:, 10:13], half_ranges["angular"])
    rpy = uniform_like(root[:, :3], half_ranges["rotation"])
    sr, sp, sy = torch.sin(rpy / 2).unbind(-1)
    cr, cp, cy = torch.cos(rpy / 2).unbind(-1)
    delta = torch.stack((sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy,
                         cr * cp * sy - sr * sp * cy,
                         cr * cp * cy + sr * sp * sy), dim=-1)
    reference = root_states[:, 3:7]
    dv, dw = delta[:, :3], delta[:, 3:4]
    rv, rw = reference[:, :3], reference[:, 3:4]
    orientation = torch.cat((dw * rv + rw * dv + torch.linalg.cross(dv, rv),
                             dw * rw - (dv * rv).sum(-1, keepdim=True)), dim=-1)
    root[:, 3:7] = orientation / torch.linalg.vector_norm(orientation, dim=-1, keepdim=True).clamp_min(torch.finfo(root.dtype).tiny)
    limits = joint_limits.to(device=dofs.device, dtype=dofs.dtype)
    q = dofs[..., 0] + uniform_like(dofs[..., 0], joint_half)
    dofs[..., 0] = q.clamp(min=limits[:, 0], max=limits[:, 1])
    return root, dofs


__all__ = ["randomize_mass_and_inertia", "perturb_reference_state"]

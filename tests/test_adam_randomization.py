"""CPU checks for physical consistency and bounded reference reset noise."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.envs.mdp.dr import body_mass
from mjlab.managers import SceneEntityCfg
from scipy.spatial.transform import Rotation

from humanoidverse.agents.envs.adam_randomization import (
    perturb_reference_state,
    randomize_mass_and_inertia,
)


@pytest.mark.parametrize("per_world", [False, True])
def test_mass_inertia_share_density_factor_preserve_com_and_do_not_compound(per_world):
    mass = torch.tensor([1., 2., 4., 8.])
    inertia = mass[:, None] * torch.tensor([0.02, 0.03, 0.04])
    if per_world:
        mass = mass.repeat(3, 1) * torch.tensor([1., 1.5, 2.])[:, None]
        inertia = inertia.repeat(3, 1, 1) * torch.tensor([1., 1.5, 2.])[:, None, None]
    nominal = {"body_mass": mass.clone(), "body_inertia": inertia.clone()}
    original_mass = mass.clone() if per_world else mass.repeat(3, 1)
    original_inertia = inertia.clone() if per_world else inertia.repeat(3, 1, 1)
    com, axes = torch.randn(3, 4, 3), torch.randn(3, 4, 4)
    model = SimpleNamespace(body_mass=original_mass.clone(), body_inertia=original_inertia.clone(),
                            body_ipos=com.clone(), body_iquat=axes.clone())
    env = SimpleNamespace(num_envs=3, device="cpu",
        scene={"robot": SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([3, 1, 2])))},
        sim=SimpleNamespace(model=model, get_default_field=nominal.__getitem__,
                            per_world_default_fields=set(nominal) if per_world else set()))
    cfg = SceneEntityCfg("robot", body_ids=[0, 2])  # Select global bodies 3 and 2.
    ids = torch.tensor([0, 2])
    for _ in range(2):
        randomize_mass_and_inertia(env, ids, ranges=(1.05, 1.05), asset_cfg=cfg)
        expected = original_mass.clone()
        expected[ids[:, None], torch.tensor([3, 2])] *= 1.05
        torch.testing.assert_close(model.body_mass, expected)
        # Constant density scaling keeps the radii of gyration unchanged.
        torch.testing.assert_close(model.body_inertia / model.body_mass[..., None],
                                   original_inertia / original_mass[..., None])
        torch.testing.assert_close(model.body_ipos, com)
        torch.testing.assert_close(model.body_iquat, axes)
        torch.testing.assert_close(nominal["body_mass"], mass)
        torch.testing.assert_close(nominal["body_inertia"], inertia)
    assert {"body_mass", "body_inertia"}.issubset(randomize_mass_and_inertia.model_fields)
    assert randomize_mass_and_inertia.recompute == body_mass.recompute


def reference_batch(count=4096):
    roots = torch.zeros(count, 13, dtype=torch.float64)
    roots[:, :3] = torch.tensor([2., -1., 0.9])
    # A nontrivial reference exposes world/body frame composition mistakes.
    roots[:, 3:7] = torch.as_tensor(Rotation.from_euler("xyz", [0.4, -0.3, 1.2]).as_quat())
    roots[:, 7:13] = torch.tensor([0.5, -0.4, 0.1, -0.2, 0.3, 0.4])
    dofs = torch.zeros(count, 3, 2, dtype=torch.float64)
    dofs[..., 0] = torch.tensor([0., 0.005, 1.])
    dofs[..., 1] = torch.tensor([0.7, -0.2, 0.3])
    limits = torch.tensor([[-1., 1.], [0., 0.01], [0., 1.]], dtype=torch.float64)
    return roots, dofs, limits


def test_reset_noise_is_bounded_in_physical_coordinates_and_preserves_inputs():
    roots, dofs, limits = reference_batch()
    before_root, before_dof, before_limits = roots.clone(), dofs.clone(), limits.clone()
    new_roots, new_dofs = perturb_reference_state(roots, dofs, limits,
        generator=torch.Generator().manual_seed(4728))
    for section, bound in ((slice(0, 3), [.025, .025, .005]),
                           (slice(7, 10), [.25, .25, .1]),
                           (slice(10, 13), [.26, .26, .39])):
        difference = new_roots[:, section] - roots[:, section]
        half = torch.tensor(bound)
        assert torch.all(difference.abs() <= half + 1e-8)
        assert torch.all(difference.max(0).values > .9 * half)
        assert torch.all(difference.min(0).values < -.9 * half)
    # Independent scipy rotation algebra checks world-frame Euler bounds.
    delta = Rotation.from_quat(new_roots[:, 3:7].numpy()) * Rotation.from_quat(roots[:, 3:7].numpy()).inv()
    euler = delta.as_euler("xyz")
    bounds = np.asarray([.05, .05, .1])
    assert np.all(np.abs(euler) <= bounds + 1e-12)
    assert np.all(euler.max(0) > .9 * bounds) and np.all(euler.min(0) < -.9 * bounds)
    torch.testing.assert_close(torch.linalg.vector_norm(new_roots[:, 3:7], dim=-1), torch.ones(len(roots), dtype=roots.dtype))
    assert torch.all((new_dofs[..., 0] - dofs[..., 0]).abs() <= .05 + 1e-12)
    assert torch.all(new_dofs[..., 0] >= limits[:, 0])
    assert torch.all(new_dofs[..., 0] <= limits[:, 1])
    assert (new_dofs[:, 1, 0] == 0).any() and (new_dofs[:, 1, 0] == .01).any()
    assert (new_dofs[:, 2, 0] == 1).any()
    torch.testing.assert_close(new_dofs[..., 1], dofs[..., 1])
    torch.testing.assert_close(roots, before_root)
    torch.testing.assert_close(dofs, before_dof)
    torch.testing.assert_close(limits, before_limits)
    assert new_roots.data_ptr() != roots.data_ptr() and new_dofs.data_ptr() != dofs.data_ptr()


def test_zero_noise_is_identity_and_reproducible_generator_is_local():
    roots, dofs, limits = reference_batch(8)
    actual_root, actual_dof = perturb_reference_state(roots, dofs, limits,
        joint_position_half_range=0., root_position_half_range=(0., 0., 0.),
        root_rotation_half_range=(0., 0., 0.), root_linear_velocity_half_range=(0., 0., 0.),
        root_angular_velocity_half_range=(0., 0., 0.))
    torch.testing.assert_close(actual_root, roots)
    torch.testing.assert_close(actual_dof, dofs)
    before_rng = torch.get_rng_state().clone()
    first = perturb_reference_state(roots, dofs, limits, generator=torch.Generator().manual_seed(7))
    second = perturb_reference_state(roots, dofs, limits, generator=torch.Generator().manual_seed(7))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    assert torch.equal(torch.get_rng_state(), before_rng)

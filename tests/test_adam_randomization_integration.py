"""CPU integration checks for the opt-in Adam randomization profile."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.terrains import TerrainEntity

from humanoidverse.agents.envs.adam_randomization import randomize_mass_and_inertia
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabCore,
    _compose_humanoidverse_config,
    make_mjlab_ufo_env_cfg,
)
from humanoidverse.train import build_ufo_mjlab_config

ROOT = Path(__file__).resolve().parents[1]


def build_config(profile, *, disable_dr=False):
    env = build_ufo_mjlab_config(
        device="cpu", work_dir="/tmp/adam_randomization_config", num_envs=2,
        num_env_steps=4096, seed=4728, use_wandb=False, wandb_run_name=None,
        disable_dr=disable_dr,
        robot_config=str(ROOT / f"configs/robots/{profile}.yaml"),
    ).env
    values = env.model_dump()
    params = {k: values[k] for k in inspect.signature(_compose_humanoidverse_config).parameters if k != "num_envs"}
    hv, _ = _compose_humanoidverse_config(num_envs=2, **params)
    native = make_mjlab_ufo_env_cfg(
        hv, num_envs=2, seed=4728, mjcf_path=env.mjcf_path,
        auto_reset=False, robot_training=env.robot_training,
    )
    return hv, native


def test_new_profile_preserves_mechanical_mapping_and_legacy_profiles():
    old, old_native = build_config("adam_sp_29dof_ufo_mechanical")
    new, native = build_config("adam_sp_29dof_ufo_mechanical_dr")
    assert old.robot.control == new.robot.control
    assert old_native.scene.entities["robot"].articulation == native.scene.entities["robot"].articulation
    assert native.events["random_link_mass"].func is randomize_mass_and_inertia
    assert old_native.events["random_link_mass"].func is not randomize_mass_and_inertia
    assert native.events["random_geom_friction"].params["shared_random"]
    assert not old_native.events["random_geom_friction"].params["shared_random"]
    assert list(new.domain_rand.inference_delay_range_ms) == [10., 20.]
    assert list(old.domain_rand.inference_delay_range_ms) == [0., 0.]
    assert native.actions["actions"].inference_delay_range_ms == (10., 20.)
    assert new.domain_rand.randomize_reference_reset and not old.domain_rand.randomize_reference_reset
    assert native.sim.mujoco.timestep == old_native.sim.mujoco.timestep == .005
    assert native.decimation == old_native.decimation == 4
    assert not old_native.scene.terrain.collisions
    g1, g1_native = build_config("g1_29dof")
    assert list(g1.domain_rand.inference_delay_range_ms) == [0., 0.]
    assert not g1.domain_rand.randomize_reference_reset
    assert not g1_native.scene.terrain.collisions


def test_disabled_domain_randomization_removes_new_effects():
    hv, native = build_config("adam_sp_29dof_ufo_mechanical_dr", disable_dr=True)
    assert not native.events
    assert list(hv.domain_rand.inference_delay_range_ms) == [0., 0.]
    assert not hv.domain_rand.randomize_reference_reset
    assert not native.scene.terrain.collisions


def test_actual_mujoco_contacts_retain_low_robot_friction():
    _, native = build_config("adam_sp_29dof_ufo_mechanical_dr")
    # Use the actual terrain builder with a simple contact probe: this checks
    # MuJoCo contact mixing, not merely the configured coefficient.
    terrain = TerrainEntity(native.scene.terrain, device="cpu")
    spec = terrain.spec
    body = spec.worldbody.add_body(name="probe", pos=(0., 0., .09))
    body.add_freejoint()
    body.add_geom(name="probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=(.1, 0., 0.), friction=(.5, .005, .0001), mass=1.)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    np.testing.assert_allclose(data.contact.friction[:data.ncon, :2], .5)
    assert model.geom("terrain").priority == model.geom("probe_geom").priority == 0
    model.geom("probe_geom").friction[0] = 1.25
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.contact.friction[:data.ncon, :2], 1.25)


def test_reset_hold_accounts_for_joint_order_and_preserves_policy_history():
    core = object.__new__(HumanoidVerseMjlabCore)
    core._action_term_dof_indices = torch.tensor([1, 0])
    core._joint_ids = torch.tensor([1, 0])
    core.default_dof_pos = torch.tensor([[.2, -.3], [.2, -.3]])
    core.action_target_scale = torch.tensor([[.5, .25]])
    writes = []
    core._inference_delay_action = SimpleNamespace(
        reset_inference_delay=lambda ids, pose: writes.append(("delay_hold", pose.clone())),
    )
    native_robot = object.__new__(Entity)
    native_robot._data = SimpleNamespace(joint_pos_target=torch.zeros(2, 2))
    core.robot = SimpleNamespace(
        reset=lambda ids: writes.append(("reset", ids.clone())),
        set_joint_position_target=native_robot.set_joint_position_target,
    )
    core.actions = torch.tensor([[1., 2.], [3., 4.]])
    before = core.actions.clone()
    pose = torch.tensor([[.4, -.1], [-.2, .3]])
    core._reset_inference_delay(torch.tensor([0, 1]), pose)
    order = core._action_term_dof_indices
    torch.testing.assert_close(writes[0][1], pose[:, order])
    torch.testing.assert_close(core.actions, before)
    assert [item[0] for item in writes] == ["delay_hold", "reset"]
    torch.testing.assert_close(native_robot._data.joint_pos_target[:, core._joint_ids], pose)
    # A subset reset must not overwrite the other environment's hold command.
    changed = torch.tensor([[.7, .1]])
    core._reset_inference_delay(torch.tensor([1]), changed)
    torch.testing.assert_close(native_robot._data.joint_pos_target[0, core._joint_ids], pose[0])
    torch.testing.assert_close(native_robot._data.joint_pos_target[1, core._joint_ids], changed[0])

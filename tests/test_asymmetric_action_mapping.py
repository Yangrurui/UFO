"""Joint-command semantics, legacy fallback and deployment action metadata."""

import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabCore,
    _action_target_ranges,
    _asymmetric_joint_position_targets,
)
from humanoidverse import tracking_inference
from humanoidverse.utils.robot_spec import load_robot_training_spec


def mapping_config(ranges=None):
    cfg = OmegaConf.create({"robot": {
        "dof_names": ["joint_0", "joint_1"],
        "dof_pos_lower_limit_list": [-3.0, -3.0],
        "dof_pos_upper_limit_list": [3.0, 3.0],
        "init_state": {"default_joint_angles": {"joint_0": 0.2, "joint_1": -0.4}},
        "control": {
            "normalize_action": True, "normalize_action_from": 1.0,
            "normalize_action_to": 5.0, "action_clip_value": 5.0,
        },
    }})
    if ranges is not None:
        cfg.robot.control.action_target_ranges = ranges
    return cfg


def test_asymmetric_zero_endpoints_midpoints_and_clipping():
    bounds = torch.tensor([[-1., 2.], [-2., .5]])
    center = torch.tensor([[.2, -.4]])
    actions = torch.tensor([[-1., -1.], [0., 0.], [1., 1.], [-.5, .5], [3., -3.]])
    before = actions.clone()
    result = _asymmetric_joint_position_targets(actions, center, bounds)
    torch.testing.assert_close(result, torch.tensor([[-1., -2.], [.2, -.4], [2., .5], [-.4, .05], [2., -2.]]))
    torch.testing.assert_close(actions, before)


def test_randomized_centers_are_batched_clamped_and_applied_once():
    bounds = torch.tensor([[-1., 2.], [-2., .5]])
    center = torch.tensor([[.5, -.6], [-10., 10.]])
    before = center.clone()
    result = _asymmetric_joint_position_targets(torch.tensor([[0., 0.], [.5, -.5]]), center, bounds)
    torch.testing.assert_close(result, torch.tensor([[.5, -.6], [.5, -.75]]))
    torch.testing.assert_close(center, before)


def test_mjlab_input_reconstructs_pd_targets_in_actuator_order_without_changing_actions():
    core = object.__new__(HumanoidVerseMjlabCore)
    core.config = mapping_config()
    raw_actions = torch.tensor([[0., .5], [-1., 1.]])
    original_raw = raw_actions.clone()
    core.actions = core._normalized_action(raw_actions)
    core.default_dof_pos = torch.tensor([[.2, -.4], [.2, -.4]])
    core.default_dof_pos_offset = torch.tensor([[.1, .2], [-.1, -.3]])
    core.action_target_scale = torch.tensor([[.25, .4]])
    core.action_target_ranges = torch.tensor([[-1., 2.], [-2., .5]])
    core._action_term_dof_indices = torch.tensor([1, 0])
    expected_targets = torch.tensor([[.3, .15], [-1., .5]])
    inputs = core._mjlab_action_input()
    order = core._action_term_dof_indices
    applied_targets = inputs * core.action_target_scale[:, order] + core.default_dof_pos[:, order]
    torch.testing.assert_close(applied_targets, expected_targets[:, order])
    torch.testing.assert_close(core.actions, torch.tensor([[0., 2.5], [-5., 5.]]))
    torch.testing.assert_close(raw_actions, original_raw)


def test_legacy_mapping_keeps_scale_offset_and_order_without_new_attributes():
    core = object.__new__(HumanoidVerseMjlabCore)
    core.actions = torch.tensor([[10., 20., 30., 40.]])
    core.default_dof_pos_offset = torch.tensor([[1., 2., 3., 4.]])
    core.action_target_scale = torch.tensor([[1., 2., 1., 4.]])
    core._action_term_dof_indices = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(core._mjlab_action_input(), torch.tensor([[33., 11., 41., 21.]]))


def test_ranges_use_control_joint_order_and_remain_optional():
    cfg = mapping_config({"joint_1": [-2., .5], "joint_0": [-1., 2.]})
    torch.testing.assert_close(_action_target_ranges(cfg), torch.tensor([[-1., 2.], [-2., .5]]))
    assert _action_target_ranges(mapping_config()) is None


@pytest.mark.parametrize("ranges", [
    {"joint_0": [-1., 2.]},
    {"joint_0": [-1., 2.], "joint_1": [-2., .5], "unknown": [-1., 1.]},
    {"joint_0": [2., -1.], "joint_1": [-2., .5]},
    {"joint_0": [1., 1.], "joint_1": [-2., .5]},
    {"joint_0": [float("nan"), 2.], "joint_1": [-2., .5]},
    {"joint_0": [-1., float("inf")], "joint_1": [-2., .5]},
    {"joint_0": [-4., 2.], "joint_1": [-2., .5]},
    {"joint_0": [-1., 4.], "joint_1": [-2., .5]},
    {"joint_0": [.3, 2.], "joint_1": [-2., .5]},
    {"joint_0": [-1., 2., 3.], "joint_1": [-2., .5]},
    {"joint_0": 1.0, "joint_1": [-2., .5]},
])
def test_invalid_target_ranges_are_rejected(ranges):
    with pytest.raises(ValueError):
        _action_target_ranges(mapping_config(ranges))


def test_adam_candidate_export_has_action_mapping_without_changing_actor(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = load_robot_training_spec(root / "configs/robots/adam_sp_29dof_ufo_asymmetric.yaml")
    model = SimpleNamespace(cfg=SimpleNamespace(archi=SimpleNamespace(z_dim=256)))
    exported_models = []

    def export_actor(actor, output_dir, output_name, **kwargs):
        exported_models.append(actor)
        return {"output_action_dim": 29, "z_dim": kwargs["z_dim"]}

    monkeypatch.setattr(tracking_inference, "export_meta_policy_as_onnx", export_actor)
    report = tracking_inference._export_policy_model(model, tmp_path, spec)
    saved = json.loads((tmp_path / "SimpleNamespace.meta.json").read_text())
    assert report == saved and exported_models == [model]
    mapping = saved["action_mapping"]
    assert mapping["raw_action_bounds"] == [-1., 1.]
    assert mapping["raw_to_native_scale"] == mapping["native_action_clip"] == 5.
    assert mapping["joint_order"] == list(spec.robot.control_joint_names)
    assert mapping["target_ranges"] == spec.actuator["action_target_ranges"]
    assert mapping["default_joint_angles"] == spec.default_joint_angles
    for group in spec.actuator["groups"]:
        for name in group["joints"]:
            assert mapping["nominal_pd_gains"][name] == {
                "kp": group["parameters"]["stiffness"],
                "kd": group["parameters"]["damping"],
            }
    assert "default_dof_pos_offset" in mapping["center_formula"]
    assert "where" in mapping["target_formula"]


def test_tracking_metadata_remains_optional_for_legacy_configs():
    assert tracking_inference._action_mapping_metadata(SimpleNamespace()) is None

"""CPU tests for explicit Adam runtime selection and checkpoint rollback."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from humanoidverse.optimizer_runtime import configure_fused_adam

_NAMES = ("actor_optimizer", "backward_optimizer", "forward_optimizer",
          "critic_optimizer", "discriminator_optimizer", "aux_critic_optimizer")


def initialized_adam():
    active = torch.nn.Parameter(torch.linspace(-0.3, 0.7, 31))
    unused = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.Adam([active, unused], lr=3e-4, weight_decay=0.01,
                                 eps=1e-7, amsgrad=True, foreach=True)
    active.grad = torch.linspace(-0.2, 0.4, 31)
    optimizer.step()
    return optimizer, active, unused


def assert_flags(optimizer, enabled):
    for group in [optimizer.defaults, *optimizer.param_groups]:
        assert group["fused"] is enabled
        assert group["foreach"] is (not enabled)
        assert group["capturable"] is False
    assert optimizer._step_supports_amp_scaling is enabled


def test_all_six_adam_optimizers_preserve_objects_and_moments():
    agent = SimpleNamespace()
    snapshots = []
    for name in _NAMES:
        optimizer, active, unused = initialized_adam()
        setattr(agent, name, optimizer)
        snapshots.append((name, optimizer, active, unused,
                          dict(optimizer.state[active]), deepcopy(optimizer.state[active])))
    report = configure_fused_adam(agent)
    assert report == {"optimizer_count": 6, "parameter_groups": 6,
                      "initialized_parameters": 6, "step_counters_moved": 0}
    for name, optimizer, active, unused, identities, values in snapshots:
        assert getattr(agent, name) is optimizer
        assert optimizer.param_groups[0]["params"][0] is active
        assert unused not in optimizer.state and unused.grad is None
        assert_flags(optimizer, True)
        for field, value in values.items():
            assert optimizer.state[active][field] is identities[field]
            torch.testing.assert_close(optimizer.state[active][field], value, rtol=0, atol=0)
        assert optimizer.param_groups[0]["lr"] == 3e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        assert optimizer.param_groups[0]["eps"] == 1e-7
    # Reapply after a fused checkpoint to restore the constructor-only AMP flag.
    agent.actor_optimizer._step_supports_amp_scaling = False
    assert configure_fused_adam(agent) == report
    assert_flags(agent.actor_optimizer, True)


def test_fused_checkpoint_rolls_back_to_foreach_and_continues_identically(tmp_path):
    original, active, unused = initialized_adam()
    reference_active = torch.nn.Parameter(active.detach().clone())
    reference_unused = torch.nn.Parameter(unused.detach().clone())
    reference = torch.optim.Adam([reference_active, reference_unused], foreach=True)
    reference.load_state_dict(deepcopy(original.state_dict()))
    configure_fused_adam(SimpleNamespace(actor_optimizer=original), enabled=True)
    checkpoint = tmp_path / "fused_adam.pt"
    torch.save({"parameters": [active.detach(), unused.detach()], "optimizer": original.state_dict()}, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored_active, restored_unused = [torch.nn.Parameter(value.clone()) for value in payload["parameters"]]
    restored = torch.optim.Adam([restored_active, restored_unused], foreach=True)
    restored.load_state_dict(payload["optimizer"])
    assert restored.param_groups[0]["fused"] is True
    # load_state_dict restores group flags but not the fused constructor AMP flag.
    configure_fused_adam(SimpleNamespace(actor_optimizer=restored), enabled=True)
    assert_flags(restored, True)
    state = restored.state[restored_active]
    identities = dict(state)
    configure_fused_adam(SimpleNamespace(actor_optimizer=restored), enabled=False)
    assert_flags(restored, False)
    assert all(state[field] is tensor for field, tensor in identities.items())
    for step in range(3):
        gradient = torch.linspace(-0.4, 0.6, 31) + step * 0.01
        reference_active.grad = gradient.clone()
        restored_active.grad = gradient.clone()
        reference.step()
        restored.step()
        torch.testing.assert_close(restored_active, reference_active, rtol=0, atol=0)
        for field, value in reference.state[reference_active].items():
            torch.testing.assert_close(state[field], value, rtol=0, atol=0)
    assert restored_unused not in restored.state and restored_unused.grad is None


@pytest.mark.parametrize("bad_configuration", ["capturable", "differentiable", "dtype", "step", "moment"])
def test_invalid_agent_is_rejected_before_any_optimizer_changes(bad_configuration):
    first, _, _ = initialized_adam()
    second, active, _ = initialized_adam()
    if bad_configuration in ("capturable", "differentiable"):
        second.param_groups[0][bad_configuration] = True
    elif bad_configuration == "dtype":
        active.data = active.data.double()
    elif bad_configuration == "step":
        second.state[active]["step"] = torch.tensor(1.0, dtype=torch.float64)
    else:
        second.state[active].pop("exp_avg")
    with pytest.raises(ValueError):
        configure_fused_adam(SimpleNamespace(actor_optimizer=first, critic_optimizer=second))
    assert first.param_groups[0]["foreach"] is True
    assert first.param_groups[0]["fused"] is None
    assert not getattr(first, "_step_supports_amp_scaling", False)


def test_shared_optimizer_is_configured_once_and_empty_state_stays_empty():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.Adam([parameter], foreach=True)
    report = configure_fused_adam(SimpleNamespace(actor_optimizer=optimizer, critic_optimizer=optimizer))
    assert report["optimizer_count"] == 1 and report["initialized_parameters"] == 0
    assert len(optimizer.state) == 0
    assert_flags(optimizer, True)

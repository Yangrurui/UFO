"""CPU tests for Adam counter restoration, including a simulated device branch.

The tensor subclass below only changes reported device metadata. All storage
and optimizer updates remain on CPU; real CUDA transfer needs GPU integration.
"""

import copy
import os
from types import SimpleNamespace

import pytest
import torch

from humanoidverse.distributed import agent_optimizers
from humanoidverse.training import workspace as training

_OPTIMIZER_NAMES = (
    "actor_optimizer",
    "backward_optimizer",
    "forward_optimizer",
    "critic_optimizer",
    "discriminator_optimizer",
    "aux_critic_optimizer",
)


class _ReportedCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor):
        result = torch.Tensor._make_subclass(cls, tensor.detach().clone(), False)
        result.cpu_calls = 0
        return result

    @property
    def device(self):
        return torch.device("cuda:0")

    def cpu(self, *args, **kwargs):
        self.cpu_calls += 1
        return self.as_subclass(torch.Tensor).clone()


def _update(optimizer, active, index):
    optimizer.zero_grad(set_to_none=True)
    target = torch.tensor([0.2 + index / 10, -0.4], dtype=active.dtype, device=active.device)
    (active - target).square().sum().backward()
    optimizer.step()


def _initialized_adam():
    active = torch.nn.Parameter(torch.tensor([0.6, -0.2], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float64))
    optimizer = torch.optim.Adam([active, unused], lr=0.007, amsgrad=True, foreach=False)
    for index in range(2):
        _update(optimizer, active, index)
    return optimizer, active, unused


@pytest.mark.parametrize("step_dtype", [torch.float32, torch.float64])
def test_saved_adam_restores_counters_and_continues_identically(tmp_path, monkeypatch, step_dtype):
    monkeypatch.delenv("UFO_CPU_OPTIMIZER_STEPS", raising=False)
    reference, active, unused = _initialized_adam()
    reference.state[active]["optional"] = None
    reference.state[active]["empty"] = torch.empty(0, dtype=active.dtype)
    checkpoint = tmp_path / "adam.pt"
    torch.save({"parameters": [active.detach(), unused.detach()], "optimizer": reference.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored_active, restored_unused = [torch.nn.Parameter(value.clone()) for value in saved["parameters"]]
    restored = torch.optim.Adam([restored_active, restored_unused], lr=0.007, amsgrad=True, foreach=False)
    restored.load_state_dict(saved["optimizer"])
    agent = SimpleNamespace(actor_optimizer=restored)

    # An ordinary CPU checkpoint already has the desired placement.
    cpu_step = restored.state[restored_active]["step"]
    assert training._restore_adam_step_devices(agent) == 0
    assert restored.state[restored_active]["step"] is cpu_step

    reference.state[active]["step"] = reference.state[active]["step"].to(step_dtype)
    restored_state = restored.state[restored_active]
    fake_step = _ReportedCudaTensor(restored_state["step"].to(step_dtype))
    restored_state["step"] = fake_step
    untouched = {key: value for key, value in restored_state.items() if key != "step"}
    assert training._restore_adam_step_devices(agent) == 1
    counter = restored_state["step"]
    assert type(counter) is torch.Tensor
    assert counter.device.type == "cpu" and counter.dtype == step_dtype
    assert counter.item() == 2
    assert fake_step.cpu_calls == 1
    assert all(restored_state[key] is value for key, value in untouched.items())
    assert restored_unused not in restored.state
    assert restored_unused.grad is None
    assert training._restore_adam_step_devices(agent) == 0
    assert restored_state["step"] is counter

    for index in range(2, 5):
        _update(reference, active, index)
        _update(restored, restored_active, index)
        torch.testing.assert_close(restored_active, active, rtol=0, atol=0)
        torch.testing.assert_close(restored_unused, unused, rtol=0, atol=0)
        assert restored_state.keys() == reference.state[active].keys()
        for key, expected in reference.state[active].items():
            actual = restored_state[key]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                assert actual is expected
        assert restored_unused.grad is None
        assert restored_unused not in restored.state
    assert counter.item() == 5


def test_all_six_agent_optimizers_are_restored(monkeypatch):
    monkeypatch.setenv("UFO_CPU_OPTIMIZER_STEPS", "1")
    agent = SimpleNamespace()
    steps = []
    for name in _OPTIMIZER_NAMES:
        optimizer, active, _ = _initialized_adam()
        step = _ReportedCudaTensor(optimizer.state[active]["step"])
        optimizer.state[active]["step"] = step
        steps.append(step)
        setattr(agent, name, optimizer)
    assert len(agent_optimizers(agent)) == 6
    assert training._restore_adam_step_devices(agent) == 6
    assert all(step.cpu_calls == 1 for step in steps)


def test_disabled_switch_preserves_non_cpu_step(monkeypatch):
    monkeypatch.setenv("UFO_CPU_OPTIMIZER_STEPS", "0")
    optimizer, active, _ = _initialized_adam()
    step = _ReportedCudaTensor(optimizer.state[active]["step"])
    optimizer.state[active]["step"] = step
    assert training._restore_adam_step_devices(SimpleNamespace(actor_optimizer=optimizer)) == 0
    assert optimizer.state[active]["step"] is step
    assert step.cpu_calls == 0


@pytest.mark.parametrize("excluded_flags", [{"capturable": True}, {"fused": True}, {"capturable": True, "fused": True}])
def test_capturable_and_fused_groups_are_excluded(monkeypatch, excluded_flags):
    monkeypatch.setenv("UFO_CPU_OPTIMIZER_STEPS", "1")
    first = torch.nn.Parameter(torch.tensor([0.4]))
    second = torch.nn.Parameter(torch.tensor([0.7]))
    optimizer = torch.optim.Adam([{"params": [first]}, {"params": [second]}], foreach=False)
    (first.square() + second.square()).sum().backward()
    optimizer.step()
    # Set these flags only after a normal CPU update initialized real states.
    optimizer.param_groups[1].update(excluded_flags)
    first_step = _ReportedCudaTensor(optimizer.state[first]["step"])
    second_step = _ReportedCudaTensor(optimizer.state[second]["step"])
    optimizer.state[first]["step"] = first_step
    optimizer.state[second]["step"] = second_step
    assert training._restore_adam_step_devices(SimpleNamespace(actor_optimizer=optimizer)) == 1
    assert optimizer.state[first]["step"].device.type == "cpu"
    assert optimizer.state[second]["step"] is second_step
    assert first_step.cpu_calls == 1 and second_step.cpu_calls == 0


def test_non_adam_and_uninitialized_states_are_unchanged(monkeypatch):
    monkeypatch.setenv("UFO_CPU_OPTIMIZER_STEPS", "1")
    parameter = torch.nn.Parameter(torch.tensor([0.4]))
    sgd = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.square().sum().backward()
    sgd.step()
    step = _ReportedCudaTensor(torch.tensor(4.0))
    sgd.state[parameter]["step"] = step
    momentum = sgd.state[parameter]["momentum_buffer"]
    uninitialized = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    agent = SimpleNamespace(actor_optimizer=sgd, critic_optimizer=uninitialized)
    assert training._restore_adam_step_devices(agent) == 0
    assert sgd.state[parameter]["step"] is step
    assert sgd.state[parameter]["momentum_buffer"] is momentum
    assert step.cpu_calls == 0
    assert len(uninitialized.state) == 0
    assert training._restore_adam_step_devices(SimpleNamespace()) == 0


@pytest.mark.parametrize("step", [None, 3])
def test_missing_or_non_tensor_steps_are_preserved(monkeypatch, step):
    monkeypatch.setenv("UFO_CPU_OPTIMIZER_STEPS", "1")
    optimizer, active, _ = _initialized_adam()
    optimizer.state[active]["step"] = step
    assert training._restore_adam_step_devices(SimpleNamespace(actor_optimizer=optimizer)) == 0
    assert optimizer.state[active]["step"] is step
    del optimizer.state[active]["step"]
    snapshot = copy.copy(optimizer.state[active])
    assert training._restore_adam_step_devices(SimpleNamespace(actor_optimizer=optimizer)) == 0
    assert optimizer.state[active].keys() == snapshot.keys()
    assert all(optimizer.state[active][key] is value for key, value in snapshot.items())


def test_workspace_broadcasts_before_restoring_adam_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("UFO_CPU_OPTIMIZER_STEPS", "1")
    events = []
    optimizer, active, _ = _initialized_adam()
    optimizer.state[active]["step"] = _ReportedCudaTensor(optimizer.state[active]["step"])

    class Model(torch.nn.Module):
        def train(self, mode=True):
            assert optimizer.state[active]["step"].device.type == "cpu"
            events.append("train")
            return super().train(mode)

    observation_space = SimpleNamespace(spaces={"obs": object(), "time": object()}, keys=lambda: ("obs", "time"))
    sample_env = SimpleNamespace(observation_space=observation_space, action_space=SimpleNamespace(shape=(2,)))
    cfg = SimpleNamespace(
        distributed_rank=1,
        distributed_world_size=2,
        rank0_only_writes=True,
        env=SimpleNamespace(build=lambda num_envs: (sample_env, {})),
        online_parallel_envs=2,
        work_dir=str(tmp_path / "workspace"),
        seed=5,
        distributed_sync=True,
        evaluations=[],
        use_wandb=False,
        prioritization=False,
    )
    agent = SimpleNamespace(actor_optimizer=optimizer, _model=Model())

    def load_agent(*args, **kwargs):
        events.append("load")
        return agent, cfg, {"local_time": 10, "global_time": 20, "optimizer_steps": 2}

    def broadcast(loaded_agent, src):
        assert loaded_agent is agent and src == 0
        assert optimizer.state[active]["step"].device.type == "cuda"
        events.append("broadcast")

    actual_restore = training._restore_adam_step_devices

    def restore(loaded_agent):
        assert events == ["load", "broadcast"]
        assert actual_restore(loaded_agent) == 1
        events.append("restore")
        return 1

    monkeypatch.setattr(training, "set_seed_everywhere", lambda seed: None)
    monkeypatch.setattr(training, "create_agent_or_load_checkpoint", load_agent)
    monkeypatch.setattr(training, "broadcast_agent_state", broadcast)
    monkeypatch.setattr(training, "_restore_adam_step_devices", restore)
    workspace = training.Workspace(cfg)
    assert workspace.agent is agent
    assert events == ["load", "broadcast", "restore", "train"]
    assert workspace._optimizer_steps == 2


@pytest.mark.skipif(
    os.environ.get("UFO_TEST_CUDA") != "1",
    reason="Explicit CUDA integration opt-in required: UFO_TEST_CUDA=1",
)
def test_cuda_checkpoint_step_restore_matches_foreach_adam(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setenv("UFO_CPU_OPTIMIZER_STEPS", "1")
    cpu_parameters = [
        torch.nn.Parameter(torch.tensor([0.6, -0.2], dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor([0.1, 0.3], dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor([0.7], dtype=torch.float64)),
    ]
    cpu_optimizer = torch.optim.Adam(cpu_parameters, lr=0.007, foreach=True, amsgrad=True)

    def update(optimizer, parameters, index):
        optimizer.zero_grad(set_to_none=True)
        target = torch.tensor([0.2 + index / 10, -0.4], device=parameters[0].device, dtype=parameters[0].dtype)
        loss = (parameters[0] - target).square().sum() + (parameters[1] + target / 2).square().sum()
        loss.backward()
        optimizer.step()

    for index in range(2):
        update(cpu_optimizer, cpu_parameters, index)
    assert all(state["step"].device.type == "cpu" for state in cpu_optimizer.state.values())
    checkpoint = tmp_path / "cuda_adam.pt"
    torch.save({"parameters": [p.detach() for p in cpu_parameters], "optimizer": cpu_optimizer.state_dict()}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cuda", weights_only=True)

    def restore_optimizer():
        parameters = [torch.nn.Parameter(value.clone()) for value in loaded["parameters"]]
        optimizer = torch.optim.Adam(parameters, lr=0.007, foreach=True, amsgrad=True)
        optimizer.load_state_dict(copy.deepcopy(loaded["optimizer"]))
        assert optimizer.param_groups[0]["foreach"] is True
        assert all(state["step"].device.type == "cuda" for state in optimizer.state.values())
        return optimizer, parameters

    baseline, baseline_parameters = restore_optimizer()
    restored, restored_parameters = restore_optimizer()
    moments = {
        (parameter, key): value
        for parameter, state in restored.state.items()
        for key, value in state.items()
        if key != "step"
    }
    assert training._restore_adam_step_devices(SimpleNamespace(actor_optimizer=restored)) == 2
    for state in restored.state.values():
        assert state["step"].device.type == "cpu"
        assert state["step"].dtype == torch.float32
        assert state["step"].item() == 2
    for (parameter, key), value in moments.items():
        assert restored.state[parameter][key] is value
        assert value.device.type == "cuda"
    assert restored_parameters[2] not in restored.state

    for index in range(2, 5):
        update(baseline, baseline_parameters, index)
        update(restored, restored_parameters, index)
        for expected_parameter, actual_parameter in zip(baseline_parameters, restored_parameters):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
            expected_state = baseline.state.get(expected_parameter, {})
            actual_state = restored.state.get(actual_parameter, {})
            assert actual_state.keys() == expected_state.keys()
            for key, expected_value in expected_state.items():
                actual_value = actual_state[key]
                if key == "step":
                    assert expected_value.device.type == "cuda"
                    assert actual_value.device.type == "cpu"
                    torch.testing.assert_close(actual_value, expected_value.cpu(), rtol=0, atol=0)
                else:
                    assert actual_value is moments[(actual_parameter, key)]
                    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        assert restored_parameters[2].grad is None
        assert restored_parameters[2] not in restored.state

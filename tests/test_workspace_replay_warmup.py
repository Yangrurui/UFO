"""Replay warmup uses new local samples without resetting policy exploration."""

import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from test_workspace_final_checkpoint import make_workspace as _make_workspace
from test_workspace_final_checkpoint import read_status

make_workspace = _make_workspace


def record_training(work, env, monkeypatch):
    """Observe actual rollout/update choices in the real CPU training loop."""
    random_actions, policy_actions, update_times = [], [], []
    original_act = work.agent.act
    original_update = work.agent.update

    def random_action():
        random_actions.append(env.steps)
        return np.zeros((env.num_envs, 1), dtype=np.float32)

    def policy_action(*args, **kwargs):
        policy_actions.append(env.steps)
        return original_act(*args, **kwargs)

    def update(replay_buffer, local_time):
        update_times.append(local_time)
        return original_update(replay_buffer, local_time)

    env.action_space = SimpleNamespace(sample=random_action)
    monkeypatch.setattr(work.agent, "act", policy_action)
    monkeypatch.setattr(work.agent, "update", update)
    return random_actions, policy_actions, update_times


@pytest.mark.parametrize("world_size", [1, 8])
def test_empty_replay_resume_collects_new_local_samples_with_learned_policy(
    make_workspace, monkeypatch, capsys, world_size
):
    initial, _, _ = make_workspace(budget=20 * world_size, world_size=world_size)
    initial.train_online()
    previous_updates = initial.agent.updates
    # Keep the model/status checkpoint, removing only this test's old replay.
    shutil.rmtree(initial._checkpoint_buffer_path(initial.work_dir / "checkpoint"))
    resumed, env, _ = make_workspace(budget=30 * world_size, world_size=world_size)
    resumed.cfg.num_seed_steps = 4
    random_actions, policy_actions, update_times = record_training(resumed, env, monkeypatch)

    resumed.train_online()

    assert env.steps == 5
    assert random_actions == []
    assert policy_actions == [0, 1, 2, 3, 4]
    # Keep the existing strict > seed boundary: start=20, warmup=4.
    assert update_times == [26, 26, 28, 28]
    assert resumed.agent.updates == previous_updates + 4
    status = read_status(resumed)
    assert status["local_time"] == 30
    assert status["global_time"] == 30 * world_size
    assert status["optimizer_steps"] == previous_updates + 4
    output = capsys.readouterr().out
    assert "replay_warmup_local_samples=4" in output
    assert "optimizer_updates_require_local_time>24" in output
    assert "random_action_seed_steps_local=4 (unchanged)" in output


@pytest.mark.parametrize("world_size", [1, 8])
def test_complete_resume_keeps_existing_update_schedule(make_workspace, monkeypatch, capsys, world_size):
    initial, _, _ = make_workspace(budget=20 * world_size, world_size=world_size)
    initial.train_online()
    previous_updates = initial.agent.updates
    resumed, env, _ = make_workspace(budget=30 * world_size, world_size=world_size)
    resumed.cfg.num_seed_steps = 4
    random_actions, policy_actions, update_times = record_training(resumed, env, monkeypatch)

    resumed.train_online()

    assert random_actions == []
    assert policy_actions == [0, 1, 2, 3, 4]
    assert update_times == [22, 22, 24, 24, 26, 26, 28, 28]
    assert read_status(resumed)["optimizer_steps"] == previous_updates + 8
    assert "empty replay after checkpoint resume" not in capsys.readouterr().out


@pytest.mark.parametrize("world_size", [1, 8])
def test_fresh_training_keeps_random_seed_and_original_update_boundary(make_workspace, monkeypatch, capsys, world_size):
    work, env, _ = make_workspace(budget=10 * world_size, world_size=world_size)
    work.cfg.num_seed_steps = 4
    random_actions, policy_actions, update_times = record_training(work, env, monkeypatch)

    work.train_online()

    assert random_actions == [0, 1]
    assert policy_actions == [2, 3, 4]
    assert update_times == [6, 6, 8, 8]
    assert read_status(work)["optimizer_steps"] == 4
    assert "empty replay after checkpoint resume" not in capsys.readouterr().out

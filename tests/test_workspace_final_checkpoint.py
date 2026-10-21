"""Exercise the training loop and real checkpoint/buffer writes on tiny CPU data.

Only the simulator, policy and distributed collectives are substitutes; these
tests do not require a GPU or change existing training runs.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoidverse.training import workspace as training


class TinyEnv:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.steps = 0
        self.closed = False

    def observation(self):
        return {
            "state": np.full((self.num_envs, 1), self.steps, dtype=np.float32),
            "time": np.full((self.num_envs, 1), self.steps, dtype=np.int64),
        }

    def reset(self):
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        return (
            self.observation(),
            np.ones(self.num_envs, dtype=np.float32),
            np.zeros(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            {},
        )

    def close(self):
        self.closed = True


class TinyAgent:
    device = "cpu"

    def __init__(self, updates=0):
        self.updates = updates
        self._model = torch.nn.Linear(1, 1)
        self.fail_on_update = None
        self.failure = None

    def maybe_update_rollout_context(self, **kwargs):
        return None

    def act(self, obs, **kwargs):
        return torch.zeros_like(obs["state"])

    def update(self, replay_buffer, local_time):
        assert len(replay_buffer["train"]) > 0
        self.updates += 1
        if self.updates == self.fail_on_update:
            if self.failure == "exception":
                raise RuntimeError("synthetic update failure")
            return {"loss": torch.tensor(float("nan"))}
        return {"loss": torch.tensor(float(self.updates))}

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tiny_agent.json").write_text(json.dumps({"updates": self.updates}))

    @classmethod
    def load(cls, folder, device):
        return cls(**json.loads((Path(folder) / "tiny_agent.json").read_text()))


@pytest.fixture
def make_workspace(tmp_path, monkeypatch):
    # Simulate global sample accounting without launching distributed workers.
    # The actual multi-rank barrier/write path is covered by GPU integration.
    monkeypatch.setattr(training, "barrier", lambda: None)
    monkeypatch.setattr(training, "sync_floating_buffers", lambda model: None)
    monkeypatch.setattr(training, "average_metrics", lambda metrics: metrics)
    monkeypatch.setattr(training, "module_sync_report", lambda model, src: {"max_abs_diff_from_rank0": 0.0})

    def make(*, budget, checkpoint_every=1000, rank=0, world_size=8):
        env = TinyEnv()
        agent_cfg = SimpleNamespace(
            object_class=TinyAgent,
            build=lambda **kwargs: TinyAgent(),
            model=SimpleNamespace(device="cpu"),
            train=SimpleNamespace(batch_size=2),
            compile=False,
        )
        cfg = SimpleNamespace(
            env=SimpleNamespace(build=lambda num_envs: (env, {})),
            agent=agent_cfg,
            online_parallel_envs=env.num_envs,
            num_env_steps=budget,
            buffer_size=64,
            buffer_device="cpu",
            use_trajectory_buffer=True,
            checkpoint_buffer=True,
            checkpoint_rank_buffers=True,
            checkpoint_every_steps=checkpoint_every,
            eval_every_steps=1000,
            log_every_updates=1000,
            update_agent_every=env.num_envs,
            num_seed_steps=0,
            num_agent_updates=2,
            distributed_rank=rank,
            distributed_world_size=world_size,
            distributed_sync=world_size > 1,
            distributed_global_steps=True,
            distributed_average_metrics=True,
            disable_tqdm=True,
            fail_on_nonfinite=True,
            nonfinite_check_rollout_every_local_steps=2,
            nonfinite_check_model_every_updates=0,
            log_torso_contact_forces=False,
            use_wandb=False,
            prioritization=False,
        )
        work = training.Workspace.__new__(training.Workspace)
        work.cfg = cfg
        work.work_dir = tmp_path
        work.distributed_rank = rank
        work.distributed_world_size = world_size
        work._write_shared_artifacts = rank == 0
        work.training_with_expert_data = False
        work.evaluations = {}
        work.evaluate = False
        work.train_logger = None
        work.agent, _, status = training.create_agent_or_load_checkpoint(tmp_path, cfg, {})
        work._checkpoint_local_time = status["local_time"]
        work._checkpoint_global_time = status["global_time"]
        work._optimizer_steps = status["optimizer_steps"]
        calls = []
        original_save = work.save

        def save(**kwargs):
            calls.append({k: v for k, v in kwargs.items() if k != "replay_buffer"})
            original_save(**kwargs)

        work.save = save
        return work, env, calls

    return make


def read_status(work):
    return json.loads((work.work_dir / "checkpoint/train_status.json").read_text())


@pytest.mark.parametrize("world_size,budget", [(1, 6), (1, 7), (8, 48), (8, 53)])
def test_final_checkpoint_contains_completed_rollouts_and_updates(make_workspace, world_size, budget):
    work, env, calls = make_workspace(budget=budget, world_size=world_size)
    work.train_online()
    assert env.steps == 3 and env.closed
    assert calls == [{"local_time": 6, "global_time": 6 * world_size, "optimizer_steps": 4}]
    status = read_status(work)
    assert status["global_time"] == 6 * world_size <= budget
    assert status["optimizer_steps"] == work.agent.updates == 4
    restored = training.TrajectoryDictBufferMultiDim.load(
        work._checkpoint_buffer_path(work.work_dir / "checkpoint"), device="cpu"
    )
    assert len(restored) == 3
    torch.testing.assert_close(restored.storage["observation"]["state"][:3, 0, 0], torch.tensor([0., 1., 2.]))


@pytest.mark.parametrize("budget", [48, 53])
def test_final_periodic_checkpoint_is_not_written_twice(make_workspace, budget):
    work, env, calls = make_workspace(budget=budget, checkpoint_every=16)
    work.train_online()
    assert [entry["global_time"] for entry in calls] == [16, 32, 48]
    assert [entry["optimizer_steps"] for entry in calls] == [0, 2, 4]
    assert read_status(work)["optimizer_steps"] == 4


def test_resume_saves_new_tail_with_restored_buffer_and_optimizer_count(make_workspace):
    initial, _, _ = make_workspace(budget=32)
    initial.train_online()
    resumed, env, calls = make_workspace(budget=85, checkpoint_every=32)
    resumed.train_online()
    assert env.steps == 3
    assert [entry["global_time"] for entry in calls] == [64, 80]
    status = read_status(resumed)
    assert status["local_time"] == 10 and status["global_time"] == 80
    assert status["optimizer_steps"] == resumed.agent.updates == 6
    restored = training.TrajectoryDictBufferMultiDim.load(
        resumed._checkpoint_buffer_path(resumed.work_dir / "checkpoint"), device="cpu"
    )
    assert len(restored) == 5


@pytest.mark.parametrize("budget", [32, 35, 16])
def test_already_completed_budget_keeps_existing_checkpoint(make_workspace, budget):
    initial, _, _ = make_workspace(budget=32)
    initial.train_online()
    original = read_status(initial)
    resumed, env, calls = make_workspace(budget=budget)
    resumed.train_online()
    assert env.steps == 0 and env.closed
    assert calls == []
    assert read_status(resumed) == original


@pytest.mark.parametrize("budget", [0, 7])
def test_no_complete_rollout_does_not_save_untrained_state(make_workspace, budget):
    work, env, calls = make_workspace(budget=budget)
    work.train_online()
    assert env.steps == 0 and env.closed and calls == []
    assert not (work.work_dir / "checkpoint").exists()


def test_nonzero_rank_also_saves_its_final_buffer(make_workspace):
    work, _, calls = make_workspace(budget=53, rank=7)
    work.train_online()
    assert [entry["global_time"] for entry in calls] == [48]
    assert (work.work_dir / "checkpoint/buffers/train_rank_7/buffer.hdf5").is_file()
    assert not (work.work_dir / "checkpoint/train_status.json").exists()


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_failed_update_does_not_replace_previous_valid_checkpoint(make_workspace, failure):
    work, _, calls = make_workspace(budget=53, checkpoint_every=16)
    work.agent.fail_on_update = 2
    work.agent.failure = failure
    error = RuntimeError if failure == "exception" else FloatingPointError
    with pytest.raises(error, match="synthetic update failure|Non-finite"):
        work.train_online()
    assert [entry["global_time"] for entry in calls] == [16]
    assert read_status(work)["optimizer_steps"] == 0
    assert json.loads((work.work_dir / "checkpoint/tiny_agent.json").read_text())["updates"] == 0

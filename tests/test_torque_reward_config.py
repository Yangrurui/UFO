"""CPU-only configuration checks; no environment, policy, or worker is built."""

from __future__ import annotations

import inspect
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import _compose_humanoidverse_config
from humanoidverse.train import build_ufo_mjlab_config, parse_args
from humanoidverse.training.workspace import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "configs/robots/adam_sp_29dof_ufo_mechanical_dr.yaml"
TORQUE_REWARDS = ("penalty_torques", "limits_torque")
TORQUE_OVERRIDES = (
    "rewards.torque_reward_mode=",
    "rewards.reward_scales.penalty_torques=",
    "rewards.reward_scales.limits_torque=",
)


def build_config(**overrides):
    params = dict(
        device="cpu",
        work_dir="/tmp/ufo_torque_reward_config_test",
        num_envs=1,
        num_env_steps=1,
        seed=4728,
        use_wandb=False,
        wandb_run_name=None,
        distributed_world_size=8,
        robot_config=ROBOT,
    )
    params.update(overrides)
    return build_ufo_mjlab_config(**params)


def compose_env_config(cfg):
    values = cfg.env.model_dump()
    arguments = {
        name: values[name]
        for name in inspect.signature(_compose_humanoidverse_config).parameters
        if name != "num_envs"
    }
    composed, _ = _compose_humanoidverse_config(num_envs=cfg.online_parallel_envs, **arguments)
    return composed


@pytest.fixture(scope="module")
def legacy_config():
    return build_config()


def without_torque_options(cfg):
    values = deepcopy(cfg.model_dump(mode="json"))
    for name in TORQUE_REWARDS:
        del values["agent"]["aux_rewards_scaling"][name]
    values["env"]["hydra_overrides"] = [
        value for value in values["env"]["hydra_overrides"]
        if not value.startswith(TORQUE_OVERRIDES)
    ]
    return values


def test_default_keeps_legacy_environment_and_disabled_learner_rewards(legacy_config):
    for name in TORQUE_REWARDS:
        assert name in legacy_config.agent.aux_rewards
        assert legacy_config.agent.aux_rewards_scaling[name] == 0.0
    assert not any(value.startswith(TORQUE_OVERRIDES) for value in legacy_config.env.hydra_overrides)
    rewards = compose_env_config(legacy_config).rewards
    assert rewards.torque_reward_mode == "legacy"
    # Existing YAML scales are intentionally distinct from the disabled learner weights.
    assert rewards.reward_scales.penalty_torques == -1e-6
    assert rewards.reward_scales.limits_torque == -5.0
    assert build_config(torque_penalty_weight=None, torque_limit_weight=None) == legacy_config


def test_explicit_weights_reach_learner_hydra_and_serialized_config(legacy_config):
    cfg = build_config(torque_penalty_weight=-1e-6, torque_limit_weight=-0.02)
    actual = compose_env_config(cfg)
    assert actual.rewards.torque_reward_mode == "dynamic_substep"
    for name, expected in zip(TORQUE_REWARDS, (-1e-6, -0.02)):
        assert cfg.agent.aux_rewards_scaling[name] == expected
        assert actual.rewards.reward_scales[name] == expected
    restored = TrainConfig.model_validate_json(cfg.model_dump_json())
    assert restored.model_dump(mode="json") == cfg.model_dump(mode="json")
    assert without_torque_options(cfg) == without_torque_options(legacy_config)
    assert cfg.agent.train.reg_coeff == 0.05
    assert cfg.agent.train.reg_coeff_aux == 0.02
    assert cfg.agent.train.scale_reg is True
    assert cfg.agent.model.norm_aux_reward == legacy_config.agent.model.norm_aux_reward

    old_rewards = OmegaConf.to_container(compose_env_config(legacy_config).rewards, resolve=True)
    new_rewards = OmegaConf.to_container(actual.rewards, resolve=True)
    for rewards in (old_rewards, new_rewards):
        del rewards["torque_reward_mode"]
        for name in TORQUE_REWARDS:
            del rewards["reward_scales"][name]
    assert old_rewards == new_rewards


def test_explicit_zero_enables_new_signals_but_keeps_both_learner_costs_zero(legacy_config):
    cfg = build_config(torque_penalty_weight=0.0, torque_limit_weight=0.0)
    rewards = compose_env_config(cfg).rewards
    assert rewards.torque_reward_mode == "dynamic_substep"
    for name in TORQUE_REWARDS:
        assert cfg.agent.aux_rewards_scaling[name] == 0.0
        assert rewards.reward_scales[name] == 0.0
    assert without_torque_options(cfg) == without_torque_options(legacy_config)


@pytest.mark.parametrize(
    ("argument", "reward", "other_reward", "weight"),
    [
        ("torque_penalty_weight", "penalty_torques", "limits_torque", -1e-6),
        ("torque_limit_weight", "limits_torque", "penalty_torques", -0.02),
    ],
)
def test_single_explicit_weight_leaves_other_learner_reward_disabled(argument, reward, other_reward, weight):
    cfg = build_config(**{argument: weight})
    rewards = compose_env_config(cfg).rewards
    assert rewards.torque_reward_mode == "dynamic_substep"
    assert cfg.agent.aux_rewards_scaling[reward] == weight
    assert rewards.reward_scales[reward] == weight
    assert cfg.agent.aux_rewards_scaling[other_reward] == 0.0


@pytest.mark.parametrize(
    ("argument", "reward"),
    [("torque_penalty_weight", "penalty_torques"), ("torque_limit_weight", "limits_torque")],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.01])
def test_nonfinite_and_positive_weights_are_rejected(argument, reward, value):
    with pytest.raises(ValueError, match=rf"{reward} weight must be finite and non-positive"):
        build_config(**{argument: value})


def test_cli_defaults_and_explicit_values_reach_configuration():
    with patch.object(sys, "argv", ["train.py", "--robot-config", str(ROBOT)]):
        default = parse_args()
    assert default.torque_penalty_weight is None
    assert default.torque_limit_weight is None
    with patch.object(sys, "argv", [
        "train.py", "--robot-config", str(ROBOT),
        "--torque-penalty-weight=-1e-6", "--torque-limit-weight=-0.02",
    ]):
        requested = parse_args()
    cfg = build_config(
        torque_penalty_weight=requested.torque_penalty_weight,
        torque_limit_weight=requested.torque_limit_weight,
    )
    assert cfg.agent.aux_rewards_scaling["penalty_torques"] == -1e-6
    assert cfg.agent.aux_rewards_scaling["limits_torque"] == -0.02

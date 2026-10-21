"""CPU physics checks for opt-in torque rewards, including native PD and delay."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.actuator import BuiltinPdActuator
from mjlab.envs import ManagerBasedRlEnv
from omegaconf import OmegaConf

from humanoidverse.agents.envs.actuators.torque_speed import TorqueSpeedEnvelopeActuator
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabCore, _compose_humanoidverse_config, make_mjlab_ufo_env_cfg,
)
from humanoidverse.agents.envs.torque_rewards import TorqueRewardCollector, torque_reward_mode
from humanoidverse.train import build_ufo_mjlab_config

ROOT = Path(__file__).resolve().parents[1]


def build_cpu_core(monkeypatch, mode, dr):
    creation = build_ufo_mjlab_config(
        device="cpu", work_dir="/tmp/ufo_torque_reward_test", num_envs=2,
        num_env_steps=4096, seed=723, use_wandb=False, wandb_run_name=None,
        disable_dr=not dr,
        robot_config=str(ROOT / "configs/robots/adam_sp_29dof_ufo_mechanical_dr.yaml"),
    ).env
    values = creation.model_dump()
    values["hydra_overrides"].append("rewards.torque_reward_mode=" + mode)
    params = {k: values[k] for k in inspect.signature(_compose_humanoidverse_config).parameters if k != "num_envs"}
    hv, _ = _compose_humanoidverse_config(num_envs=2, **params)
    native = make_mjlab_ufo_env_cfg(hv, num_envs=2, seed=723, mjcf_path=creation.mjcf_path,
                                  auto_reset=False, robot_training=creation.robot_training)
    # Motion files are unrelated to torque generation. Use the real default
    # state, actions, delay buffers, randomized model, physics and core step.
    monkeypatch.setattr(HumanoidVerseMjlabCore, "_init_motion_lib", lambda self: None)
    env = ManagerBasedRlEnv(native, device="cpu")
    env.reset()
    core = HumanoidVerseMjlabCore(hv, env, creation_config=creation)
    return core


@pytest.mark.parametrize("mode,dr", [("legacy", False), ("dynamic_substep", False), ("dynamic_substep", True)])
def test_actual_cpu_substeps_match_independent_pd_and_limits(monkeypatch, mode, dr):
    torch.set_num_threads(1)
    core = build_cpu_core(monkeypatch, mode, dr)
    env, robot = core.mjlab_env, core.robot
    dynamic = mode == "dynamic_substep"
    assert ("torque_saturation" in env.metrics_manager.active_terms) == dynamic
    collector = core._torque_rewards
    if dynamic:
        assert collector.substeps == 0 and not collector.active
        if not dr:
            # Reject configurations where native actuator_force is no longer
            # the unclipped joint PD demand. Use the actual compiled model.
            native = next(a for a in robot.actuators if isinstance(a, BuiltinPdActuator))
            act_id = int(native.global_ctrl_ids[0])
            model = env.sim.mj_model
            original_gear = model.actuator_gear[act_id].copy()
            model.actuator_gear[act_id, 0] = 2.
            with pytest.raises(ValueError, match="unit joint gear"):
                TorqueRewardCollector(None, env)
            model.actuator_gear[act_id] = original_gear
            model.actuator_forcelimited[act_id] = True
            with pytest.raises(ValueError, match="individual force"):
                TorqueRewardCollector(None, env)
            model.actuator_forcelimited[act_id] = False
            assert env._ufo_torque_rewards is collector
    commands = {}
    for act in robot.actuators:
        if isinstance(act, TorqueSpeedEnvelopeActuator):
            original = act.compute
            def observed_command(cmd, act=act, original=original):
                commands[act] = (cmd.position_target.clone(), cmd.velocity_target.clone(),
                                 cmd.effort_target.clone(), cmd.pos.clone(), cmd.vel.clone())
                return original(cmd)
            monkeypatch.setattr(act, "compute", observed_command)
    original_step = env.sim.step
    samples = []
    delayed_difference = []
    native_cancellation = []
    def observed_step():
        raw = torch.empty_like(robot.data.joint_pos)
        clipped = torch.empty_like(raw)
        q, dq = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
        for act in robot.actuators:
            local = act.target_ids
            if isinstance(act, TorqueSpeedEnvelopeActuator):
                target, dtarget, ff, pos, vel = commands[act]
                demand = act.stiffness * (target - pos) + act.damping * (dtarget - vel) + ff
                raw[:, local] = demand
                if dynamic:
                    torch.testing.assert_close(act.raw_torque, demand)
                delayed_difference.append(float((target - robot.data.joint_pos_target[:, local]).abs().max()))
                expected = np.empty_like(demand.numpy())
                for i in range(demand.shape[0]):
                    for j in range(demand.shape[1]):
                        v, tau = float(vel[i, j]), float(demand[i, j])
                        side = ("forward_motoring" if v >= 0 else "reverse_braking") if tau >= 0 else (
                            "reverse_motoring" if v <= 0 else "forward_braking")
                        limit = np.interp(abs(v), getattr(act.cfg, side + "_velocity"),
                                          getattr(act.cfg, side + "_effort"))
                        limit = min(limit, float(act.force_limit[i, j]))
                        expected[i, j] = np.clip(tau, -limit, limit)
                clipped[:, local] = torch.from_numpy(expected)
            else:
                assert isinstance(act, BuiltinPdActuator)
                ids, n = act.global_ctrl_ids, act.num_targets
                ctrl = env.sim.data.ctrl[:, ids]
                gain = env.sim.model.actuator_gainprm[:, ids, 0]
                bias = env.sim.model.actuator_biasprm[:, ids]
                position = torch.cat((q[:, local], q[:, local]), dim=1)
                velocity = torch.cat((dq[:, local], dq[:, local]), dim=1)
                force = gain * ctrl + bias[:, :, 0] + bias[:, :, 1] * position + bias[:, :, 2] * velocity
                demand = force[:, :n] + force[:, n:]
                native_cancellation.append(float((force.abs().sum(1) - demand.abs().sum(1)).max()))
                raw[:, local] = demand
                clipped[:, local] = demand.clamp(-act.cfg.effort_limit, act.cfg.effort_limit)
        original_step()
        actual = robot.data.qfrc_actuator.clone()
        torch.testing.assert_close(actual, clipped, atol=.005, rtol=2e-5)
        samples.append((raw, actual))
    monkeypatch.setattr(env.sim, "step", observed_step)
    try:
        caps = torch.zeros(len(robot.joint_names))
        for act in robot.actuators:
            caps[act.target_ids] = act.cfg.effort_limit
        if dr:
            explicit = next(a for a in robot.actuators if isinstance(a, TorqueSpeedEnvelopeActuator))
            assert not torch.allclose(explicit.stiffness, torch.full_like(explicit.stiffness, explicit.cfg.stiffness))
        for step in range(4):
            # Alternation makes any command lag observable; envs get opposite targets.
            action = torch.full((2, core.num_dof), .65 if step % 2 == 0 else -.4)
            action[1].neg_()
            samples.clear()
            _, _, done, info = core.step(action)
            assert not done.any() and len(samples) == 4
            aux = info["aux_rewards"]
            if dynamic:
                power = torch.stack([actual.square().sum(1) for _, actual in samples]).mean(0)
                saturation = torch.stack([((raw - actual).abs() / caps).sum(1) for raw, actual in samples]).mean(0)
                torch.testing.assert_close(aux["penalty_torques"], power)
                torch.testing.assert_close(aux["limits_torque"], saturation, atol=1e-5, rtol=1e-5)
                assert (saturation > 0).any()
                assert collector.substeps == 4 and not collector.active
                before = collector.values.clone()
                env.sim.forward()
                torch.testing.assert_close(collector.values, before)
            else:
                torch.testing.assert_close(aux["penalty_torques"], core.torques.square().sum(1))
                expected = (core.torques.abs() - .95 * core.torque_limits).clamp_min(0).sum(1)
                torch.testing.assert_close(aux["limits_torque"], expected)
                assert all(a.raw_torque is None for a in robot.actuators if isinstance(a, TorqueSpeedEnvelopeActuator))
            if step == 1:
                # Reset one environment between steps; the next reward must
                # still contain exactly four samples for each environment.
                env.reset(env_ids=torch.tensor([0]))
                core._refresh_state()
        if dr:
            assert max(delayed_difference) > .01
        assert max(native_cancellation) > 0.01
    finally:
        env.close()


def test_native_cancellation_middle_substep_and_step_reset():
    # A native P/D pair may individually exceed the cap yet cancel completely.
    collector = object.__new__(TorqueRewardCollector)
    act = SimpleNamespace(num_targets=1, ctrl_ids=torch.tensor([0, 1]), target_ids=torch.tensor([0]))
    data = SimpleNamespace(actuator_force=torch.tensor([[30., -30.], [30., 0.]]),
                           qfrc_actuator=torch.tensor([[0.], [10.]]))
    collector.robot = SimpleNamespace(data=data)
    collector.raw = torch.zeros(2, 1)
    collector.nominal_caps = torch.tensor([10.])
    collector.accum = torch.zeros(2, 2)
    collector.values = torch.zeros(2, 2)
    collector.envelopes, collector.native = [], [act]
    collector.expected_substeps = 4
    collector.begin_step()
    for substep in range(4):
        # Only the second substep saturates env 1. The final sample is zero.
        data.actuator_force[1, 0] = 30. if substep == 1 else 0.
        data.qfrc_actuator[1, 0] = 10. if substep == 1 else 0.
        collector(None)
    collector.finish_step()
    torch.testing.assert_close(collector.values, torch.tensor([[0., 0.], [25., .5]]))
    collector.begin_step()
    for _ in range(4):
        collector(None)
    collector.finish_step()
    torch.testing.assert_close(collector.values, torch.zeros(2, 2))


def test_unknown_mode_rejected_and_absent_mode_is_legacy():
    assert torque_reward_mode(OmegaConf.create({"rewards": {}})) == "legacy"
    with pytest.raises(ValueError, match="torque_reward_mode"):
        torque_reward_mode(OmegaConf.create({"rewards": {"torque_reward_mode": "typo"}}))

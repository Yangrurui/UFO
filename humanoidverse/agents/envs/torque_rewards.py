"""Opt-in substep torque rewards using the forces actually computed by MJLab."""

import math

import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPdActuator

from .actuators.torque_speed import TorqueSpeedEnvelopeActuator


def torque_reward_mode(config) -> str:
    mode = str(config.rewards.get("torque_reward_mode", "legacy"))
    if mode not in ("legacy", "dynamic_substep"):
        raise ValueError(f"Unknown rewards.torque_reward_mode: {mode}")
    return mode


class TorqueRewardCollector:
    """A per-substep metric, with control-step buffers owned outside metric resets.

    Outputs are mean(sum(actual**2)) in Nm^2 and
    mean(sum(abs(raw - actual) / nominal_static_cap)), dimensionless.

    Native actuator_force is the force used by the just-completed physics step,
    before the joint sum clamp. Its paired position/velocity outputs already
    include command delay and randomized gains. Do not reconstruct it from the
    now-integrated qpos/qvel, or sample again after the final sim.forward().
    """

    def __init__(self, cfg, env):
        del cfg
        self.robot = env.scene["robot"]
        self.expected_substeps = int(env.cfg.decimation)
        model = env.sim.mj_model
        njoints = len(self.robot.joint_names)
        self.raw = torch.zeros(env.num_envs, njoints, device=env.device)
        self.nominal_caps = torch.zeros(njoints, device=env.device)
        self.accum = torch.zeros(env.num_envs, 2, device=env.device)
        self.values = torch.zeros_like(self.accum)
        self.substeps = 0
        self.active = False
        self.envelopes = []
        self.native = []
        covered = set()
        for act in self.robot.actuators:
            if not isinstance(act, (TorqueSpeedEnvelopeActuator, BuiltinPdActuator)):
                raise ValueError("dynamic_substep torque rewards require envelope or native paired PD actuators")
            targets = act.target_ids.tolist()
            if covered.intersection(targets):
                raise ValueError("dynamic_substep torque rewards require one actuator group per joint")
            covered.update(targets)
            ids = act.global_ctrl_ids.cpu().numpy()
            if (not np.all(model.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_JOINT)
                    or not np.all(model.actuator_gear[ids] == [1., 0., 0., 0., 0., 0.])):
                raise ValueError("dynamic_substep torque rewards require fixed unit joint gear")
            joints = model.actuator_trnid[ids, 0]
            expected_joints = self.robot.indexing.joint_ids[act.target_ids].cpu().numpy()
            expected_joints = np.tile(expected_joints, 2 if isinstance(act, BuiltinPdActuator) else 1)
            if not np.array_equal(joints, expected_joints) or np.any(model.jnt_actgravcomp[joints]):
                raise ValueError("dynamic_substep torque rewards require matching joint transmissions without actuator gravity compensation")
            cap = act.cfg.effort_limit
            if cap is None or not math.isfinite(float(cap)) or float(cap) <= 0:
                raise ValueError("dynamic_substep torque rewards require positive nominal static effort limits")
            self.nominal_caps[act.target_ids] = float(cap)
            if isinstance(act, TorqueSpeedEnvelopeActuator):
                act.enable_torque_recording()
                self.envelopes.append(act)
            else:
                if np.any(model.actuator_forcelimited[ids]) or np.any(model.actuator_ctrllimited[ids]):
                    raise ValueError("native PD torque rewards require no individual force or target caps")
                self.native.append(act)
        if covered != set(range(njoints)):
            raise ValueError("dynamic_substep torque rewards must cover every robot joint")
        env._ufo_torque_rewards = self

    def begin_step(self) -> None:
        self.accum.zero_()
        self.substeps = 0
        self.active = True

    def __call__(self, env) -> torch.Tensor:
        del env
        if not self.active:
            return self.values[:, 1] * 0.0
        actual = self.robot.data.qfrc_actuator
        for act in self.envelopes:
            self.raw[:, act.target_ids] = act.raw_torque
        if self.native:
            forces = self.robot.data.actuator_force
            for act in self.native:
                n = act.num_targets
                # Sum P and D before measuring clipping; they may cancel.
                self.raw[:, act.target_ids] = (forces[:, act.ctrl_ids[:n]]
                                               + forces[:, act.ctrl_ids[n:]])
        squared = actual.square().sum(dim=1)
        excess = ((self.raw - actual).abs() / self.nominal_caps).sum(dim=1)
        self.accum[:, 0].add_(squared)
        self.accum[:, 1].add_(excess)
        self.substeps += 1
        return excess

    def finish_step(self) -> None:
        self.active = False
        if self.substeps != self.expected_substeps:
            raise RuntimeError(f"Expected {self.expected_substeps} torque samples, got {self.substeps}")
        self.values.copy_(self.accum / self.substeps)

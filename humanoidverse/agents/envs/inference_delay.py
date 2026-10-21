"""Whole-body inference latency, scheduled on the physics-step clock."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _integer_steps(duration: float, dt: float, name: str) -> int:
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    steps = duration / dt
    if not math.isfinite(steps) or not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be an integer number of physics steps")
    return int(round(steps))


def _env_indices(env_ids, device):
    if env_ids is None:
        return slice(None)
    if isinstance(env_ids, slice):
        return env_ids
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    if ids.ndim != 1:
        raise ValueError("env_ids must be a one-dimensional sequence")
    return ids


class InferenceDelayBuffer:
    """Schedule one pending policy command per environment on a physics clock.

    Submit once per policy period; apply once at the beginning of each physics
    step. Every submission samples a new integer delay per environment, shared
    across all command dimensions. Delays cannot exceed one policy period, so
    only one pending command is needed. The previous command remains active
    until the new one arrives. Reset uses a holding command, not a policy output.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        command_dim: int,
        inference_delay_range_ms: tuple[float, float],
        physics_dt: float,
        policy_dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics_dt must be finite and positive")
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError("policy_dt must be finite and positive")
        self.control_steps = _integer_steps(policy_dt, physics_dt, "policy_dt")
        if self.control_steps < 1:
            raise ValueError("policy_dt must cover at least one physics step")
        if len(inference_delay_range_ms) != 2:
            raise ValueError("inference_delay_range_ms must contain two endpoints")
        self.min_delay_steps, self.max_delay_steps = (
            _integer_steps(float(value) / 1000.0, physics_dt, "inference delay") for value in inference_delay_range_ms
        )
        if self.min_delay_steps > self.max_delay_steps:
            raise ValueError("inference delay range must be ordered")
        if self.max_delay_steps > self.control_steps:
            raise ValueError("inference delay cannot exceed the policy control period")
        if num_envs <= 0 or command_dim <= 0:
            raise ValueError("num_envs and command_dim must be positive")

        self.num_envs = num_envs
        self.command_dim = command_dim
        self._active = torch.empty((num_envs, command_dim), device=device, dtype=dtype)
        self._pending = torch.empty_like(self._active)
        self._remaining = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.sampled_delay_steps = torch.zeros_like(self._remaining)
        self._initialized = False

    def _check_command(self, command: torch.Tensor, num_envs: int) -> None:
        if command.shape != (num_envs, self.command_dim):
            raise ValueError(f"Expected command shape {(num_envs, self.command_dim)}, got {tuple(command.shape)}")
        if command.device != self._active.device or command.dtype != self._active.dtype:
            raise ValueError("Command dtype and device must match the inference delay buffer")

    @torch.no_grad()
    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | slice | None, initial_command: torch.Tensor
    ) -> None:
        """Cancel pending commands and hold selected rows; first reset covers all."""
        ids = _env_indices(env_ids, self._active.device)
        num_envs = len(range(*ids.indices(self.num_envs))) if isinstance(ids, slice) else ids.numel()
        self._check_command(initial_command, num_envs)
        self._active[ids] = initial_command
        self._pending[ids] = initial_command
        self._remaining[ids] = -1
        self.sampled_delay_steps[ids] = 0
        if num_envs == self.num_envs:
            self._initialized = True

    def _release_due(self) -> None:
        self._active.copy_(torch.where((self._remaining == 0).unsqueeze(-1), self._pending, self._active))

    @torch.no_grad()
    def submit(self, command: torch.Tensor) -> None:
        """Enqueue a processed policy command without changing its input tensor."""
        self._check_command(command, self.num_envs)
        if not self._initialized:
            raise RuntimeError("Reset all environments with a holding command before submitting")
        # A command delayed exactly one policy period is due at this boundary.
        # Promote it before replacing the pending command.
        self._release_due()
        self._pending.copy_(command)
        self.sampled_delay_steps.random_(self.min_delay_steps, self.max_delay_steps + 1)
        self._remaining.copy_(self.sampled_delay_steps)

    @torch.no_grad()
    def apply(self) -> torch.Tensor:
        """Return the command for the current substep, then advance one tick."""
        if not self._initialized:
            raise RuntimeError("Reset all environments with a holding command before applying")
        self._release_due()
        self._remaining.sub_(1).clamp_min_(-1)
        return self._active.clone()


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    inference_delay_range_ms: tuple[float, float]

    def build(self, env: ManagerBasedRlEnv) -> DelayedJointPositionAction:
        return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
    """Native joint-position action with shared inference-command latency.

    Scaling, offsets, clipping, and raw action history retain native semantics.
    Only processed targets wait for inference completion; encoder bias is still
    subtracted at application time, and local actuator feedback is not delayed.
    """

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.inference_delay = InferenceDelayBuffer(
            num_envs=self.num_envs,
            command_dim=self.action_dim,
            inference_delay_range_ms=cfg.inference_delay_range_ms,
            physics_dt=float(env.physics_dt),
            policy_dt=float(env.step_dt),
            device=self.device,
            dtype=self._processed_actions.dtype,
        )
        self.reset_inference_delay(None, self._entity.data.joint_pos[:, self._target_ids])

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self.inference_delay.submit(self._processed_actions)

    def apply_actions(self) -> None:
        target = self.inference_delay.apply() - self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

    def reset_inference_delay(
        self,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        physical_hold_target_in_action_order: torch.Tensor,
    ) -> None:
        """Hold physical joint positions and discard selected pending commands.

        Call after writing a reset pose, before writing actuator data to the
        simulator. Positions have selected-env rows and action-term joint order.
        This method also immediately sets the physical hold target.
        """
        ids = _env_indices(env_ids, self._raw_actions.device)
        bias = self._entity.data.encoder_bias[ids][:, self._target_ids]
        self.inference_delay.reset(ids, physical_hold_target_in_action_order + bias)
        # Entity's setter uses direct advanced indexing, so tensor row IDs
        # need a column axis to select every target joint in every reset env.
        target_env_ids = ids[:, None] if isinstance(ids, torch.Tensor) else ids
        self._entity.set_joint_position_target(
            physical_hold_target_in_action_order, joint_ids=self._target_ids, env_ids=target_env_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = _env_indices(env_ids, self._raw_actions.device)
        self.reset_inference_delay(ids, self._entity.data.joint_pos[ids][:, self._target_ids])

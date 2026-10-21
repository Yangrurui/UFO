"""CPU checks for substep inference latency, native actions, and motor delay."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.utils.buffers import DelayBuffer

from humanoidverse.agents.envs.inference_delay import (
    DelayedJointPositionAction,
    DelayedJointPositionActionCfg,
    InferenceDelayBuffer,
)


def make_buffer(delay_range=(10.0, 20.0), num_envs=2, command_dim=3):
    result = InferenceDelayBuffer(
        num_envs=num_envs,
        command_dim=command_dim,
        inference_delay_range_ms=delay_range,
        physics_dt=0.005,
        policy_dt=0.02,
        device="cpu",
    )
    result.reset(None, torch.full((num_envs, command_dim), -1.0))
    return result


@pytest.mark.parametrize("lag", [2, 3, 4])
def test_exact_substep_arrival_and_20ms_boundary_never_drops_old_command(lag):
    delay = make_buffer((lag * 5.0, lag * 5.0))
    outputs = []
    for value in (10.0, 20.0, 30.0):
        delay.submit(torch.full((2, 3), value))
        for _ in range(4):
            outputs.append(delay.apply())
    for tick, output in enumerate(outputs):
        expected = -1.0 if tick < lag else 10.0 if tick < 4 + lag else 20.0 if tick < 8 + lag else 30.0
        torch.testing.assert_close(output, torch.full((2, 3), expected))


def test_each_command_resamples_per_environment_shared_by_all_29_joints():
    torch.manual_seed(721)
    delay = make_buffer(num_envs=256, command_dim=29)
    command = torch.arange(29, dtype=torch.float32).expand(256, 29).clone()
    delay.submit(command)
    first_lags = delay.sampled_delay_steps.clone()
    assert set(first_lags.tolist()) == {2, 3, 4}
    for tick in range(4):
        output = delay.apply()
        expected = torch.where((first_lags <= tick).unsqueeze(-1), command, torch.full_like(command, -1.0))
        torch.testing.assert_close(output, expected)
    delay.submit(command + 100)
    second_lags = delay.sampled_delay_steps.clone()
    assert set(second_lags.tolist()) == {2, 3, 4}
    assert torch.any(second_lags != first_lags)
    for tick in range(4):
        expected = torch.where((second_lags <= tick).unsqueeze(-1), command + 100, command)
        torch.testing.assert_close(delay.apply(), expected)


def test_partial_reset_cancels_pending_command_only_for_selected_environment():
    delay = make_buffer((15.0, 15.0))
    delay.submit(torch.tensor([[1.0] * 3, [2.0] * 3]))
    delay.apply()
    delay.apply()
    delay.reset([1], torch.full((1, 3), -9.0))
    torch.testing.assert_close(delay.apply(), torch.tensor([[-1.0] * 3, [-9.0] * 3]))
    torch.testing.assert_close(delay.apply(), torch.tensor([[1.0] * 3, [-9.0] * 3]))
    delay.submit(torch.tensor([[3.0] * 3, [4.0] * 3]))
    for _ in range(3):
        torch.testing.assert_close(delay.apply(), torch.tensor([[1.0] * 3, [-9.0] * 3]))
    torch.testing.assert_close(delay.apply(), torch.tensor([[3.0] * 3, [4.0] * 3]))


def test_input_hold_and_returned_commands_do_not_alias_internal_state():
    delay = make_buffer((10.0, 10.0))
    hold = torch.full((2, 3), -5.0)
    delay.reset(None, hold)
    hold.fill_(99.0)
    command = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    original = command.clone()
    delay.submit(command)
    torch.testing.assert_close(command, original)
    command.fill_(88.0)
    result = delay.apply()
    torch.testing.assert_close(result, torch.full((2, 3), -5.0))
    result.fill_(77.0)
    torch.testing.assert_close(delay.apply(), torch.full((2, 3), -5.0))
    torch.testing.assert_close(delay.apply(), original)


@pytest.mark.parametrize("delay_range", [(-1.0, 20.0), (10.0, 17.0), (20.0, 10.0), (10.0, 25.0), (float("nan"), 20.0)])
def test_rejects_negative_fractional_reversed_or_overlapping_policy_period_delays(delay_range):
    with pytest.raises(ValueError):
        make_buffer(delay_range)


@pytest.mark.parametrize("physics_dt,policy_dt", [(0.0, 0.02), (-0.005, 0.02), (0.005, 0.0), (0.005, 0.023)])
def test_rejects_invalid_clocks(physics_dt, policy_dt):
    with pytest.raises(ValueError):
        InferenceDelayBuffer(
            num_envs=2,
            command_dim=3,
            inference_delay_range_ms=(10.0, 20.0),
            physics_dt=physics_dt,
            policy_dt=policy_dt,
            device="cpu",
        )


class FakeEntity:
    def __init__(self):
        self.data = SimpleNamespace(
            joint_pos=torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]]),
            default_joint_pos=torch.tensor([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3]]),
            encoder_bias=torch.tensor([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]]),
            joint_pos_target=torch.zeros(2, 3),
        )

    def find_joints_by_actuator_names(self, names):
        return [2, 0, 1], ["j2", "j0", "j1"]

    @property
    def _data(self):
        return self.data

    # Exercise the actual setter: it does not expand 1-D environment indices.
    set_joint_position_target = Entity.set_joint_position_target


def make_action(delay_range=(10.0, 20.0), native=False):
    entity = FakeEntity()
    env = SimpleNamespace(num_envs=2, device="cpu", physics_dt=0.005, step_dt=0.02, scene={"robot": entity})
    kwargs = dict(
        entity_name="robot",
        actuator_names=(".*",),
        scale={"j0": 2.0, "j1": 3.0, "j2": 0.5},
        clip={".*": (-0.8, 0.8)},
    )
    cfg = JointPositionActionCfg(**kwargs) if native else DelayedJointPositionActionCfg(**kwargs, inference_delay_range_ms=delay_range)
    return cfg.build(env), entity


def test_native_action_integration_preserves_affine_clip_bias_and_raw_actions():
    action, entity = make_action((10.0, 10.0))
    assert isinstance(action, DelayedJointPositionAction)
    physical_hold = entity.data.joint_pos.clone()
    command = torch.tensor([[3.0, -0.3, 0.1], [-3.0, 0.4, -0.1]])
    original = command.clone()
    action.process_actions(command)
    processed = torch.clamp(command * torch.tensor([0.5, 2.0, 3.0]) + torch.tensor([0.3, 0.1, -0.2]), -0.8, 0.8)
    for _ in range(2):
        action.apply_actions()
        torch.testing.assert_close(entity.data.joint_pos_target, physical_hold)
    action.apply_actions()
    torch.testing.assert_close(
        entity.data.joint_pos_target[:, [2, 0, 1]], processed - entity.data.encoder_bias[:, [2, 0, 1]]
    )
    torch.testing.assert_close(action.raw_action, original)
    torch.testing.assert_close(command, original)
    torch.testing.assert_close(action._processed_actions, processed)


def test_native_public_reset_holds_physical_pose_with_bias_and_cancels_only_selected_pending():
    action, entity = make_action((15.0, 15.0))
    action.process_actions(torch.ones(2, 3))
    action.apply_actions()
    action.apply_actions()
    hold = torch.tensor([[0.9, -0.7, 0.5]])  # Action order, not entity order.
    action.reset_inference_delay(torch.tensor([1]), hold)
    torch.testing.assert_close(entity.data.joint_pos_target[1, [2, 0, 1]], hold[0])
    action.apply_actions()
    action.apply_actions()
    torch.testing.assert_close(entity.data.joint_pos_target[1, [2, 0, 1]], hold[0])
    expected_other = action._processed_actions[0] - entity.data.encoder_bias[0, [2, 0, 1]]
    torch.testing.assert_close(entity.data.joint_pos_target[0, [2, 0, 1]], expected_other)


def test_native_manager_reset_zeros_raw_actions_and_holds_current_joint_positions():
    action, entity = make_action()
    action.process_actions(torch.ones(2, 3))
    entity.data.joint_pos[0] = torch.tensor([-0.1, 0.2, -0.3])
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.raw_action[0], torch.zeros(3))
    torch.testing.assert_close(action.raw_action[1], torch.ones(3))
    for _ in range(5):
        action.apply_actions()
        torch.testing.assert_close(entity.data.joint_pos_target[0], entity.data.joint_pos[0])


def test_zero_delay_matches_native_joint_position_action():
    delayed, delayed_entity = make_action((0.0, 0.0))
    native, native_entity = make_action(native=True)
    for command in (torch.ones(2, 3), -torch.ones(2, 3)):
        delayed.process_actions(command)
        native.process_actions(command)
        for _ in range(4):
            delayed.apply_actions()
            native.apply_actions()
            torch.testing.assert_close(delayed_entity.data.joint_pos_target, native_entity.data.joint_pos_target)


@pytest.mark.parametrize("inference_lag", [2, 3, 4])
@pytest.mark.parametrize("motor_lag", range(5))
@pytest.mark.parametrize("reset_target_write", [False, True])
def test_real_motor_buffer_adds_0_to_20ms_to_inference_10_to_20ms_from_first_action(
    inference_lag, motor_lag, reset_target_write
):
    inference = make_buffer((inference_lag * 5.0, inference_lag * 5.0), num_envs=1, command_dim=1)
    motor = DelayBuffer(
        min_lag=motor_lag,
        max_lag=motor_lag,
        batch_size=1,
        device="cpu",
        update_period=1_000_000,
        per_env_phase=False,
    )
    if reset_target_write:
        # Discard the zero-target write inside MJLab reset, then write hold.
        motor.append(torch.zeros(1, 1))
        motor.compute()
        motor.reset()
        motor.append(torch.tensor([[-1.0]]))
        motor.compute()
    outputs = []
    for value in (10.0, 20.0, 30.0, 40.0):
        inference.submit(torch.tensor([[value]]))
        for _ in range(4):
            motor.append(inference.apply())
            outputs.append(motor.compute().item())
    total_lag = inference_lag + motor_lag
    expected = [-1.0 if tick < total_lag else 10.0 * (1 + (tick - total_lag) // 4) for tick in range(16)]
    assert outputs == expected


@pytest.mark.parametrize("env_ids", [None, slice(None), torch.tensor([0, 1]), torch.tensor([1]), torch.tensor([1, 0])])
@pytest.mark.parametrize("joint_ids_as_list", [False, True])
def test_real_entity_setter_full_and_partial_reset_use_cartesian_indices(env_ids, joint_ids_as_list):
    # Two environments and three joints intentionally cannot pairwise broadcast.
    action, entity = make_action((15.0, 15.0))
    if joint_ids_as_list:
        action._target_ids = [2, 0, 1]
    ids = slice(None) if env_ids is None else env_ids
    num_rows = entity.data.joint_pos[ids].shape[0]
    hold = torch.arange(num_rows * 3, dtype=torch.float32).reshape(num_rows, 3) / 10 + 2
    expected = entity.data.joint_pos.clone()
    expected[ids] = hold[:, [1, 2, 0]]
    action.reset_inference_delay(env_ids, hold)
    torch.testing.assert_close(entity.data.joint_pos_target, expected)
    action.apply_actions()
    torch.testing.assert_close(entity.data.joint_pos_target, expected)

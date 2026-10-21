"""Measured motor limits, command delay and Adam/G1 configuration separation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from mjlab.actuator import ActuatorCmd, BuiltinPdActuator, BuiltinPdActuatorCfg, DcMotorActuatorCfg

from humanoidverse.agents.envs.actuators.config import build_grouped_actuators
from humanoidverse.agents.envs.actuators.torque_speed import TorqueSpeedEnvelopeActuatorCfg
from humanoidverse.agents.envs.actuators.torque_speed_envelope import clip_effort_by_envelope
from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _action_target_scale, _compose_humanoidverse_config, make_mjlab_ufo_env_cfg,
)

ROOT=Path(__file__).resolve().parents[1]
ROBOT=ROOT/'configs/robots/adam_sp_29dof.yaml'


def test_four_quadrants_and_breakpoints():
    # Different forward/reverse and motoring/braking limits expose swapped signs.
    speed=torch.tensor([-12.,-5.,0.,5.,12.])
    curves=[torch.tensor(x) for x in [
        [0.,10.],[20.,0.], [0.,10.],[30.,0.],
        [0.,20.],[40.,0.], [0.,20.],[60.,0.],
    ]]
    positive=clip_effort_by_envelope(torch.full_like(speed,100.),speed,*curves,torch.full_like(speed,100.))
    negative=clip_effort_by_envelope(torch.full_like(speed,-100.),speed,*curves,torch.full_like(speed,100.))
    torch.testing.assert_close(positive,torch.tensor([24.,45.,20.,10.,0.]))
    torch.testing.assert_close(negative,torch.tensor([0.,-15.,-30.,-30.,-16.]))


def delayed_motor(min_lag=4,max_lag=4,batch=2,joints=2):
    curves={}
    for quadrant in ('forward_motoring','reverse_motoring','forward_braking','reverse_braking'):
        curves[quadrant+'_velocity']=(0.,10.)
        curves[quadrant+'_effort']=(20.,0.)
    cfg=TorqueSpeedEnvelopeActuatorCfg(target_names_expr=tuple(f'j{i}' for i in range(joints)),
        stiffness=100.,damping=0.,effort_limit=20.,velocity_limit=10.,delay_min_lag=min_lag,
        delay_max_lag=max_lag,delay_update_period=1_000_000,delay_per_env_phase=False,**curves)
    act=cfg.build(SimpleNamespace(),list(range(joints)),list(cfg.target_names_expr))
    act._init_delay_buffer(batch,'cpu')
    act.stiffness=torch.full((batch,joints),100.);act.damping=torch.zeros(batch,joints)
    act.force_limit=torch.full((batch,joints),20.)
    for name,values in curves.items():setattr(act,name,torch.tensor(values))
    return act


def command(value,speed,batch=2,joints=2):
    return ActuatorCmd(position_target=torch.full((batch,joints),value),
        velocity_target=torch.full((batch,joints),value*2),effort_target=torch.zeros(batch,joints),
        pos=torch.zeros(batch,joints),vel=torch.full((batch,joints),speed))


def test_delay_uses_old_commands_current_feedback_and_isolated_reset():
    act=delayed_motor()
    for step in range(7):
        cmd=command(.1*(step+1),float(step));delayed=act.apply_delay(cmd)
        expected=.1*(max(0,step-4)+1)
        torch.testing.assert_close(delayed.position_target,torch.full((2,2),expected))
        torch.testing.assert_close(delayed.velocity_target,torch.full((2,2),2*expected))
        assert delayed.pos is cmd.pos and delayed.vel is cmd.vel
        torch.testing.assert_close(act.compute(delayed),torch.full((2,2),min(expected*100,20.-2*step)))
    act.reset(torch.tensor([0]))
    delayed=act.apply_delay(command(.99,9.))
    torch.testing.assert_close(delayed.position_target,torch.tensor([[.99,.99],[.4,.4]]))
    torch.testing.assert_close(act.compute(delayed),torch.full((2,2),2.))


def test_delays_cover_0_to_20ms_hold_per_episode_and_zero_delay():
    torch.manual_seed(42);act=delayed_motor(0,4,batch=1024)
    act.apply_delay(command(.1,0.,batch=1024));lags=act._delay_buffer.current_lags.clone()
    assert set(lags.tolist())==set(range(5))
    for _ in range(8):
        act.apply_delay(command(.2,0.,batch=1024))
        torch.testing.assert_close(act._delay_buffer.current_lags,lags)
    zero=delayed_motor(0,0);cmd=command(.2,9.)
    assert zero.apply_delay(cmd) is cmd
    torch.testing.assert_close(zero.compute(cmd),torch.full((2,2),2.))


def test_groups_match_independent_xlab_compiler_export():
    cfg=yaml.safe_load(ROBOT.read_text());a=cfg['training']['actuator'];names=cfg['control_joints']['names']
    oracle=json.loads((ROOT.parent/'adam_sp/xlab_adam_training_profile.json').read_text())
    groups=build_grouped_actuators(a['groups'],names,enable_delay=True,delay_step_range=(0,4),physics_dt=.005)
    assert len(groups)==10
    for current,original in zip(groups,oracle['groups']):
        assert list(current.target_names_expr)==original['selectors']
        for key,value in original['parameters'].items():
            if key=='delay_state_group':continue
            assert getattr(current,key)==(tuple(value) if isinstance(value,list) else value)
    assert sum(isinstance(g,TorqueSpeedEnvelopeActuatorCfg) for g in groups)==6
    assert sum(isinstance(g,BuiltinPdActuatorCfg) for g in groups)==4
    with pytest.raises(ValueError,match='5 ms'):
        build_grouped_actuators(a['groups'],names,enable_delay=True,delay_step_range=(0,4),physics_dt=.01)


def test_adam_action_scales_runtime_and_g1_baseline():
    # Build through the real configuration entry point; no policy or GPU needed.
    from humanoidverse.train import build_ufo_mjlab_config
    import inspect
    for name in ['adam_sp_29dof','g1_29dof']:
        built=build_ufo_mjlab_config(device='cuda:0',work_dir='/tmp/adam_config_test',num_envs=16,
            num_env_steps=4096,seed=4728,use_wandb=False,wandb_run_name=None,
            robot_config=str(ROOT/f'configs/robots/{name}.yaml')).env.model_dump()
        params={k:built[k] for k in inspect.signature(_compose_humanoidverse_config).parameters if k!='num_envs'}
        cfg,_=_compose_humanoidverse_config(num_envs=16,**params)
        mj=make_mjlab_ufo_env_cfg(cfg,num_envs=16,seed=4728,mjcf_path=built['mjcf_path'],
            auto_reset=True,robot_training=built['robot_training'])
        groups=mj.scene.entities['robot'].articulation.actuators
        if name.startswith('adam'):
            assert len(groups)==10 and 'random_pd_gains' in mj.events
            assert len(cfg.robot.motion.extend_config)==1
            assert cfg.robot.motion.extend_config[0].joint_name=='head_link'
            assert 'robot' not in cfg.robot
            assert not mj.scene.entities['robot'].sort_actuators
            from mjlab.entity import Entity
            entity=Entity(mj.scene.entities['robot'])
            for motor in entity.actuators:
                if isinstance(motor,BuiltinPdActuator):
                    n=motor.num_targets
                    channels=[a.name for a in motor._mjs_actuators]
                    assert all(x.endswith('_pd_pos') for x in channels[:n])
                    assert all(x.endswith('_pd_vel') for x in channels[n:])
            scales=_action_target_scale(cfg)
            for i,joint in enumerate(cfg.robot.dof_names):
                assert scales[i].item()==pytest.approx(mj.actions['actions'].scale[joint])
                if 'hip_roll' in joint or 'hip_yaw' in joint:assert scales[i].item()==pytest.approx(.2)
        else:
            assert len(groups)==29 and all(isinstance(g,DcMotorActuatorCfg) for g in groups)
            assert all(g.delay_max_lag==0 for g in groups) and 'random_pd_gains' not in mj.events


@pytest.mark.parametrize("keep_legacy_group_range", [True, False])
def test_domain_randomization_delay_range_drives_real_group_configs(keep_legacy_group_range):
    cfg = yaml.safe_load(ROBOT.read_text())
    groups = cfg["training"]["actuator"]["groups"]
    names = cfg["control_joints"]["names"]
    if not keep_legacy_group_range:
        for group in groups:
            group["parameters"].pop("delay_min_lag", None)
            group["parameters"].pop("delay_max_lag", None)
    built = build_grouped_actuators(groups, names, enable_delay=True, delay_step_range=(1, 3), physics_dt=0.005)
    assert len(built) == 10
    assert all((group.delay_min_lag, group.delay_max_lag) == (1, 3) for group in built)
    assert sum(isinstance(group, TorqueSpeedEnvelopeActuatorCfg) for group in built) == 6
    assert sum(isinstance(group, BuiltinPdActuatorCfg) for group in built) == 4
    disabled = build_grouped_actuators(groups, names, enable_delay=False, delay_step_range=(1, 3), physics_dt=0.005)
    assert all((group.delay_min_lag, group.delay_max_lag) == (0, 0) for group in disabled)
    # Building never mutates the source profile.
    if keep_legacy_group_range:
        assert all(
            (group["parameters"]["delay_min_lag"], group["parameters"]["delay_max_lag"]) == (0, 4)
            for group in groups
        )
    else:
        assert all("delay_min_lag" not in group["parameters"] and "delay_max_lag" not in group["parameters"] for group in groups)


@pytest.mark.parametrize("delay_step_range", [(-1, 4), (4, 1), (0, 2.5), (0,), (False, 4)])
def test_invalid_group_delay_ranges_are_rejected(delay_step_range):
    cfg = yaml.safe_load(ROBOT.read_text())
    with pytest.raises(ValueError, match="ordered, nonnegative integer"):
        build_grouped_actuators(
            cfg["training"]["actuator"]["groups"],
            cfg["control_joints"]["names"],
            enable_delay=True,
            delay_step_range=delay_step_range,
            physics_dt=0.005,
        )

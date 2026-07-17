# Adam SP 机器人迁移方案

> 目标：在 UFO 框架中用 BFM-Zero (FB-CPR) 算法训练 Adam SP 29-DOF 人形机器人
> 原则：只做机器人资产替换，不改奖励/终止/算法
> 运动数据：用户自有

---

## 一、Adam SP vs G1 关键差异

| 维度 | G1 | Adam SP |
|------|----|---------|
| DOF | 29 | 29 |
| 浮动基座 | pelvis / `floating_base_joint` | pelvis / `root` |
| 基座高度 | 0.793 m | **0.96** m (XML) / **0.89** m (站立 RESET) |
| 刚体数 | 30 | ≈54 (含手指视觉 body) |
| 腰关节顺序 | yaw→roll→pitch | **roll→pitch→yaw** |
| 腕关节顺序 | roll→pitch→yaw | **yaw→pitch→roll** |

### 执行器参数

| 关节组 | G1 kp/kd | Adam SP kp/kd | SP effort | SP armature |
|--------|---------|---------------|-----------|-------------|
| hip_pitch, knee | 99/6.3 | **300/7** | 230 | 0.13426 |
| hip_roll | 99/6.3 | **600/10** | 180 | 0.281573 |
| hip_yaw | 40/2.6 | **300/2** | 105 | 0.23409 |
| ankle_pitch | 28/1.8 | **130/3.5** | 80 | 0.1098 |
| ankle_roll | 28/1.8 | **70/2** | 40 | 0.1098 |
| waist | 300/5 | **400/11** | 150 | 0.23409 |
| shoulder | 14/0.9 | **60/3** | 65 | 0.01 |
| elbow | 14/0.9 | **60/3** | 30 | 0.01 |
| wrist | 14/0.5 | **20/1** | 6.4 | 0.01 |

### 初始姿态 (来自 InstinctMJ HOME_KEYFRAME)

```python
pos = (0, 0, 0.89)
.*_hip_pitch_joint: -0.32, .*_knee_joint: 0.66, .*_ankle_pitch_joint: -0.39
left_hip_yaw: -0.18, right_hip_yaw: 0.18
left_shoulder_roll: 0.1, right_shoulder_roll: -0.1, .*_elbow: -0.3
其余关节: 0.0
```

---

## 二、需要创建的文件

### 文件 1（新增）：修改后的 MuJoCo XML

**目标路径**：`humanoidverse/data/robots/adam_sp_mjlab/adam_sp.xml`

**源文件**：`/home/yangrr/codes/InstinctMJ/src/instinct_mj/assets/resources/adam_sp/xml/adam_sp.xml`

**必须修改**：

1. 删除所有 `<joint>` default class 中的 `actuatorfrcrange` 属性（16 处）
   - 代码依据：`humanoidverse_mjlab.py:393` `if "actuatorfrcrange" in xml_path.read_text(): raise ValueError(...)`
2. 删除所有 `<joint>` default class 中的 `armature` 属性
   - 代码依据：MJLab actuator 也设置 armature，MuJoCo 会叠加，导致 double-stack
3. 删除所有 `<joint>` default class 中的 `frictionloss` 属性
   - 同上原因
4. 删除所有 `<joint>` default class 中的 `damping` 属性
   - 同上原因
5. 修改 `meshdir="../meshes/"` → `meshdir="meshes"`
6. 在 `</worldbody>` 之后添加 29 个 motor 的 `<actuator>` 段：
   - 代码依据：`mujoco_parser.py:57` 要求 actuator_joint_names 非空

```xml
<actuator>
  <motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint"/>
  <motor name="left_hip_roll_joint" joint="left_hip_roll_joint"/>
  <motor name="left_hip_yaw_joint" joint="left_hip_yaw_joint"/>
  <motor name="left_knee_joint" joint="left_knee_joint"/>
  <motor name="left_ankle_pitch_joint" joint="left_ankle_pitch_joint"/>
  <motor name="left_ankle_roll_joint" joint="left_ankle_roll_joint"/>
  <motor name="right_hip_pitch_joint" joint="right_hip_pitch_joint"/>
  <motor name="right_hip_roll_joint" joint="right_hip_roll_joint"/>
  <motor name="right_hip_yaw_joint" joint="right_hip_yaw_joint"/>
  <motor name="right_knee_joint" joint="right_knee_joint"/>
  <motor name="right_ankle_pitch_joint" joint="right_ankle_pitch_joint"/>
  <motor name="right_ankle_roll_joint" joint="right_ankle_roll_joint"/>
  <motor name="waist_roll_joint" joint="waist_roll_joint"/>
  <motor name="waist_pitch_joint" joint="waist_pitch_joint"/>
  <motor name="waist_yaw_joint" joint="waist_yaw_joint"/>
  <motor name="left_shoulder_pitch_joint" joint="left_shoulder_pitch_joint"/>
  <motor name="left_shoulder_roll_joint" joint="left_shoulder_roll_joint"/>
  <motor name="left_shoulder_yaw_joint" joint="left_shoulder_yaw_joint"/>
  <motor name="left_elbow_joint" joint="left_elbow_joint"/>
  <motor name="left_wrist_yaw_joint" joint="left_wrist_yaw_joint"/>
  <motor name="left_wrist_pitch_joint" joint="left_wrist_pitch_joint"/>
  <motor name="left_wrist_roll_joint" joint="left_wrist_roll_joint"/>
  <motor name="right_shoulder_pitch_joint" joint="right_shoulder_pitch_joint"/>
  <motor name="right_shoulder_roll_joint" joint="right_shoulder_roll_joint"/>
  <motor name="right_shoulder_yaw_joint" joint="right_shoulder_yaw_joint"/>
  <motor name="right_elbow_joint" joint="right_elbow_joint"/>
  <motor name="right_wrist_yaw_joint" joint="right_wrist_yaw_joint"/>
  <motor name="right_wrist_pitch_joint" joint="right_wrist_pitch_joint"/>
  <motor name="right_wrist_roll_joint" joint="right_wrist_roll_joint"/>
</actuator>
```

**Motor 顺序 = 关节在 dof_names 中的顺序**：
```
腿 L(6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
腿 R(6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
腰(3): waist_roll, waist_pitch, waist_yaw        ← Adam SP body tree 顺序
臂 L(7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_pitch, wrist_roll
臂 R(7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_pitch, wrist_roll
```

**Mesh 文件**：从 `InstinctMJ/.../meshes/` 复制所有 STL 到 `humanoidverse/data/robots/adam_sp_mjlab/meshes/`

---

### 文件 2（新增）：机器人配置 YAML

**目标路径**：`configs/robots/adam_sp_29dof.yaml`

**代码依据**：`training_spec.py:120-202` `load_robot_training_spec()` 逐字段格式检查。

```yaml
name: adam_sp_29dof
xml_path: humanoidverse/data/robots/adam_sp_mjlab/adam_sp.xml
base_body: pelvis
root_quat_order: xyzw
coordinate_system: z_up
dof_unit: rad

control_joints:
  mode: all_actuated

feet:
  - left_ankle_roll_link
  - right_ankle_roll_link

hands:
  - left_wrist_roll_link
  - right_wrist_roll_link

key_bodies:
  - pelvis
  - torso_link
  - left_ankle_roll_link
  - right_ankle_roll_link
  - left_wrist_roll_link
  - right_wrist_roll_link

default_dof_pos: {}

training:
  hydra_robot: adam_sp/adam_sp_29dof_hard_waist
  hydra_overrides:
    - env.config.lie_down_init=True
    - env.config.lie_down_init_prob=0.3

  semantics:
    contact_bodies:
      - left_ankle_roll_link
      - right_ankle_roll_link
    undesired_contact_bodies:
      - pelvis
      - shoulder
      - hip
    torso_name: torso_link
    left_ankle_dof_names:
      - left_ankle_pitch_joint
      - left_ankle_roll_joint
    right_ankle_dof_names:
      - right_ankle_pitch_joint
      - right_ankle_roll_joint

  init_state:
    pos: [0.0, 0.0, 0.89]
    rot: [0.0, 0.0, 0.0, 1.0]
    lin_vel: [0.0, 0.0, 0.0]
    ang_vel: [0.0, 0.0, 0.0]
    default_joint_angles:
      left_hip_pitch_joint: -0.32
      left_hip_roll_joint: 0.0
      left_hip_yaw_joint: -0.18
      left_knee_joint: 0.66
      left_ankle_pitch_joint: -0.39
      left_ankle_roll_joint: 0.0
      right_hip_pitch_joint: -0.32
      right_hip_roll_joint: 0.0
      right_hip_yaw_joint: 0.18
      right_knee_joint: 0.66
      right_ankle_pitch_joint: -0.39
      right_ankle_roll_joint: 0.0
      waist_roll_joint: 0.0
      waist_pitch_joint: 0.0
      waist_yaw_joint: 0.0
      left_shoulder_pitch_joint: 0.0
      left_shoulder_roll_joint: 0.1
      left_shoulder_yaw_joint: 0.0
      left_elbow_joint: -0.3
      left_wrist_roll_joint: 0.0
      left_wrist_pitch_joint: 0.0
      left_wrist_yaw_joint: 0.0
      right_shoulder_pitch_joint: 0.0
      right_shoulder_roll_joint: -0.1
      right_shoulder_yaw_joint: 0.0
      right_elbow_joint: -0.3
      right_wrist_roll_joint: 0.0
      right_wrist_pitch_joint: 0.0
      right_wrist_yaw_joint: 0.0

  control:
    action_scale: 0.25
    action_clip_value: 5.0
    normalize_action_to: 5.0
    effort_limit_scale: 0.8
    effort_limit: [230.0, 180.0, 105.0, 230.0, 80.0, 40.0, 230.0, 180.0, 105.0, 230.0, 80.0, 40.0, 150.0, 150.0, 150.0, 65.0, 65.0, 65.0, 30.0, 6.4, 6.4, 6.4, 65.0, 65.0, 65.0, 30.0, 6.4, 6.4, 6.4]
    velocity_limit: [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
    stiffness:
      hip_pitch: 300.0
      hip_roll: 600.0
      hip_yaw: 300.0
      knee: 300.0
      ankle_pitch: 130.0
      ankle_roll: 70.0
      waist_roll: 400.0
      waist_pitch: 400.0
      waist_yaw: 400.0
      shoulder_pitch: 60.0
      shoulder_roll: 60.0
      shoulder_yaw: 60.0
      elbow: 60.0
      wrist_roll: 20.0
      wrist_pitch: 20.0
      wrist_yaw: 20.0
    damping:
      hip_pitch: 7.0
      hip_roll: 10.0
      hip_yaw: 2.0
      knee: 7.0
      ankle_pitch: 3.5
      ankle_roll: 2.0
      waist_roll: 11.0
      waist_pitch: 11.0
      waist_yaw: 11.0
      shoulder_pitch: 3.0
      shoulder_roll: 3.0
      shoulder_yaw: 3.0
      elbow: 3.0
      wrist_roll: 1.0
      wrist_pitch: 1.0
      wrist_yaw: 1.0

  actuator:
    source: yaml
    joints:
      left_hip_pitch_joint:    {effort_limit: 230.0, velocity_limit: 60.0, armature: 0.13426, friction: 1.589}
      left_hip_roll_joint:     {effort_limit: 180.0, velocity_limit: 60.0, armature: 0.281573, friction: 4.216}
      left_hip_yaw_joint:      {effort_limit: 105.0, velocity_limit: 60.0, armature: 0.23409, friction: 3.774}
      left_knee_joint:         {effort_limit: 230.0, velocity_limit: 60.0, armature: 0.13426, friction: 1.589}
      left_ankle_pitch_joint:  {effort_limit: 80.0, velocity_limit: 60.0, armature: 0.1098, friction: 0.3552}
      left_ankle_roll_joint:   {effort_limit: 40.0, velocity_limit: 60.0, armature: 0.1098, friction: 0.3552}
      right_hip_pitch_joint:   {effort_limit: 230.0, velocity_limit: 60.0, armature: 0.13426, friction: 1.589}
      right_hip_roll_joint:    {effort_limit: 180.0, velocity_limit: 60.0, armature: 0.281573, friction: 4.216}
      right_hip_yaw_joint:     {effort_limit: 105.0, velocity_limit: 60.0, armature: 0.23409, friction: 3.774}
      right_knee_joint:        {effort_limit: 230.0, velocity_limit: 60.0, armature: 0.13426, friction: 1.589}
      right_ankle_pitch_joint: {effort_limit: 80.0, velocity_limit: 60.0, armature: 0.1098, friction: 0.3552}
      right_ankle_roll_joint:  {effort_limit: 40.0, velocity_limit: 60.0, armature: 0.1098, friction: 0.3552}
      waist_roll_joint:        {effort_limit: 150.0, velocity_limit: 60.0, armature: 0.23409, friction: 1.887}
      waist_pitch_joint:       {effort_limit: 150.0, velocity_limit: 60.0, armature: 0.23409, friction: 1.887}
      waist_yaw_joint:         {effort_limit: 150.0, velocity_limit: 60.0, armature: 0.23409, friction: 1.887}
      left_shoulder_pitch_joint:  {effort_limit: 65.0, velocity_limit: 60.0, armature: 0.01, friction: 0.7548}
      left_shoulder_roll_joint:   {effort_limit: 65.0, velocity_limit: 60.0, armature: 0.01, friction: 0.7548}
      left_shoulder_yaw_joint:    {effort_limit: 65.0, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
      left_elbow_joint:           {effort_limit: 30.0, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
      left_wrist_yaw_joint:       {effort_limit: 6.4, velocity_limit: 60.0, armature: 0.01, friction: 3.213}
      left_wrist_pitch_joint:     {effort_limit: 6.4, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
      left_wrist_roll_joint:      {effort_limit: 6.4, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
      right_shoulder_pitch_joint: {effort_limit: 65.0, velocity_limit: 60.0, armature: 0.01, friction: 0.7548}
      right_shoulder_roll_joint:  {effort_limit: 65.0, velocity_limit: 60.0, armature: 0.01, friction: 0.7548}
      right_shoulder_yaw_joint:   {effort_limit: 65.0, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
      right_elbow_joint:          {effort_limit: 30.0, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
      right_wrist_yaw_joint:      {effort_limit: 6.4, velocity_limit: 60.0, armature: 0.01, friction: 3.213}
      right_wrist_pitch_joint:    {effort_limit: 6.4, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
      right_wrist_roll_joint:     {effort_limit: 6.4, velocity_limit: 60.0, armature: 0.01, friction: 0.80325}
```

**格式要点**：
- `joints` 是普通 YAML mapping（不是 `!!omap`）。`training_spec.py:158` 检查 `isinstance(joints, dict)`，`!!omap` 解析为 list[tuple] 会失败
- joints 中关节的顺序无所谓——`_actuator_params_from_training()` 按 `dof_names` 顺序遍历，用 `joints.get(joint_name)` 取值
- 每个 joint 必须有 `effort_limit, velocity_limit, armature, friction` 四个 float
- 注意 `left_wrist_yaw` 和 `right_wrist_yaw` 的 friction=3.213（不同于其他腕关节的 0.80325，来自 Adam SP XML）

---

### 文件 3（新增）：Hydra 机器人配置

**目标路径**：`humanoidverse/config/robot/adam_sp/adam_sp_29dof_hard_waist.yaml`

**代码依据**：`_patch_humanoidverse_robot_config()` (line 168-233) 会**覆盖**大部分字段。因此 Hydra 配置中的值只是"初始模板"，不对的值也会被覆盖。

**被 `_patch` 覆盖的字段**（可以不精确）：
- dof_names, dof_obs_size, actions_dim, body_names, num_bodies
- key_bodies, contact_bodies, num_feet, torso_name
- penalize_contacts_on, terminate_after_contacts_on
- dof_pos_lower/upper_limit_list, dof_vel_limit_list, dof_effort_limit_list
- init_state (全部), control.stiffness, control.damping, control.action_scale/clip_value/normalize_action_to
- asset.asset_root, asset.assetFileName, motion.asset.*

**不被覆盖的字段**（需要为 Adam SP 正确设置）：
- `robot.control.control_type`, `robot.control.integral`, `robot.control.action_rescale`, `robot.control.clip_torques`
- `robot.motion.extend_config`（被 `_patch` 读取并过滤，需要 parent_name 在 body_names 中存在）
- `robot.motion.motion_file`, `robot.motion.big_ankle`, 等 motion.* 字段
- `robot.foot_name`, `robot.knee_name` 等

**方案**：从 `g1_29dof_hard_waist.yaml` 复制，然后：
1. `dof_names` 列表按 Adam SP XML body tree 顺序写（虽然会被覆盖，但保持一致性）
2. `body_names` 列表更新为 Adam SP 的实际值（`robot_inspect` 生成）
3. motion 段 `extend_config` 的 parent_name 确认在 Adam SP body_names 中存在
4. `motion.motion_file` 指向用户数据
5. `control.integral` 中 arm 关节值根据用户决策设置（先填 0.0）
6. `motion.smpl_pose_modifier` — Adam SP 无此需求，可移除或清空

**关键确认项**：
- `foot_name: ankle_roll_link`（从 Adam SP body 名称 `left_ankle_roll_link` 推断共同后缀）
- `knee_name: knee_link`（确认在 Adam SP body_names 中存在）

---

### 文件 4（新增）：实验配置

**目标路径**：`humanoidverse/config/exp/bfm_zero/bfm_zero_adam_sp.yaml`

**方案**：从 `bfm_zero.yaml` 复制，仅改一行：

```yaml
defaults:
  - /robot: adam_sp/adam_sp_29dof_hard_waist    # ← 唯一必须修改的行
```

其余 defaults (/env, /simulator, /domain_rand, /rewards, /obs, /terrain, /callbacks) 完全相同。

`robot.motion.motion_file` 和 `robot.motion.max_len` 保持原样（训练时由 `--data-manifest` 覆盖）。

---

## 三、不需要修改的文件

| 文件 | 原因 |
|------|------|
| `humanoidverse/agents/envs/humanoidverse_mjlab.py` | `source: yaml` 使 actuator 参数完全来自 YAML，不需要硬编码函数 |
| `humanoidverse/utils/robot_spec/training_spec.py` | `source: yaml` 在白名单中 |
| `humanoidverse/agents/fb/*`, `fb_cpr/*`, `fb_cpr_aux/*` | 算法层与机器人无关 |
| `humanoidverse/agents/nn_models.py` | 维度自适应 |
| `humanoidverse/training/workspace.py` | 训练循环通用 |
| `humanoidverse/agents/presets/fb.py` | 超参数不变 |
| `humanoidverse/config/obs/bfm_zero_obs.yaml` | `${eval:'...'}` 自适应 num_bodies |
| `humanoidverse/config/rewards/reward_bfm_zero.yaml` | 奖励不变 |
| `humanoidverse/config/domain_rand/domain_rand.yaml` | 域随机化不变 |
| `humanoidverse/agents/evaluations/humanoidverse_mjlab.py` | `QVEL_IDX=23` 硬编码影响评估指标但不阻塞训练 |

---

## 四、实施步骤

### Step 1：复制并修改 XML

```bash
mkdir -p humanoidverse/data/robots/adam_sp_mjlab/meshes
cp /home/yangrr/codes/InstinctMJ/src/instinct_mj/assets/resources/adam_sp/xml/adam_sp.xml \
   humanoidverse/data/robots/adam_sp_mjlab/
cp /home/yangrr/codes/InstinctMJ/src/instinct_mj/assets/resources/adam_sp/meshes/*.STL \
   humanoidverse/data/robots/adam_sp_mjlab/meshes/
```

然后编辑 XML：删 actuatorfrcrange + armature + frictionloss + damping、改 meshdir、加 actuator 段。

### Step 2：运行 robot_inspect

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml humanoidverse/data/robots/adam_sp_mjlab/adam_sp.xml \
  --name adam_sp_29dof \
  --out configs/robots/adam_sp_29dof.yaml \
  --hydra-out humanoidverse/config/robot/adam_sp/adam_sp_29dof_auto.yaml
```

robot_inspect 自动将 robot YAML 的 `training.hydra_robot` 字段写为与 `--hydra-out` 文件名一致的路径。按 G1 惯例，Hydra 配置使用 `_hard_waist` 后缀。因此需要手动统一：

```bash
# 重命名 Hydra 配置
mv humanoidverse/config/robot/adam_sp/adam_sp_29dof_auto.yaml \
   humanoidverse/config/robot/adam_sp/adam_sp_29dof_hard_waist.yaml
```

然后将 robot YAML 第 151 行从 `hydra_robot: adam_sp/adam_sp_29dof_auto` 改为 `hydra_robot: adam_sp/adam_sp_29dof_hard_waist`。

### Step 3：手动完善 YAML + Hydra + exp 配置

在 robot_inspect 生成的草稿基础上：
1. 填写 robot YAML 的 training 段参数（见 §一 和 §二.文件2）
2. 完善 Hydra 机器人配置（从 auto 重命名，补充 motion 段字段，见 §二.文件3）
3. 创建 exp 配置（复制自 bfm_zero.yaml，改 robot 引用，见 §二.文件4）

### Step 4：准备运动数据并验证

用户数据为 UFO pkl 格式，结构与 `lafan_29dof_10s-clipped.pkl` 一致，直接通过 `--data-path` 加载。

**pkl 格式要求**：每条运动为 dict，含 `root_trans_offset [T,3]`, `pose_aa [T,J,3]`, `dof [T,29]`, `root_rot [T,4]`, `fps`。

**dof 列顺序校验**（最关键）：pkl 的 `dof` 列顺序必须与 Adam SP XML 的 actuator 顺序一致。用脚本交叉验证：

```bash
python3 -c "
import joblib, yaml
from pathlib import Path

# 加载机器人配置
with open('configs/robots/adam_sp_29dof.yaml') as f:
    robot_cfg = yaml.safe_load(f)

# 加载运动数据
data = joblib.load('/path/to/adam_sp_motion.pkl')
sample_key = list(data.keys())[0]
dof_shape = data[sample_key]['dof'].shape
assert dof_shape[1] == 29, f'dof width {dof_shape[1]} != 29'

# 加载 XML 获取 control_joint_names
import mujoco
xml_path = robot_cfg['xml_path']
model = mujoco.MjModel.from_xml_path(xml_path)
ctrl_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
assert len(ctrl_names) == 29, f'Expected 29 actuators, got {len(ctrl_names)}'

print(f'XML actuator order ({len(ctrl_names)} joints):')
for i, name in enumerate(ctrl_names):
    print(f'  [{i:2d}] {name}')
print(f'dof shape: {dof_shape}')
print('Cross-check: dof width matches actuator count ✓')
print()
print('WARNING: Verify that dof column order matches the above actuator order.')
print('If data was generated for G1, columns at indices 12-14 (waist) and 19-21/26-28 (wrist) may be swapped.')
"
```

### Step 5：训练前强制校验

**effort_limit 交叉校验**：`control.effort_limit[i]` 按下标用于 action_rescale，必须与同序关节的 `actuator.joints[name].effort_limit` 一致：

```bash
python3 -c "
import yaml
with open('configs/robots/adam_sp_29dof.yaml') as f:
    cfg = yaml.safe_load(f)
dof = cfg['training']['control']['effort_limit']
act = cfg['training']['actuator']['joints']
# dof_names 从 XML motor 顺序决定，robot_inspect 生成
# 此处做一致性校验：effort_limit 列表长度必须 = 29
assert len(dof) == 29, f'effort_limit length {len(dof)} != 29'
print('effort_limit cross-check: OK (29 values)')
"
```

### Step 6：Smoke test

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/adam_sp_29dof.yaml \
  --data-path /path/to/adam_sp_motion.pkl \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_adam_sp
```

### Step 7：长训前短门禁（推荐）

`--smoke` 只验证环境能初始化，挡不住数据错序/错 action_rescale。长训前建议额外跑一个短门禁：

```bash
# 短训 50K 步，生成一个 checkpoint
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/adam_sp_29dof.yaml \
  --data-path /path/to/adam_sp_motion.pkl \
  --gpu-ids single \
  --num-env-steps 50000 \
  --work-dir /tmp/ufo_gate_adam_sp

# 用这个 checkpoint 跑 tracking_inference，目视检查是否正常跟踪
uv run python -m humanoidverse.tracking_inference \
  --model-folder /tmp/ufo_gate_adam_sp \
  --robot-config configs/robots/adam_sp_29dof.yaml \
  --data-path /path/to/adam_sp_motion.pkl \
  --headless --save-mp4 --motion-list 0
```

**Goal 推理注意**：Adam SP 是 29-DOF，`goal_inference.py` 不会触发非 G1 检查（`num_dof==29` 通过），会静默加载 `humanoidverse/data/robots/g1/goal_frames_lafan29dof.json`。必须传 `--goal-json` 指向 Adam SP 专用的目标帧文件。

---

## 五、已知风险

| # | 风险 | 严重性 | 说明 | 缓解 |
|---|------|--------|------|------|
| 1 | NPZ 缺 `joint_names` → 静默错序 | 🔴 致命 | 只发 warning，不报错。腰/腕顺序与 G1 不同时，dof_pos 列映射错位，训练正常收敛但策略学歪 | Step 4 强制检查 NPZ 含此字段 |
| 2 | `control.effort_limit[i]` 顺序错 → action_rescale 错增益 | 🔴 致命 | 按下标用于 action_rescale，不验证与 actuator.joints 一致性。腰/腕下标与 G1 混用时 PD 目标尺度错 | Step 5 effort_limit 交叉校验 |
| 3 | Goal 推理静默加载 G1 goal JSON | 🔴 致命 | `num_dof==29` 通过非 G1 检查，默认加载 `goal_frames_lafan29dof.json`。只传 `--robot-config` 不够 | 推理时必须传 `--goal-json` |
| 4 | Smoke test 挡不住上述问题 | 🟠 严重 | `--smoke` 步数极少且关 eval，只能验证环境初始化 | Step 7 短门禁：50K 步短训 + tracking_inference 目视验证 |
| 5 | `num_bodies ≈ 54` | 🟡 中等 | privileged_obs 从 462 增至 ~807 维 | 网络自动适应 |
| 6 | 推理默认指向 G1 | 🟡 中等 | 推理入口默认加载 G1 配置 | 推理时传 `--robot-config` |

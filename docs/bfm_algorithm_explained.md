# BFM-Zero / FB-CPR 算法学习笔记

> 基于 UFO 项目 (github.com/Roboparty/UFO) 的代码分析  
> 核心论文：[Meta Motivo (arXiv:2504.11054)](https://arxiv.org/abs/2504.11054) + [BFM-Zero (arXiv:2511.04131)](https://arxiv.org/abs/2511.04131)

---

## 总体架构

### 类继承链

```
Model 层次:
  FBModel (fb/model.py)                 ← B(s), F(s,z,a), π(s,z)
    └── FBcprModel (fb_cpr/model.py)    ← + Critic, Discriminator
         └── FBcprAuxModel (fb_cpr_aux/model.py)  ← + AuxCritic

Agent 层次:
  FBAgent (fb/agent.py)                 ← update_fb(), update_td3_actor()
    └── FBcprAgent (fb_cpr/agent.py)    ← + Discriminator/Critic/z混合
         └── FBcprAuxAgent (fb_cpr_aux/agent.py)  ← + AuxCritic
```

### 预设配置 (presets/fb.py)

```yaml
z_dim: 256
norm_z: true                   # 映射到单位超球面
seq_length: 8                  # 专家轨迹编码时域窗口
actor_std: 0.05                # 策略标准差
num_parallel: 2                # Ensemble 网络数

# 网络架构
Forward/Critic:  ResidualForward, hidden_dim=2048, layers=6, num_parallel=2
Backward:        MLP, hidden_dim=256, layers=1, norm=true
Actor:           ResidualActor, hidden_dim=2048, layers=6
Discriminator:   MLP, hidden_dim=1024, layers=3

# 训练超参
learning_rates:  lr_f=3e-4, lr_b=1e-5, lr_actor=3e-4, lr_disc=1e-5, lr_critic=3e-4
ortho_coef:      100.0         # 正交正则强度
discount:        0.98
batch_size:      1024
update_z_every_step: 100
z_buffer_size:   8192
```

---

## 第一阶段：核心概念 — FBModel 的三个网络

### 文件：`agents/fb/model.py`

### B(obs)：Backward 编码器

```
obs [526] → Linear → LayerNorm → Tanh → [ReLU×1] → Linear → Norm() → z [256]
```

- **输入**：`state(64)` + `privileged_state(462)` = 526 维
- **输出**：z ∈ ℝ²⁵⁶，归一化到单位超球面
- `Norm()`: `z = sqrt(256) × normalize(z)` → z 在球面上
- **功能**：把观测编码到 latent 空间，z 代表"当前状态属于什么运动模式"

```python
# 核心代码
self._backward_map = arch.b.build(obs_space, arch.z_dim)

def backward_map(self, obs):
    return self._backward_map(self._normalize(obs))
```

### F(s,z,a)：Forward 预测器

```
obs + action → embed_sa → h_dim//2 ─┐
                                      → concat → [ResidualBlock×6] → z'
obs + z     → embed_z  → h_dim//2 ──┘
```

- **输入**：`state + privileged_state + last_action + history_actor` + z + action
- **输出**：预测的未来 latent z' ∈ ℝ²⁵⁶
- **Ensemble**：`num_parallel=2`，两个并行网络，用差异度估计不确定性
- **功能**：从 (当前状态, 技能 z, 动作 a) 预测未来的 latent

```python
self._forward_map = arch.f.build(obs_space, arch.z_dim, action_dim)
```

### π(s,z)：Actor 策略

```
obs → embed_s → h_dim//2 ──┐
                              → concat → [ResidualBlock×6] → tanh → μ
z   → embed_z → h_dim//2 ──┘
                                          + σ=0.05 → TruncatedNormal(μ, σ)
```

- **输入**：`state(64)` + `last_action(29)` + `history_actor(372)` + z(256) = **721 维**
- **输出**：TruncatedNormal(μ, σ=0.05)，截断到 [-1, 1]
- **功能**：给定状态和技能 z，输出动作

```python
self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)

def act(self, obs, z, mean=True):
    dist = self.actor(obs, z, self.cfg.actor_std)
    return dist.mean if mean else dist.sample()
```

### 观测组装

```python
# humanoidverse_mjlab.py:955-968
obs = {
    "state":             [dof_pos(29) + dof_vel(29) + proj_gravity(3) + base_ang_vel(3)]  = 64
    "privileged_state":  [全身运动学: 位置+旋转+线速+角速+高度]                            = 462
    "last_action":       [上一时刻的 29 维动作]                                            = 29
    "history_actor":     [最近 4 帧的 dof_pos/dof_vel/proj_gravity/base_ang_vel/actions]  = 372
    "time":              [当前步数]                                                        = 1
}
```

各网络的输入选择（通过 input_filter）：

| 网络 | 使用的 keys | 输入维度 |
|------|------------|---------|
| Actor | state + last_action + history_actor + z | 64+29+372+256 = 721 |
| Backward | state + privileged_state | 64+462 = 526 |
| Forward | state + privileged_state + last_action + history_actor + z + action | 927+256+29 = 1212 |
| Critic | 同 Forward | 1212 |
| Discriminator | state + privileged_state + z | 526+256 = 782 |

---

## 第二阶段：核心算法 — FB 的对比学习

### 文件：`agents/fb/agent.py` — `update_fb()` (line 217-298)

### FB Loss 的核心公式

没有 reward 函数。FB 的"学习信号"来自对比一致性：

```python
Fs = F(obs, z, action)                    # [2, B, 256] F预测的未来z'
B = B(goal)                                # [B, 256] B编码的目标帧z
Ms = Fs @ B.T                              # [2, B, B] 相似度矩阵

# 目标侧（target网络，无梯度）
target_Fs = target_F(next_obs, z, next_action)
target_B = target_B(goal)
target_Ms = target_Fs @ target_B.T

# Bellman残差
diff = Ms - discount * target_Ms           # [2, B, B]

# 对比损失
fb_diag    = -mean(diag(diff)) × B         # 对角元推大（正样本匹配）
fb_offdiag = 0.5 × sum(diff_offdiag²)      # 非对角元推小（负样本不匹配）
fb_loss    = fb_offdiag + fb_diag
```

**直观理解**：F 要预测"执行技能 z 后未来的 latent z'"。B 要把"真实的下一帧"编码成 z。两者应该一致。

### 正交正则

```python
Cov = B @ B.T                               # Gram矩阵
orth_loss = -mean(diag(Cov))                # 每个z和自己的内积要大（区分度）
          + 0.5 × sum(offdiag²)             # 不同z之间要正交
fb_loss += 100.0 × orth_loss
```

**防止坍塌**：没有这个，B 会把所有帧都编成同一个 z，Loss 直接归零。

### Actor 更新

```python
action = π(obs, z).sample()
Fs = F(obs, z, action)
Q_fb = (Fs × z).sum(-1)                    # F预测的z'和z的内积
actor_loss = -Q_fb.mean()                   # 最大化这个内积
```

Actor 的目标：选动作 a 使 F 相信"我正在执行技能 z"。

---

## 第三阶段：从对比学习到有行为 — FB-CPR

### 文件：`agents/fb_cpr/agent.py`

### 核心问题

第一阶段只让 F/B 学会了 latent 空间的一致性，但策略并不知道什么样的行为是"好"的。FB-CPR 引入：

1. **运动数据 → 专家编码**     : 告诉策略"世界上存在这些行为模式"
2. **鉴别器 D(s,z)**            : 告诉策略"什么样的 (s,z) 像自然的运动"
3. **z 混合采样**               : 控制策略探索什么样的技能

### 专家编码 encode_expert()

```python
# 运动轨迹 8 帧 → B 逐帧编码 → 平均 → 1 个 z
B_expert = B(next_obs)                      # [B, 256]
B_expert = B_expert.view(N, 8, 256)         # 每条轨迹 8 帧
z_expert = B_expert.mean(dim=1)             # 帧平均
z_expert = project_z(z_expert)              # 投影到超球面
return repeat_interleave(z_expert, 8)       # 展开回 batch
```

### z 混合采样 sample_mixed_z()

三种来源，每个样本按概率选一个：

| 来源 | 比例 | 意义 |
|------|------|------|
| 目标编码 | 20% | z = B(随机目标帧) → "去到这个状态" |
| 专家编码 | **60%** | z = 随机专家轨迹平均 → "模仿这个运动" |
| 先验采样 | 20% | z = N(0,I) 投影 → "探索新技能" |

### 鉴别器 D(s,z)

```python
# 网络
[obs(526) + z(256)] → Linear → LayerNorm → Tanh → [ReLU×3] → Linear → 1 → sigmoid

# 训练：WGAN-GP
expert_logits = D(expert_obs, expert_z)     # 推大
train_logits = D(train_obs, train_z)        # 推小
loss = -log(sigmoid(expert)) + softplus(train) + 10.0 × gradient_penalty

# 奖励
reward = log(D) - log(1-D)                  # log-odds 形式
```

### Critic 更新

```python
reward = D.compute_reward(obs, z)            # 鉴别器奖励
next_action = π(next_obs, z).sample()
next_Q = target_critic(next_obs, z, next_action)
next_V = mean(next_Q) - pessimism × uncertainty(next_Q)  # 悲观估计
target_Q = reward + discount × next_V

Q = critic(obs, z, action)
critic_loss = MSE(Q, target_Q)
```

### 完整 update() 流程

```python
def update(self, replay_buffer, step):
    # 1. 采样
    expert_batch = buffer["expert"].sample(B)
    train_batch  = buffer["train"].sample(B)

    # 2. 编码专家轨迹
    expert_z = encode_expert(expert_next_obs)

    # 3. 训练鉴别器
    update_discriminator(expert_obs, expert_z, train_obs, train_z)

    # 4. 混合采样 z + 重标记
    z = sample_mixed_z(train_next_obs, expert_z)
    train_z = relabel(train_z, z, ratio=0.8)   # 80% 替换

    # 5. FB 更新（对比学习）
    update_fb(train_obs, train_action, discount, train_next_obs, z)

    # 6. Critic 更新（鉴别器奖励）
    update_critic(train_obs, train_action, discount, train_next_obs, z)

    # 7. Actor 更新
    update_actor(train_obs, train_action, z)

    # 8. 软更新目标网络
    soft_target_update()
```

---

## 第四阶段：物理惩罚辅助奖励 — FB-CPR-Aux

### 文件：`agents/fb_cpr_aux/agent.py`

### 为什么要加 AuxCritic

运动数据只告诉"做什么"，没告诉"别做什么"。物理惩罚是硬编码的"安全底线"：

```yaml
# presets/fb.py
aux_rewards:
  - penalty_torques            # 力矩过大
  - penalty_action_rate        # 动作抖动
  - limits_dof_pos             # 关节超限
  - limits_torque              # 力矩超限
  - penalty_undesired_contact  # 不该接触的碰到了
  - penalty_feet_ori           # 脚掌姿势不对
  - penalty_ankle_roll         # 踝关节异常
  - penalty_slippage           # 打滑

aux_rewards_scaling:
  penalty_action_rate: -0.1
  penalty_feet_ori: -0.4
  penalty_ankle_roll: -4.0
  limits_dof_pos: -10.0
  penalty_slippage: -2.0
  penalty_undesired_contact: -1.0
```

### 三个 Q 的组合

```python
# Actor 最终目标
actor_loss = -Q_fb.mean()                          # FB 一致性（主力）
           - 0.05 × weight × Q_disc.mean()          # 像专家
           - 0.02 × weight × Q_aux.mean()           # 不摔不滑
```

| Q 来源 | 网络 | 信号来自 | 权重 |
|--------|------|---------|------|
| Q_fb | Forward | F(s,z,a) · z 内积 | 1.0 |
| Q_disc | Critic | D(s,z) log-odds 奖励 | 0.05 × 自适应 |
| Q_aux | AuxCritic | 物理惩罚加权和 | 0.02 × 自适应 |

自适应权重：`weight = abs(Q_fb).mean().detach()`，用 FB Q 的值级自动缩放鉴别器和辅助 Q。

---

## 训练循环 — workspace.py

### 文件：`training/workspace.py` — `train_online()` (line 556)

### 主循环

```python
for local_time in range(0, max_time, 1024):
    # ── 环境交互 ──
    z = maybe_update_rollout_context(z)
    action = π(obs, z) if step > num_seed_steps else random_action()
    new_td, reward, done, info = env.step(action)
    buffer.extend({obs, z, action, reward, next_obs, aux_rewards...})

    # ── 训练 ──
    if step % 1024 == 0 and buffer非空:
        for _ in range(16):                     # 16 次更新
            metrics = agent.update(buffer)
            sync_gradients()                    # 多卡
            sync_buffers()                      # BN 同步
            total_metrics += metrics

    # ── 日志/评估/checkpoint ──
    if step % 384000 == 0:   log_metrics()
    if step % 3200000 == 0:  eval() + checkpoint()
```

### 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| num_envs | 1024/GPU | 每卡并行环境数 |
| num_seed_steps | 10240 | 初始随机填充步数 |
| update_agent_every | 1024 | 训练间隔 |
| num_agent_updates | 16 | 每次训练更新轮数 |
| update_z_every_step | 100 | rollout 中 z 的切换频率 |
| buffer_size | 5,120,000/GPU | Replay buffer 容量 |
| num_env_steps | 192,000,000 | 全局总步数（8GPU） |

---

## 推理模式

### 文件：`tracking_inference.py` / `goal_inference.py` / `reward_inference.py`

三种推理共享同一个 API：编个 z，喂给 π。

### Tracking（运动跟踪）

```python
# z = B(运动序列的每一帧) → 滑动平均
z_t = B(motion_frame_t)
actions = π(current_state, z_t)   # 跟随运动
```

### Goal（目标到达）

```python
# z = B(目标姿态)
z = B(goal_pose)
actions = π(state, z)              # 向目标收敛，z 不变
```

### Reward（奖励优化）

```python
# z = Σ B(s_i) × softmax(10 × reward_i)  ← 用 replay buffer 算
z = weighted_avg_B(replay_buffer, reward_fn)
actions = π(state, z)              # 执行优化后的行为
```

### 部署时的实际方案

```python
# 所有 z 在训练阶段离线算好，部署时查表
z = load("z_walk.pt")
while running:
    obs = robot.get_observation()
    action = π(obs, z)
    robot.step(action)
```

---

## 为什么会这样设计

### 对比学习代替 reward

传统 RL 的 Bellman 方程：
```
Q(s,a) = r(s,a) + γ · Q(s',a')       ← 需要 reward 函数 r
```

FB 的对比一致性：
```
F(s,z,a) · B(s')  vs  γ · F(s',z,a') · B(s'')
          ↑ 不需要 reward
```

### z 的三重角色

1. `z = B(s)` → **技能编码**
2. `Q_fb = F(s,z,a) · z` → **隐式 reward 定义**
3. `z = B(s_goal)` → **目标编码**

### 为什么推理能零样本

训练时 Actor 被训练成"最大化 F(s,z,a) · z"。这个操作等价于"向 latent 空间中 z 指向的区域运动"。推理时你只需要构造出那个 z（从运动数据、目标姿态或 reward 加权），π 自然就跟着走了。

### 关键限制

- 需要大量运动数据（60% z 来自专家编码）
- 8 GPU × 6 小时训练
- z 维度 (256) 靠经验定
- sim-to-real 需要域随机化
- 不能自动重定向到不同形态的机器人

---

## 完整的数据流图

```text
replay buffer ──→ expert_z ──→ z混合 ──→ π(s,z) ──→ action ──→ env.step
                  (60%)        (20%目标           │
                               20%先验)            │
                                                   │
                                                   ▼
                                           buffer.extend()
                                                   │
                                          ┌────────┴────────┐
                                          ▼                  ▼
                                   update_fb()        update_discriminator()
                                   F·B 对比学习       D(s,z): 专家 vs 策略
                                          │                  │
                                          ▼                  ▼
                                   update_critic()     update_aux_critic()
                                   D 奖励的 Q          物理惩罚的 Q
                                          │                  │
                                          └────────┬─────────┘
                                                   ▼
                                            update_actor()
                                            最大化 Q_fb + Q_disc + Q_aux
```

---

## 对应论文

| 内容 | 论文 |
|------|------|
| FB-CPR 算法（Meta Motivo） | [arXiv:2504.11054](https://arxiv.org/abs/2504.11054) |
| BFM-Zero 工程实现 | [arXiv:2511.04131](https://arxiv.org/abs/2511.04131) |
| UFO 框架 | [Project Page](https://roboparty.github.io/UFO/) |

---

*笔记生成日期: 2026-07-17*  
*基于 UFO Commit: 6e19dc4*

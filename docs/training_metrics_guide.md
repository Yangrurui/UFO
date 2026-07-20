# BFM-Zero / FB-CPR 训练指标指南

> 所有指标均从 `agent.update()` 各步骤中收集，经过多卡 average（如果分布式），每 `log_every_updates` (384000) 步取平均后写入 W&B / CSV。

---

## 一、FB（Forward-Backward）对比学习指标

这些是 FB 算法**最核心**的指标，反映 latent 空间的学习质量。

| 指标 | 公式/含义 | 好不好 | 走向 |
|------|----------|--------|------|
| **`fb_loss`** | `fb_diag + fb_offdiag + ortho_coef × orth_loss` | FB 对比学习的总损失 | ↓ 下降 |
| **`fb_diag`** | `-mean(diag(M - γ·target_M)) × B` | **对角元匹配度**：F(s,z,a) 预测的 z' 与 B(next_obs) 编码的 z' 是否一致。负值推大 → 越小（越负）越好 | ↓ 越负越好 |
| **`fb_offdiag`** | `0.5 × sum(offdiag_diff²)` | **非对角元区分度**：不同样本的预测和编码是否不匹配。越接近 0 越好 | ↓ 趋近 0 |
| **`orth_loss`** | `-mean(diag(Cov)) + 0.5 × sum(offdiag(Cov)²)` | **正交正则化**：不同帧的 z 编码是否彼此正交（保证多样性）。分两项： | ↓ 趋近 0 |
| **`orth_loss_diag`** | `-mean(diag(B@B.T))` | 每个 z 和自己的内积是否够大（区分度）。越负 → z 越有区分度 | ↓ 越负越好 |
| **`orth_loss_offdiag`** | `0.5 × sum(offdiag(Cov)²)` | 不同 z 之间的内积是否趋近 0（正交性） | ↓ 趋近 0 |
| **`target_M`** | 目标矩阵均值 | target network 的相似度矩阵值。正常训练时应稳定 | → 稳定 |
| **`M1`** | Ms[0] 的均值 | 当前 F 预测的相似度矩阵值。应随训练逐渐逼近 target_M | → 追踪 target_M |
| **`F1`** | Forward 嵌入 Fs[0] 的均值 | F 网络的嵌入值。应在合理范围内（-1~1） | → 稳定 |
| **`B`** | Backward 嵌入 B 的均值 | B 网络的嵌入值。理想情况下接近 0（球面均匀分布） | → 稳定 |
| **`B_norm`** | B 的 L2 范数均值 | z 在超球面上的归一化程度。应接近 sqrt(z_dim)=16（若 norm_z=True） | ≈ 16 稳定 |
| **`z_norm`** | 采样 z 的 L2 范数均值 | z 先验分布的范数，应与 B_norm 匹配 | ≈ B_norm |
| **`q_loss`** | TD 学习损失（默认 q_loss_coef=0，通常为 0） | 仅在启用 Q-loss 时有效 | = 0 |

### FB 指标诊断

- **fb_loss 不降或震荡** → 学习率太高或 buffer 样本多样性不足
- **fb_diag 不降（接近 0 或正）** → F 和 B 没有学到有效的对比关系，可能是 z 更新太频繁或 batch size 太小
- **fb_offdiag 很大** → 不同样本的编码没有被充分区分，正交正则可能不够
- **orth_loss 很高** → 正交正则权重 `ortho_coef=100` 可能需要微调
- **B_norm 偏离 sqrt(z_dim)** → z 的归一化有问题（极少发生，Norm() 层保证了归一化）
- **三个 loss 分量中 fb_diag 主导** → 正常现象，对角项始终是最大项

---

## 二、鉴别器（Discriminator）指标

反映策略生成的数据与专家运动数据的**分布对齐程度**。

| 指标 | 含义 | 走向 |
|------|------|------|
| **`disc_loss`** | 鉴别器总损失 = expert_loss + train_loss + wgan_gp | ↓ 趋近稳定 |
| **`disc_expert_loss`** | `-log(sigmoid(D(expert_obs, expert_z)))`：D 对专家数据的置信度损失 | ↓ 下降 |
| **`disc_train_loss`** | `softplus(D(train_obs, train_z))`：D 对策略数据的抑制损失 | ↓ 下降 |
| **`disc_wgan_gp_loss`** | WGAN 梯度惩罚项 | ↓ 趋近 0 |
| **`mean_disc_reward`** | `log(D) - log(1-D)`，即 D 给出的 log-odds 奖励 | → 逐渐上升（策略越像专家，奖励越高） |

### 鉴别器指标诊断

- **disc_expert_loss 很低但 disc_train_loss 很高** → 鉴别器太强，轻松区分专家和策略。正常训练初期
- **disc_expert_loss 和 disc_train_loss 都低** → 鉴别器被策略"骗过"了（策略产生的数据很像专家），**理想状态**
- **disc_loss 快速降到零** → 鉴别器坍缩，检查梯度惩罚
- **mean_disc_reward 持续上升** → 策略越来越像运动数据，**训练健康**
- **mean_disc_reward 持续下降** → 策略退化了

---

## 三、Critic / Q 值指标

| 指标 | 含义 | 走向 |
|------|------|------|
| **`target_Q`** | Bellman target = reward + γ × next_V | → 稳定 |
| **`Q1`** | Critic 当前估计的 Q 值 | → 上升（策略在变好） |
| **`mean_next_Q`** | 下一状态的 Q 均值 | → 与 Q1 同步上升 |
| **`unc_Q`** | Ensemble 不确定性 = 两个 Critic 的差异度 | ↓ 下降（不确定性降低） |
| **`critic_loss`** | Critic 的 MSE 损失 | ↓ 下降 |

### Q 值指标诊断

- **Q1 持续上升且 critic_loss 下降** → 策略在持续改善，**理想状态**
- **Q1 上升但 critic_loss 也上升** → Critic 学习不稳定或奖励噪声太大
- **unc_Q 很大** → 两个 Critic 分歧大，可能数据覆盖不足
- **target_Q 偏离 Q1** → Critic 更新滞后或 discount 设置有误

---

## 四、Actor（策略）指标

| 指标 | 含义 | 走向 |
|------|------|------|
| **`actor_loss`** | `-(Q_fb + reg_coeff×weight×Q_disc + reg_coeff_aux×weight×Q_aux)` | ↓ 越负越好 |
| **`Q_fb`** | `(F(s,z,a) × z).sum(-1)` — FB 内积，表示"动作 a 是否在执行技能 z" | ↑ 上升 |
| **`Q_discriminator`** | 鉴别器 Critic 的 Q 值 — "动作 a 是否让 (s,z) 像专家" | ↑ 上升 |
| **`Q_aux`** | 物理惩罚 Critic 的 Q 值 — "动作 a 是否避免了物理惩罚" | ↑ 上升（不摔不滑） |

### Actor 指标诊断

- **actor_loss 持续下降（变负）** → 策略在学习，**核心健康信号**
- **Q_fb 上升** → FB 空间中的行为质量在改善
- **Q_discriminator 上升** → 策略产生的数据越来越像专家
- **Q_aux 持续为负或不上升** → 物理惩罚严重，机器人可能在摔、滑、关节超限
- **actor_loss 震荡但 Q_fb/Q_disc 持续上升** → 可能是 aux Q 在波动（正常），只看 Q_fb 趋势即可

---

## 五、辅助奖励（Aux Reward）分解指标

每个 aux reward 项都有独立的均值日志（带 `aux_rew/` 前缀）：

| 指标 | 含义 | 好不好 |
|------|------|--------|
| **`aux_rew/penalty_torques`** | 力矩平方和均值 | ↓ 越小越好 |
| **`aux_rew/penalty_action_rate`** | 相邻帧动作变化平方和 | ↓ 越小越好（动作平滑） |
| **`aux_rew/limits_dof_pos`** | 关节超出软限位的累计量 | → 趋近 0 |
| **`aux_rew/limits_dof_vel`** | 关节速度超出软限位的累计量 | → 趋近 0 |
| **`aux_rew/limits_torque`** | 力矩超出软限位的累计量 | → 趋近 0 |
| **`aux_rew/penalty_undesired_contact`** | 非期望接触的二元指示均值 | → 趋近 0 |
| **`aux_rew/penalty_ankle_roll`** | 踝关节内/外翻角度平方和 | ↓ 越小越好 |
| **`aux_rew/penalty_feet_ori`** | 脚掌触地时偏离水平的角度 | ↓ 越小越好 |
| **`aux_rew/penalty_slippage`** | 脚部打滑速度 × 接触力指示 | ↓ 越小越好 |
| **`aux_rew/feet_heading_alignment`** | 脚掌前进方向与躯干的夹角 | ↓ 越小越好 |
| **`mean_aux_reward`** | 上述所有惩罚的加权和（经 EMA 归一化） | → 逐渐上升（惩罚减少） |
| **`aux_critic_loss`** | AuxCritic 的 TD 损失 | ↓ 下降 |

### Aux 指标诊断

- **penalty_torques 不降** → PD 跟踪误差大，可能需要检查 kp/kd 是否过小
- **penalty_undesired_contact > 0 持续** → 机器人在摔/蹭地，策略质量差
- **penalty_slippage 很高** → 脚底接触力检测可能有问题，或者 body 名称映射有误
- **limits_dof_pos > 0 持续** → 运动数据中某些帧超出 Adam SP 关节限位，或者策略在驱使关节超限

---

## 六、训练系统指标

| 指标 | 含义 |
|------|------|
| **`distributed/global_env_steps`** | 全局环境步数 |
| **`distributed/optimizer_steps`** | 优化器更新次数 |
| **`FPS`** | 每秒环境步数（含评估开销）。G1 参考值 ~500-1000 FPS/GPU |
| **`duration [minutes]`** | 累计训练时间 |
| **`torso_contact_force_*`** | 躯干接触力统计（均值、P95、最大值），用于监控是否摔倒 |
| **`torso_z_*`** | 躯干（骨盆）高度统计。P05 过低意味着机器人长期处于低位/摔倒 |

### 系统指标诊断

- **FPS 太低** → GPU 利用不足或 body 数过多导致 FK 开销大
- **torso_z_p05 < 0.5** → 机器人频繁摔倒，骨盆离地太近
- **torso_contact_force_contact_frac > 0** → 机器人摔倒/蹭地比例

---

## 七、训练阶段的典型指标变化

### 初期（0 - 10M steps）

```
fb_loss:  从高值震荡 → 缓慢下降
fb_diag:  从接近 0 → 越来越负
orth_loss: 从高值 → 缓慢下降
disc_loss: 较高（鉴别器轻松区分专家和策略）
Q_fb:     从接近 0 → 缓慢上升
Q_aux:    通常为负（策略在摔）
actor_loss: 从接近 0 → 变负
aux_rew/penalty_undesired_contact: 可能 > 0.1（策略在摔倒）
torso_z_p05: 可能 < 0.5（频繁摔倒）
```

### 中期（10M - 50M steps）

```
fb_loss:  继续下降，趋于稳定
fb_diag:  稳定在较负的值
orth_loss: 趋于稳定（接近 0）
disc_loss: 缓慢下降（鉴别器开始被策略蒙蔽）
Q_fb:     持续上升
Q_disc:   开始上升
Q_aux:    从负值趋近 0（物理惩罚减少）
actor_loss: 持续变负
aux_rew/p_undesired: 趋近 0
torso_z_p05 趋近 0.7-0.8
```

### 后期（50M - 192M steps）

```
fb_loss:  基本稳定，微调
disc_loss: 很低（策略与专家难以区分）
Q_fb/Q_disc/Q_aux: 全部趋于平台
actor_loss: 稳定在一个负值
各 aux 惩罚: 稳定在很低的水平
torso_z_p05 ≥ 0.75
```

### 异常信号

| 信号 | 可能原因 |
|------|---------|
| fb_loss 不降 | lr 太高、batch 太小、z 更新频率不当 |
| fb_diag 始终接近 0 | ortho_coef 太低导致 z 坍塌，所有 z 相同 |
| orth_loss 持续高位 | B 编码多样性不足，运动数据太少或 batch 太小 |
| disc_loss 快速降到 0 | 判别器过强，梯度惩罚不足 |
| Q_aux 始终为负不改善 | 物理参数（PD、effort_limit）配置有误 |
| penalty_undesired_contact 不降 | torso_name 或罚则 body 名称不对 |
| 所有指标都正常但跟踪效果差 | 运动数据 dof 列顺序与 actuator 顺序不匹配（**静默错误**） |

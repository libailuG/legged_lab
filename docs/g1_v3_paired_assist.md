# G1 v3 成对助力改造

任务：`LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v3`。

## 动作与部署输入

- Actor 输入保持 150 维：25 帧左右外骨骼角度、角速度、已处理指令力矩。
- Actor 输出由 2 维改为 1 维有符号动作 `u`。负值抬左腿、压右腿；正值压左腿、抬右腿。
- 目标标量 `T_target = 10 * clip(u, -1, 1) * g`，最终两个电机指令为 `[T, -T]`。
- `g` 仅由两侧外骨骼角速度计算：先各自做时间常数 0.05 s 的低通，再对绝对速度取最大值，使用 0.15–0.8 rad/s 的 smoothstep 门控。
- 两侧共用门控，支撑腿静止不会单独关闭该侧反向力矩。
- 对标量 T 统一限速 80 N·m/s；100 Hz 下每步最多变化 0.8 N·m。符号反转时先到零，下一步才进入另一方向。
- 释放、限速和最终写入执行器时始终保持左右等大反向，限幅为 ±10 N·m。
- 两腿停止后，目标为零，但因低通和限速存在短暂退出过程。
- 足端信息、本体髋关节状态、机身状态均不进入助力动作处理。仿真中的冻结 G1 行走策略仍单独使用本体状态。

## 仅用于训练的足端评价

Critic 输入保持本次前置修改后的 825 维，其中足端信息为左右 Fz 与触地标志的历史。Actor 不含这些信息。

奖励使用足端 link 的世界 Z 方向净法向接触力构造承重系数 `sL, sR`：

- 进入接触：Fz ≥ 20 N；保持接触：Fz > 10 N。
- 承重幅值在 10–200 N 之间平滑增加，达到 200 N 时幅值因子为 1。
- 接触持续时间在 0–0.05 s 之间平滑增加，避免刚触地就建立满参考力矩。
- 两因子相乘；失去接触时立即清空接触计时和承重系数。

以上门限是可调的仿真参考参数，不代表已经验证的人体舒适度或承载阈值。接触评价只影响 Critic/奖励，不能在部署时保证“未触地绝不下压”。

## 新参考力矩

髋关节负角速度表示屈髋。每侧抬升需求为：

`d_i = tanh(max(-(2*v_i + 0.04*a_filtered_i), 0)/1.5) * smoothstep(-v_i, 0.15, 0.8)`。

其中髋加速度低通时间常数为 0.08 s。正角速度的下落/伸髋不产生抬升需求。

`eL = dL * (1-sL) * sR`，`eR = dR * (1-sR) * sL`。

`T_ref = 10 * (eR-eL)`，参考力矩为 `[T_ref, -T_ref]`。

因此左腿抬升、左侧卸载且右脚承重时，参考为左负右正；反之镜像。双脚无承重、双脚完全承重或没有屈髋需求时，参考为零。承重过渡期间连续变化。

## 当前启用的奖励

| 项目 | 权重 | 说明 |
|---|---:|---|
| assist_supported_lift_tracking | -8 | 跟踪上述成对参考，按 10 N·m 归一化 |
| assist_unsupported_press | -8 | `mean((max(tau,0)/10)^2 * (1-support))`，惩罚未承重腿下压 |
| assist_torque_zero_at_bilateral_rest | -2 | 用两侧髋运动量的最大值计算共同静止权重，避免单独惩罚静止支撑腿 |
| assist_hip_angle_error | -5 | 保留外骨骼与本体髋角度误差 |
| assist_hip_velocity_error | -0.05 | 保留角速度误差 |
| rear_mechanism_zero_position_error | -1 | 保留滑块/圆柱零位置误差 |
| rear_mechanism_velocity | -0.5 | 新增机构速度惩罚，归一化尺度为 0.2 m/s、1 rad/s |
| assist_output_smoothness | -0.2 | 统一按 80 N·m/s 归一化 |
| assist_requested_rate_excess | -0.005 | 惩罚目标力矩请求超出每步 0.8 N·m 的部分 |
| termination_penalty | -200 | 保留异常终止惩罚 |

旧的逐腿动态跟踪和逐腿静止奖励已替换。左右力矩和由动作结构保证为零，不另加重复的力矩和奖励。

## 兼容性与验证

- Actor 网络现在为 `150 -> 256 -> 64 -> 16 -> 1`。旧的 2 输出助力检查点不能直接续训或部署，需重新训练此版本。
- Critic 为 `825 -> 256 -> 64 -> 16 -> 1`。
- 冻结行走策略仍为 `model_51800.pt` 对应的 TorchScript 导出；G1 控制频率和机器人本体控制器未在本次改造中调整。
- 单机器人无助力绘图程序已使用环境动作维数创建零动作，可继续使用。
- 后部机构原有 PD 保留。等大反向电机力矩并不保证实际背板反力完全抵消，需要训练后的背板位移/速度对照评价。
- 新参考和权重是第一版设计，需要用相同速度指令比较训练后的背板运动、悬空下压、抬升助力及实际步速，不能由代码验证推断已改善人体体验。

CPU 数值回归测试：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python source/legged_lab/test/test_g1_v3_paired_assist.py
```

无助力数据记录：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/play_g1_v3_foot_contact.py --vx 0.7 --duration 15 --real-time
```

注意：零动作播放仅验证无助力基线，不能展示尚未训练的新助力策略效果。

本次验证结果：8 项 CPU 数值测试通过；4 个仿真环境完成 160 步脚本动作检查，峰值指令 7.5 N·m，左右指令力矩和、实际电机力矩和的最大绝对值均为 0；变化率最大约 80 N·m/s。奖励滤波/接触计时的重置检查和一次小规模 PPO 更新通过。这次 PPO 更新仅用于接口验证，不是可用的训练结果。报告位于 `logs/check_g1_v3_paired_assist/verification.json`。


## 2026-09-08 本体 PID 对齐与重新训练

v3 深拷贝 assist-v1 的五组本体 PID 执行器，保留当前外骨骼模型和初始姿态。Kp/Kd/Ki、积分力矩限幅及抗饱和算法复用 v1；积分每个物理步更新，重置清零。后部机构保持 PD，外骨骼保持 ±10 N·m 直接力矩控制，观测与奖励保持原设置。本体策略仍冻结使用 policy_51800.pt，本体策略更新频率已对齐 v1 Play 的 50 Hz，外骨骼策略保持 100 Hz。

从头训练外骨骼，不加载旧 PD 环境下的 PPO 检查点：

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v3 \
  --num_envs 4096 --max_iterations 10000 --headless \
  --run_name paired_assist_10nm_body_pid_50hz
```

日志保存到 `logs/rsl_rl/g1_assist_exoskeleton_v2_v3_ppo/日期_时间_paired_assist_10nm_body_pid_50hz/`。训练后导出新策略，Sim2Sim 必须指定此目录的 `params/env.yaml`。

本体观测噪声在 v3 训练、Play 和 Sim2Sim 中全部关闭；外骨骼 Actor/Critic 的观测设置不受此调整影响。

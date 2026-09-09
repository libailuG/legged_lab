# v6 有效阶段增强助力

基于实机反馈“出力阶段正确但偏弱”，从v5独立建立v6训练/Play任务。v5和已有Sim2Real文件不变。v6仍需训练、导出后才能评价实际体感；修改奖励不会改变已有检查点。

## 训练变化

| 项目 | v5 | v6 |
|---|---|---|
| 完成脉冲目标峰值 | 3～6.5 N·m | 6～10 N·m |
| 完成脉冲峰值误差权重 | -4 | -6 |
| 末端姿态参考最大幅值 | 6.5 N·m | 10 N·m |
| 末端姿态支撑误差权重 | -4 | -6 |

峰值目标为 `6 + 4 × posture` N·m，在选中腿卸载、对侧承重、本侧抬升/保持且未明显下落时按有效机会加权累计；完成脉冲且累计有效机会至少0.05秒才评价。它是训练目标，不是每次输出的最低限制。

末端参考为 `10 × posture × eligibility × (1-lowering)`，左右组成等大反向参考；无需速度或加速度大于零即可对抬腿末端保持提供目标。姿态posture沿用v5：相对默认髋角屈曲0.1～0.5 rad的平滑映射。绝对默认角代理仍有局限，需结合实测评价。

保持±10 N·m、80 N·m/s限制，脉冲峰值/时长触发锁定，0.25秒上升/释放、0.6～1.4秒总时长。此次不延迟下落/反向释放，不降低触发阈值，不强制最低输出。保留无支撑下压、下落阻力、站立零力矩与背板稳定约束。

Actor仍450维、两个动作、100 Hz，只含外骨骼可获得的信号和内部历史。足端信号用于Critic与奖励。训练角度偏置每回合固定±10°；Play不添加偏置。本体仍model_51800、原PID、无本体观测噪声。

## 诊断日志

除已有 `Pulse/mean_onset_peak_nm` 和 `Pulse/zero_peak_fraction`，新增：

- `Pulse/mean_actual_peak_nm`：已完成脉冲的实际输出峰值均值。
- `Pulse/mean_duration_s`：完成脉冲实际持续时间，包含释放段。
- `Pulse/mean_eligible_support_s`：每个完成脉冲的加权有效支撑机会时长，不是足接触时长或超阈值力矩时长。
- `Pulse/release_lowering_fraction`、`release_opposite_lift_fraction`、`release_quiet_fraction`、`release_timeout_fraction`：完成脉冲释放原因占比；同时满足时按下落、对侧抬腿、静止、到时排序，归入唯一原因。

统计按各环境当前回合累计，reset时清零。开始峰值与实际峰值的分母分别为开始数、完成数，短窗口不宜直接相减。诊断状态不进入Actor，不影响脉冲曲线。

## 训练

建议从头训练，独立日志目录：

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v6 \
  --num_envs 1024 --max_iterations 10000 --headless \
  --run_name stronger_support
```

日志：`logs/rsl_rl/g1_assist_exoskeleton_v2_v6_ppo/<时间>_stronger_support/`。

优先观察 `Episode_Reward/assist_completed_pulse_peak`、`assist_terminal_support` 与上述Pulse日志，同时检查 `assist_unsupported_press`、`assist_lowering_resistance` 是否恶化。负误差奖励更接近零表示匹配更好，不能只追求力矩大。奖励权重已变化，v5/v6奖励值不能直接比较，需比较相同速度下实际峰值、持续时间、支撑条件和行走稳定性。

Play任务：`LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2-v6`。本次仅建立训练与Isaac Play任务，尚未生成v6 Sim2Sim/Sim2Real部署包。

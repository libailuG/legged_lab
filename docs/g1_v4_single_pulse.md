# v4 单峰成对助力任务

任务 `LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v4`；播放任务 `LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2-v4`。独立目录，保留 v3。新的任务尚未完成正式训练，当前验证不代表实机体感已解决。

## 为什么重新设计

v3 每 10 ms 更新助力目标；速度门控和含加速度的训练参考也会逐帧变化。80 N·m/s 限速只能限制相邻采样的变化，不能防止同一次助力反复升降。这是可解释的振荡来源，但尚不能据此排除传感器、传动或通信造成的机械振动。

v4 将策略权限改为“选择本次助力峰值”，由固定波形发生器输出完整曲线。目标是阻止逐帧策略变化在单个助力阶段造成多个脉冲。

## 动作与状态机

- 仍为一个动作 u∈[-1,1]，现在映射峰值 A=10×(u+1)/2×起始门控，范围 0～10 N·m。动作语义与 v3 不同。
- 仅用外骨骼左右速度识别开始方向。速度先做 0.05 s 低通，较快屈曲侧速度<-0.2 rad/s，左右速度差绝对值>0.3 rad/s，方向连续确认 0.03 s。
- 同一方向仅启动一次。双腿同时同速屈曲不会强制输出反向助力。安静后再次运动或确认相反方向可重新触发。
- 开始时锁定峰值及方向；之后策略动作和速度门控的波动都不再调整该次峰值。
- 常规曲线 T(t)=direction×A×sin²(pi×t/0.4)，从零平滑上升并回到零；中途没有第二个峰。左右始终 `[T,-T]`。
- 确认停止或相反运动时，从当前力矩用 0.2 s 的 cos² 单调下降曲线释放到零，完成前不启动另一方向。释放与原曲线在力矩值上连续；提前释放交接处不保证二阶导数连续。
- 最大值 ±10 N·m；0.4 s 正常曲线理论最大斜率约 78.54 N·m/s，0.2 s 释放曲线同样不超过 78.54 N·m/s。代码拒绝会突破 80 N·m/s 的持续时间设置。
- reset 清空脉冲状态。固定0.4秒是经用户确认的初始实验参数，后续需按不同速度的实测助力时机调整；不假定所有步态都适合相同宽度。

控制器自身状态（进度、方向、锁定峰值、释放进度、释放起点、已消耗方向、候选方向及确认时长、低通后的左右速度，共10维）加入每帧观测。Actor 共400维，Critic共1075维，均保留25帧历史。没有将足端、人体髋关节或人体姿态加入部署输入。

## 奖励调整

不再使用瞬时速度/加速度参考力矩的逐帧跟踪误差，也移除针对旧请求力矩的限速超额惩罚。

- `assist_supported_positive_work`，+4：有对侧承重时，在屈曲腿提供正向抬腿做功；速度按2 rad/s截断归一化。
- `assist_lowering_resistance`，-4：腿已经下落时，惩罚继续向上拉造成的阻力。
- `assist_unsupported_press`，-8：保留无承重腿上的下压力惩罚。
- `assist_torque_zero_at_bilateral_rest`，-2：保留双侧静止时出力惩罚。
- `hip_motor_torque_burden`，-0.2：增加本体髋电机负担惩罚，作为次要辅助指标，不能单独视为人体省力证明。
- 保留输出变化惩罚、背板机构稳定、髋/外骨骼同步、速度跟踪和跌倒惩罚。

足端信号仍只供 Critic/奖励评价。接触变化不会在部署后处理里立即切割助力曲线。代价是放弃部分瞬时跟随能力，策略需要学习在可能造成无支撑下压时选择较低峰值。

## 本体与训练

本体仍冻结 policy_51800.pt，50 Hz 推理、1000 Hz PID、无本体观测噪声。外骨骼 100 Hz。外骨骼模型、后部PD机构及接触评价约束保留。

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v4 \
  --num_envs 4096 --max_iterations 10000 --headless \
  --run_name single_pulse_04s
```

从头训练，不传 --resume。日志目录 `logs/rsl_rl/g1_assist_exoskeleton_v2_v4_ppo/日期_时间_single_pulse_04s/`。v3 的网络与部署程序不适用 v4 的400维输入及新动作语义。当前已提供 v4 Isaac 训练/Play 和 TorchScript MuJoCo Sim2Sim；NumPy/C 部署已提供于 sim2real/g1_assist_exoskeleton_v4，使用对应脉冲状态机，不能使用 v3 exporter。

## 验证与观察

4项单元测试覆盖：策略高频跳变仍只有一个峰；停止单调释放和下一方向重启；起始门控锁定与独立环境复位；超出斜率限制的参数被拒绝。2个环境完成一次PPO训练迭代，实际Actor/Critic维度确认400/1075。

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python source/legged_lab/test/test_g1_v4_pulse.py
```

正式训练后重点看：单侧连续助力区间是否单峰、助力峰值是否落在屈曲阶段、释放是否拖入下落阶段、无支撑下压是否减少、背板是否稳定、步速是否退化。不能只看正向做功奖励增大：峰值偏大或时机错误也可能使其他指标恶化。


## v4 MuJoCo Sim2Sim

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v4.py --vx 0.7 --plot-window 5
```

默认使用 `2026-09-08_15-42-59_single_pulse_04s` 目录及其 `exported/policy.pt`（当前为 model_499.pt 导出）。400维历史包含脉冲内部状态；直接复用训练 SinglePulse 实现。保留实时曲线、力矩对比及峰值、速度滑条；不保存数据。关闭窗口或 Ctrl+C 退出。需要 NumPy、PyTorch、MuJoCo、Matplotlib、PyYAML、tkinter。

更换训练目录传 `--env-config /新目录/params/env.yaml`，默认读取对应 exported/policy.pt；也可显式指定 `--assist-policy`。`--disable-assist` 输出零助力。该入口是 TorchScript 推理版本，不是此前 v3 的全 NumPy 入口。已通过短时 GUI 和400维历史排列检查。


已生成 model_499 的 Sim2Real NumPy/C99 部署包，接口与验证见 `sim2real/g1_assist_exoskeleton_v4/README.md`。

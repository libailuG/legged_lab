# v5 低速与末端支撑优化

训练任务：`LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v5`。
播放任务：`LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2-v5`。
独立于v4，本次未修改旧任务、旧权重和旧部署程序。

## 动作优化

v4 的起始速度门控会在低速触发时把峰值锁定得很小。v5完全取消这个幅值乘数；滤波速度仅用于阶段触发/释放，峰值由策略决定。

策略输出两个动作，均在[-1,1]：

- 峰值：A=10×(u0+1)/2 N·m，低速也可以选择完整0～10 N·m范围。
- 总持续时间：D=0.6+0.8×(u1+1)/2秒，即0.6～1.4秒。

触发时锁定两者，中途动作跳变不改这次脉冲。方向仍由本地外骨骼编码器判断，不接收足端/人体状态。输入速度先做0.05秒低通；较快屈曲侧速度<-0.08 rad/s、左右速度差绝对值>0.12 rad/s，连续确认0.06秒后触发。

脉冲为0.25秒sin²平滑上升、保持峰值、0.25秒cos²平滑释放。正常总时长由D决定，因此保持约0.1～0.9秒。释放会在以下情况提前开始：

- 当前被助力的腿出现>0.12 rad/s的滤波伸展速度；
- 确认对侧屈曲方向；
- 双腿滤波速度均<0.04 rad/s连续0.3秒，且脉冲已运行至少0.5秒；
- 达到已锁定的总时长减去释放时间。

“变慢或短暂停留”不再直接触发释放，但不是无限保持；有静止超时和最长持续时间。发生释放后只单调减小，回到零前不触发新方向；同方向不能在一次持续运动中反复触发。

左右始终[T,-T]，±10 N·m。0.25秒上升/释放的理论最大斜率约62.83 N·m/s，低于80 N·m/s。提前释放力矩连续，但交接点不保证二阶导数连续。低触发阈值是否会受到实机微小运动干扰，需要后续实测。

## 奖励优化

保留有支撑抬腿做功、无支撑下压、下落阻力、输出平滑、背板稳定与步速评价。

新增 `assist_terminal_support`（权重-4）：用真实髋相对默认姿态的屈曲角度生成低速支撑参考；角度超过0.10 rad开始建立，0.50 rad达到满姿态系数。乘上对侧承重/本侧卸载系数，参考最大幅值6.5 N·m；伸展速度从0到0.25 rad/s时逐渐撤销参考。参考由左右需求差形成等大反向力矩，用跟踪误差评价。

它是训练用的姿态支撑启发式，不是人体逆动力学模型，也不能声称6.5 N·m适合所有穿戴者。重要变化是：即使关节速度和加速度接近零，抬腿姿态仍可以产生支撑参考。

站立出力惩罚改为 `standing_rest` 模式，按双腿安静程度和直立姿态权重计算；抬高一条腿时显著降低“站立不该出力”的惩罚，避免与末端支撑直接冲突。对应日志名仍为 `assist_torque_zero_at_bilateral_rest`。

## 零位偏置训练

每个环境reset时，左右分别独立采样[-10°,10°]的偏置，内部使用弧度。整个回合固定，加到Actor外骨骼角度观测及其历史。原±0.5°逐帧噪声保留，模拟零偏之上的测量噪声。

偏置不改变物理关节状态、不改变速度、不进入真实角度奖励。Critic使用真实外骨骼角度，不直接把随机偏置值传给Actor。只对外骨骼角度随机化，本体行走策略观测仍无噪声。

v5 Play默认偏置范围为0，以便重复对比；如需检验零偏鲁棒性，可在配置中恢复angle_bias_range。部署不应随机增加偏置，而应使用实际传感器标定；训练随机化增强对残余偏差的容忍，不能代替机械错位/传动误差建模。

## 维度、频率与命令分布

- Actor450维：25帧的左右q、dq、指令力矩，以及12维控制器状态（10维脉冲状态+两侧滤波速度）。
- Critic1125维，额外足端/人体信息仅用于训练。
- 策略输出2维：峰值、时长。与v4的400维/1动作不兼容。
- 本体仍冻结policy_51800.pt，策略50 Hz、PID1000 Hz、无本体观测噪声；助力100 Hz。
- 训练前进速度范围调整为0.1～1.2 m/s，站立环境比例25%；当前训练更聚焦低速，因此不能据此保证高速或倒退表现。Play仍可使用原速度范围进行评估。

## 训练与验证

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v5 \
  --num_envs 1024 --max_iterations 10000 --headless \
  --run_name low_speed_terminal_support_bias10
```

从头训练，不传--resume。日志位于 `logs/rsl_rl/g1_assist_exoskeleton_v2_v5_ppo/日期_时间_low_speed_terminal_support_bias10/`。

本次5项单元测试通过，覆盖低速完整幅值、周期内幅值锁定、末端短暂停留、反向释放、限速、固定零偏及独立环境重置。2个环境完成1次PPO迭代，实际输入输出维度确认450/1125/2。尚未完成正式训练或验证真实穿戴体感。

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python source/legged_lab/test/test_g1_v5_pulse.py
```

训练后重点观察低速峰值、末端支撑误差、下落阻力、无支撑下压、站立残余力矩、背板稳定及步速。若末端支撑误差下降但下落阻力升高，应减少保持时长或调整释放，而不是单纯加大奖励。v5后续Sim2Sim/Sim2Real必须使用450维历史与新的两动作状态机，旧v4程序不能直接加载。

## v5 Sim2Sim 播放

已适配450维观测、两动作峰值/时长和v5状态机。默认读取 `2026-09-08_17-12-18_low_speed_terminal_support_bias10` 目录，目前导出来自 `model_499.pt`，并保留检查点专用 `exported/policy_499.pt`。

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v5.py --vx 0.7 --plot-window 5
```

这是TorchScript推理版本，不是v3/v4部署入口。无本体观测噪声，默认不添加随机角度零偏；保留实时曲线、力矩峰值、速度滑条，不保存数据。`--disable-assist` 为零助力对照。关闭任一窗口或Ctrl+C退出。

换模型可指定 `--env-config /新目录/params/env.yaml --assist-policy /新目录/exported/policy.pt`。已通过导出与训练Actor数值对比，以及短时MuJoCo图形播放和450维历史检查。

## 2026-09-09 零峰值退化修正

诊断旧model_499：vx=0.7、30秒MuJoCo中49次触发有48次选择零峰值，实际峰值0.88 N·m、RMS0.071 N·m。固定5 N·m离线对照可正确施加力矩。导出与检查点一致，问题主要是触发时的策略选择，不是执行器没有收到输出。

本次保持450维输入、2动作及脉冲物理规则，调整训练信号：

- 新增 `assist_completed_pulse_peak`，权重-4。在一次脉冲结束时，用该阶段积累的对侧承重、本侧卸载、屈曲/抬腿姿态，评价锁定峰值。至少积累0.05秒有效加权支撑机会才生效。目标为姿态相关3～6.5 N·m的加权平均，属于训练启发式，不是实机最低出力要求。
- 即使本次脉冲峰值为零，也累计机会并评价；无有效支撑机会则这项误差为零。已有下落阻力和无支撑下压惩罚继续生效。按step_dt归一化，使一次性事件评价不会被10 ms时间缩放稀释。
- 新增 `assist_action_bounds`，权重-0.2。训练包装器不再预先裁剪动作，奖励能看到超出[-1,1]的原始值；脉冲发生器仍裁剪物理峰值和持续时间，因此不改变±10 N·m上限。
- rollout由48增加为192步（1.92秒）；gamma由0.995改为0.999，lambda由0.975改为0.995，加强跨脉冲奖励回传。这仍是逐控制步PPO，不是完整事件级PPO。
- 初始探索标准差0.3，最小值0.15，降低过早收敛到零幅值区域的风险。

新增日志（当前各环境回合内累计统计）：

- `Pulse/starts_per_env`：平均触发次数。
- `Pulse/zero_peak_fraction`：触发峰值≤0.1 N·m的比例。
- `Pulse/mean_onset_peak_nm`：平均触发峰值。
- `Pulse/supported_completion_fraction`：完成脉冲中具备有效评价机会的比例。

如果触发次数少或有效机会比例接近零，应检查阶段识别和承重条件，而不是仅增加奖励权重。零峰值比例不应被要求恒为零：确实不适合助力时，选择零仍是正确行为。

建议从头训练，新日志名区分旧版本：

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v5 \
  --num_envs 1024 --max_iterations 10000 --headless \
  --run_name pulse_credit_fix
```

旧检查点不会因修改训练奖励而自动改善。新策略训练完成后需要重新导出。新增开始/结束标志只用于奖励和日志，不进入Actor观测，也不改变物理脉冲曲线。


## 16 GiB GPU 显存配置

192步rollout保持不变，默认并行环境降为1024，PPO mini-batch数量增为8。相较4096环境/4个mini-batch，rollout样本数为1/4，单次更新mini-batch为1/8（24576样本）。每轮总样本数也下降，不能把新旧相同迭代数视为相同训练量。

OOM发生在第一次PPO更新的TensorDict采样分配；报错时显存几乎耗尽且PyTorch未用缓存很少，主要是总占用不足。并行运行的旧v4 Isaac Play已结束。训练时避免同时打开Isaac Play。

重新运行上面的1024环境命令。该次失败在第一次更新，通常尚无可续训检查点；不要从旧零助力model_499继续。

## 最新500轮Sim2Real

已将 `2026-09-09_09-24-28_pulse_credit_fix_1024/model_499.pt` 导出到 `sim2real/g1_assist_exoskeleton_v5/weights/`，并生成等价NumPy/C99控制器。部署输入450维、双动作，保持100 Hz、±10 N·m成对输出；部署不添加训练用随机角度偏置。接入、离线演示及验证命令见 [v5 Sim2Real说明](../sim2real/g1_assist_exoskeleton_v5/README.md)。

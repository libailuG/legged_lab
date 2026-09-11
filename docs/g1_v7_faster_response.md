# v7 更快起始响应

基于v6独立新建任务，减少可观测抬腿后的出力滞后。本版没有新增预测网络或预助力，不声称能够在静止、无运动信号时识别人抬腿意图。v6及其部署包保持不变。

| 参数 | v6 | v7 |
|---|---|---|
| 速度滤波时间常数 | 50 ms | 30 ms |
| 开始助力的方向确认 | 60 ms | 30 ms |
| 平滑上升时间 | 250 ms | 200 ms |
| 平滑释放时间 | 250 ms | 250 ms |
| 对侧抬腿释放确认 | 60 ms | 60 ms |

对侧抬腿释放单独保留60 ms确认，候选方向时长仍最多积累60 ms，观测按60 ms归一化。启动判断使用30 ms阈值。下落速度阈值、静止退出、到时释放不变；滤波加快也会影响这些条件实际达到的时刻，所以不能认为释放时刻严格等同v6。

保持6～10 N·m有效支撑目标、末端10 N·m参考上限、两项权重-6；保持±10 N·m、80 N·m/s限幅、单峰/保持/释放、峰值和时长锁定。200 ms正弦平方上升到10 N·m的最大导数约78.54 N·m/s，满足80限制。持续时间仍0.6～1.4秒，较短上升在相同总时长下会增加保持段。

Actor450维、双动作、100 Hz；足端仍只供评价和Critic。左右各自每回合独立固定±10°偏置、本体model_51800及PID、本体无观测噪声不变。Play偏置为零。

## 延迟日志

新增训练诊断，完全不进入Actor或控制决策。抬腿起点定义：未滤波真实髋角速度连续满足与触发相同的方向阈值（两腿速度差超过0.12 rad/s、抬升侧速度小于-0.08 rad/s）的第一帧。它是运动阈值代理，不是神经意图或从零速度开始的生理起点。奖励读取仿真步后的髋速度，存在控制帧采样误差，时间分辨率10 ms。

- `Response/mean_trigger_delay_s`：匹配同侧运动起点到脉冲开始的均值。
- `Response/mean_half_peak_delay_s`：上述运动起点到实际输出首次达到锁定峰值50%的均值。
- `Response/matched_start_fraction`、`matched_starts_per_env`：用于判断有效匹配数量；触发时原始速度方向不匹配则不纳入延迟均值。
- `Response/half_peak_events_per_env`：达到50%峰值的事件数量；峰值≤0.1 N·m、提前结束未达50%的脉冲不计入半峰延迟。必须结合计数查看，避免把零样本均值0误认为零延迟。

统计按环境回合累计，reset清零。与v6一样保留实际峰值、实际时长、各释放原因占比等Pulse日志。

## 验证

9项单元测试通过，包含零峰值不产生半峰事件、局部reset、反向仍需60 ms、支持条件奖励与平滑/限幅。相同合成阶跃速度输入下，v6/v7触发延迟为70/30 ms，半峰延迟为190/120 ms；这是合成信号结果，不是实机改善保证。32环境1轮Isaac训练及PPO更新完成，新日志成功输出。实际训练后要重点观察提前出力是否增加无支撑下压或下落阻力。

## 训练

建议从头训练，不沿用v6旧部署包：

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v7 \
  --num_envs 1024 --max_iterations 10000 --headless \
  --run_name faster_response
```

日志：`logs/rsl_rl/g1_assist_exoskeleton_v2_v7_ppo/<时间>_faster_response/`。

Play任务：`LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2-v7`。训练完成后需生成使用30/30/200 ms参数的v7 Sim2Sim/Sim2Real；不能仅将新权重放进v6控制器。

## 已训练2000轮的Sim2Sim

当前运行 `2026-09-09_18-02-01_faster_response/model_1999.pt`，已核对导出推理一致。注意：检查该次保存的env.yaml发现环境配置覆盖了滤波默认值，这次模型实际训练参数是50 ms滤波、30 ms启动确认、200 ms上升。Sim2Sim忠实读取这份保存配置。已修正源环境配置为30 ms，影响后续新训练，不修改旧训练文件或旧权重。

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v7.py --vx 0.7 --plot-window 5
```

默认使用该次训练的env.yaml与exported/policy.pt；单机器人、实时滚动曲线、独立速度滑条，不保存CSV。关闭任一窗口或Ctrl+C退出。

## 当前2000轮Sim2Real

已生成 `sim2real/g1_assist_exoskeleton_v7/`，与上述Sim2Sim同一检查点、同一保存参数。含NumPy/C99和权重，5项测试通过，含1500步控制及观测历史对齐。实机通信由调用方接入，不添加随机偏置或额外力矩倍率。详见 [部署说明](../sim2real/g1_assist_exoskeleton_v7/README.md)。

## 前进速度范围更新

v7新训练与Isaac Play的vx范围统一为0.4～3.0 m/s。站立采样概率为10%，站立时指令仍为零；vy、转向与朝向配置不变。已保存的训练配置、检查点和现有Sim2Sim/Sim2Real包不修改，新范围在重新启动训练/Play后生效。

## 9月10日新训练Sim2Sim

当前脚本默认运行已验证导出的 `2026-09-10_09-25-22_faster_response/model_500.pt`，读取同一运行保存的env.yaml。确认实际训练滤波30 ms、启动确认30 ms、上升200 ms，vx范围0.4～3.0 m/s、站立10%。上述不带路径的Sim2Sim命令现在指向本次模型。此前Sim2Real包仍是旧2000轮模型，未同步替换。

## 9月10日500轮Sim2Real

当前 `sim2real/g1_assist_exoskeleton_v7/` 已更新为9月10日model_500.pt及30 ms滤波，启动30 ms、上升200 ms，来源参数和权重经过验证。旧2000轮部署完整备份到 `sim2real/g1_assist_exoskeleton_v7_2026-09-09_model1999/`。5项测试通过，1500步C/NumPy最大力矩误差约9.54e-7 N·m。接入接口不变，必须成套使用当前权重与C源码。

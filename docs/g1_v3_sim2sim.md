# G1 v3 MuJoCo Sim2Sim

程序：`scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v3.py`。

本体已切换为 assist-v1 的 PID，冻结行走策略仍为 `model_51800.pt` 对应导出。必须通过 `--env-config` 指定新 PID 训练目录的 `params/env.yaml`。默认加载该目录的 `exported/policy.pt`，也可通过 `--assist-policy` 指定。旧 PD 训练配置会被拒绝。

## 实时播放

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v3.py \
  --env-config /绝对路径/新PID训练目录/params/env.yaml --vx 0.7 --plot-window 5
```

只运行一个机器人，默认开启助力，持续运行直到关闭任一窗口或按 Ctrl+C。跌倒、姿态超限或髋角失配会自动重置继续运行，跨回合曲线断开。W/S 修改 vx，A/D 修改 vy，Q/E 修改转向，空格将速度命令清零，R 重置。

独立的 `G1 v3 - speed control` 窗口提供 **vx 滑条**，范围 -0.5～3.0 m/s，步进 0.05 m/s；拖动即修改前进速度指令。W/S 和空格的速度调整也会同步到滑条。实际速度请看 Measured 曲线。

实时图默认滚动显示最近 5 秒仿真时间，`--plot-window` 调整窗口长度，`--plot-interval` 调整刷新间隔（默认 0.1 秒墙钟时间）。包含：

- 左右髋与外骨骼角度、角速度，以及足端接触力；绿色背景表示触地。
- 每侧髋电机力矩和实际助力力矩叠加比较：各自除以当前窗口内的最大绝对值，保留正负方向，纵轴固定为 -1.1～1.1。零力矩显示为零。图例实时标注各自的绝对峰值（N·m）；归一化曲线用于比较形状与相位，实际幅值应看峰值。
- 背部滑块位置、圆柱角度，以及实际/指令前进速度。

仅保留有限长度的内存绘图缓存，不创建数据目录，不保存 CSV、PNG 或 JSON。已移除 `--duration`、`--output-dir`、`--keep-plots`、`--headless`、`--no-live-plot` 和 `--auto-reset` 参数。Plot 使用 TkAgg，需要桌面显示和 tkinter。刷新可能降低播放速度，但不改变物理步长和控制周期。

零助力对照在命令后加 `--disable-assist`；后部机构 PD 继续保持。

## 对齐内容

- 从训练时保存的 `params/env.yaml` 读取关节顺序、默认角度、PID 增益及积分限幅、力矩上限、armature、仿真步长和控制周期。
- 本次配置是 **1 kHz 物理及本体 PID、50 Hz 本体行走策略、100 Hz 助力**。PID 积分每 0.001 s 更新，积分力矩限幅 ±10 N·m，饱和时拒绝继续同向积分，机器人重置时清零。本体目标每 20 ms 更新，中间保持；各环境重置后立即推理并重新计时。
- 助力网络 150 -> 1，归一化在 TorchScript 内；历史按位置、速度、指令力矩分组，各 25 帧，最旧到最新，初始历史用第一帧填充。
- 与训练一致的共同速度低通/门控、[T,-T] 输出、±10 N·m 限幅、80 N·m/s 共同限速、换向经过零。
- 同时修改 MuJoCo 电机控制、执行器力和关节执行器力限幅，避免 XML 中遗留的 ±8 N·m 截断。
- 后部机构限位从旧 XML 的滑块 ±0.01 m、圆柱 ±pi 修正为训练 USD 的 **±0.05 m、±0.2 rad**，仅修改本次运行内存中的模型。
- 足端接触力仅作实时绘图，不进入助力 Actor 或动作后处理。
- MuJoCo 无外部随机扰动。v3 训练、Play 和 Sim2Sim 均关闭本体观测噪声。Sim2Sim 即使读取训练 YAML，也使用无噪声本体观测。MuJoCo 和 PhysX 的接触和动力学求解并非完全相同，控制对齐不代表动力学完全一致。

## 更换策略

后续替换助力模型时，先通过 Isaac 的 `play.py --checkpoint ...` 导出，再同时指定对应的导出和训练配置：

```bash
python scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v3.py \
  --assist-policy /绝对路径/新训练目录/exported/policy.pt \
  --env-config /绝对路径/新训练目录/params/env.yaml \
  --vx 0.7
```

`--assist-policy` 接受包含归一化的 TorchScript，不能直接传 `model_N.pt`。若冻结行走模型移到了其他目录，使用 `--gait-policy` 指定其 TorchScript。

## 当前 PID 验证

6 项 CPU 测试通过（含跨环境重置的 50 Hz 保持测试），其中直接执行 v1 原始 PID compute/reset 方法，与 MuJoCo 控制器逐步对比 2000 步。2 个 Isaac 环境完成 1 次 PPO 迭代，保存配置确认 29 个本体关节使用 PID。MuJoCo 零助力窗口检查通过，包含滑条同步和关闭窗口退出；这不是长时间稳定性验证。

## 旧 PD 控制迁移的历史验证（不代表当前 PID 效果）

4 项 CPU 测试通过，包括 1000 步随机运动/换向/停止的 NumPy 与训练 PyTorch 后处理一致性，以及历史排列检查。

开启/关闭助力均完成 15 秒、1500 条记录，没有重置或终止。开启助力的实际峰值约 9.01 N·m，左右实际助力力矩和最大误差为 0；所有控制步满足 0.8 N·m 变化量限制。

相同 vx=0.7 指令下，2 秒后平均实际速度：助力开启约 **0.592 m/s**，关闭约 **0.709 m/s**。迁移程序运行成功，但这个检查点的助力会改变步态/步速，不能据此认定助力效果已经达到目标。

验证数据：`logs/sim2sim_g1_v3/validation_assist/`、`logs/sim2sim_g1_v3/validation_no_assist/`。

测试命令：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python source/legged_lab/test/test_g1_v3_sim2sim.py
```


全 NumPy 入口：`sim2real/g1_assist_exoskeleton_v3/sim2sim_numpy.py --vx 0.7 --plot-window 5`。本体和助力均用 NumPy，默认读取部署目录内的权重和配置，保留本页的实时显示功能。

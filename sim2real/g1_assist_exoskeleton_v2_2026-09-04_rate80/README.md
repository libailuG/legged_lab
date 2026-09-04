# G1 assist-exoskeleton v2 Sim2Real (2026-09-04 rate80)

本目录对应当前策略：

`logs/rsl_rl/g1_assist_exoskeleton_v2_ppo/2026-09-04_10-19-39_dynamics_tanh_gate_02_07_rate80/model_1999.pt`

导出的策略部署核心不依赖 PyTorch，提供纯 NumPy 验证实现和固化权重的 C99 实现。
真实电机通信、编码器读取、急停和故障状态机需要接入实际硬件 SDK，本目录不会假设
某一种电机总线。

## 策略接口

- 控制周期：10 ms（100 Hz），硬件电流/力矩内环可保持 1 ms。
- 网络：`150 → 256 → 64 → 16 → 2`，隐藏层 ELU。
- 输出：动作限幅到 `[-1, 1]` 后乘以 8 Nm。
- 平滑：每个策略周期最多变化 0.8 Nm，对应 80 Nm/s。
- 速度门控：`|vx|≤0.2 m/s` 时目标助力为0，`|vx|≥0.7 m/s` 时完全开放，
  中间使用与训练一致的 smoothstep 过渡。
- 左右顺序：始终为 `[left, right]`。
- 角度单位：rad；角速度：rad/s；力矩：Nm。

150 维观测仅使用外骨骼可测量量，不包含人体真实髋关节状态：

```text
obs[0:50]    = 25帧 [左外骨骼角度, 右外骨骼角度]
obs[50:100]  = 25帧 [左外骨骼角速度, 右外骨骼角速度]
obs[100:150] = 25帧 [左上一周期平滑力矩, 右上一周期平滑力矩]
```

每段历史均从最旧到最新排列。历史中的力矩是乘最终输出系数之前的平滑策略力矩，
不是电机传感器测得的实际力矩。`output_scale` 只在最后写入电机前应用，不直接改
policy、obs、历史或 80 Nm/s 平滑过程；最终输出仍限幅在 ±8 Nm。

## 生成与验证

在项目根目录运行：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2_2026-09-04_rate80/export_assist_policy_numpy.py

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2_2026-09-04_rate80/test_numpy_assist_policy.py

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2_2026-09-04_rate80/export_assist_policy_c.py

conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2_2026-09-04_rate80/test_c_assist_policy.py
```

MuJoCo 中使用纯 NumPy 助力策略测试（步态策略仍使用 TorchScript）：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
python sim2real/g1_assist_exoskeleton_v2_2026-09-04_rate80/sim2sim_numpy_assist.py \
  --assist-torque-scale 0.5 \
  --vx 0.7 --vy 0.0 --yaw 0.0
```

## C99 控制流程

将 `c/g1_assist_v2_policy.c` 和 `.h` 加入控制器工程并链接 `libm`：

```bash
gcc -std=c99 -O2 controller.c \
  sim2real/g1_assist_exoskeleton_v2_2026-09-04_rate80/c/g1_assist_v2_policy.c -lm
```

控制器复位时调用 `g1_assist_v2_history_reset()`，并将上一周期平滑力矩清零。
每 10 ms 按以下顺序执行：

1. 除复位后的第一次推理外，调用 `g1_assist_v2_history_append()`，传入当前外骨骼
   角度、角速度以及上一周期的未缩放平滑力矩。
2. 调用 `g1_assist_v2_history_build_observation()`。
3. 调用 `g1_assist_v2_policy_step()`，同时传入机器人当前的前向速度指令 `vx`
   （不是测量速度），得到经过速度门控和 80 Nm/s 平滑的策略力矩。
4. 调用 `g1_assist_v2_apply_output_scale()`，仅在最后应用测试系数。
5. 将结果送入电机力矩接口。

正常速度指令归零时，助力会按80 Nm/s斜率退回零；急停、通信超时或故障状态必须
由上层安全逻辑绕过该斜率限制并立即给电机零力矩/失能。

后部滑块和圆柱机构的零位 PD 不属于这个 2 输出策略，需要由硬件控制层单独实现。

## 文件说明

- `weights/assist_policy_v2.npz/json`：当前策略权重和接口清单。
- `numpy_assist_policy.py`：无 PyTorch 的 NumPy 推理、历史、力矩平滑和可直接接入
  100 Hz 硬件循环的 `V2AssistController`。
- `c/g1_assist_v2_policy.c/.h`：无动态内存的 C99 固定策略。
- `test_numpy_assist_policy.py`：TorchScript 与 NumPy 数值对比。
- `test_c_assist_policy.py`：NumPy 与生成 C99 数值对比。
- `sim2sim_numpy_assist.py`：用 NumPy 助力策略进行 MuJoCo 验证。

## 上真实硬件前

先在空载台架从 `output_scale=0` 开始，确认左右方向、单位、编码器零位、10 ms
调度、±8 Nm 限幅、通信超时归零和硬件急停均正确，再逐步提高系数。不要直接在人
体穿戴状态下首次测试未验证的电机接口。

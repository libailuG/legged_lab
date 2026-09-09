# G1 外骨骼 v3 Sim2Real

独立部署目录，参考已有 sim2real 的接口形式；不包含 CAN/串口/电机 SDK，不会向硬件发送命令。

本目录已导出并逐项核对 `2026-09-08_10-06-19_paired_assist_10nm_body_pid/model_2000.pt` 的 Actor 权重及归一化状态。`weights/manifest.json` 保存来源、SHA256 和训练控制参数。人体外骨骼只用辅助控制器；G1 步态及 PID 是可选机器人联调接口。

## 文件与依赖

- `controller.py`：100 Hz 外骨骼控制器，只依赖 NumPy 和标准库。
- `numpy_policy.py`：150 → 256 → 64 → 16 → 1 的 ELU 网络及 25 帧历史。
- `weights/assist_policy.npz`：便携 NumPy 权重；`assist_policy.pt`：对应 TorchScript。
- `c/g1_assist_v3_policy.c/.h`：固定权重 C99 网络、历史和完整助力后处理，无堆分配、无文件读写，链接 libm 即可。
- `c/example_assist.c`、`demo.py`：合成编码器输入演示，不驱动电机，不保存数据。
- `gait_controller.py`：可选 G1 本体 50 Hz 推理、1000 Hz PID、背部机构 PD，依赖 PyTorch。
- `weights/gait_policy.pt`：冻结本体 `model_51800.pt` 对应导出，无本体观测噪声。
- `export_policy.py`、`export_policy_c.py`：重建部署权重和 C 文件。
- `test_deployment.py`：离线验证，需 PyTorch、MuJoCo、NumPy 和 gcc，运行时不启动仿真窗口。

## Python 助力调用

```python
from sim2real.g1_assist_exoskeleton_v3.controller import AssistController

controller = AssistController()
controller.reset([q_left, q_right], [dq_left, dq_right])
# 每 10 ms 用同一时刻的左右传感器样本调用一次：
torque_nm = controller.step(
    [q_left, q_right], [dq_left, dq_right], timestamp=sample_time_seconds
)
# torque_nm 顺序为 [left, right]，在外部适配层转换并发送。
# 停止时：
zero_nm = controller.stop()
```

输入角度是训练关节坐标系的绝对关节角（rad），速度为 rad/s。硬件层负责电机 ID、零位、左右正方向、减速比和力矩单位转换，不能直接把当前穿戴姿态当作零位。角度/速度/力矩方向变换必须一致。

历史是 term-major：25 帧左右角度，25 帧左右角速度，25 帧左右上一周期处理后的指令力矩，各段最旧到最新。首帧复制填充 q/dq，力矩历史清零。每次 step 使用当前编码器和上一周期实际返回的指令更新历史后推理。

只接收两个外骨骼关节的 q/dq，不接收脚触地、髋状态、人体状态、vx 或 G1 本体观测。网络输出一个有符号动作，经共同速度低通（0.05 s）、共同速度门控（0.15～0.8 rad/s）处理后，输出 `[T,-T]`，双方统一 ±10 N·m，变化速率 80 N·m/s（每 10 ms 最大 0.8 N·m），换向先经过零。没有旧版的单侧门控、±8 N·m 限制或 +4 N·m 下压上限。

`enabled=False` 或 `stop()` 返回零并锁定；重新运行须显式 reset。NaN/Inf、错误形状、错误输出会清零内部控制状态并抛异常。可选时间戳检测非递增和超过 5～15 ms 的采样间隔；开启时间戳后不允许中途省略。必须按 100 Hz 调度，容差不表示可以按变步长运行。

stop 只返回零，不发送硬件指令。外部循环需要在异常/禁用时发送零或失能，并处理样本陈旧、断线、错过周期等故障；进程卡住时 Python 无法执行 stop，硬件侧应有独立 watchdog。该接口的普通输出是完整训练力矩，不提供额外缩放开关。每个实例只能由一个控制线程使用。

## C99 助力调用

使用独立、静态初始化的 `G1AssistV3Controller`：

```c
static G1AssistV3Controller controller;
float torque[2];
g1_assist_v3_reset(&controller, position_rad, velocity_rad_s);
/* 每 10 ms 调用；enabled=1 */
int status = g1_assist_v3_step(&controller, position_rad, velocity_rad_s, 1, torque);
/* 退出或异常： */
g1_assist_v3_stop(&controller, torque);
```

reset 返回 0 成功，-1 输入错误。step 返回 0 成功，1 禁用/未复位，-1 输入/推理错误；禁用或故障返回零并锁定。C 接口不带时间戳，外部调度器负责采样时效和通信故障。低层 `policy_forward` 仅用于网络数值验证，部署优先调用完整 `step`。C 输出仅含辅助网络与控制器，不包含 G1 步态/PID。

## 离线命令

在仓库根目录运行：

```bash
python sim2real/g1_assist_exoskeleton_v3/demo.py --seconds 10

gcc -std=c99 -O2 -Wall -Wextra -Werror \
  sim2real/g1_assist_exoskeleton_v3/c/example_assist.c \
  sim2real/g1_assist_exoskeleton_v3/c/g1_assist_v3_policy.c \
  -lm -o /tmp/g1_assist_v3_example
/tmp/g1_assist_v3_example

OPENBLAS_NUM_THREADS=1 /home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v3/test_deployment.py
```

更换检查点时先用 Isaac Play 导出对应 `exported/policy.pt`，然后执行：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v3/export_policy.py \
  --run /绝对路径/新训练目录 --checkpoint model_XXXX.pt
python sim2real/g1_assist_exoskeleton_v3/export_policy_c.py
```

导出脚本拒绝与检查点不匹配的 TorchScript；C 生成器拒绝与当前固定后处理参数不一致的配置。重新生成会替换本目录部署权重，随后必须重跑验证。

## 可选 G1 机器人接口

`GaitPIDController.reset()` 后每 1 ms 调用 `step(q,dq,gyro,gravity,command)`，返回 29 关节力矩。关节顺序见 `POLICY_JOINT_NAMES`。gyro 为 pelvis 局部角速度，gravity 为局部单位重力方向，command 为 `[vx,vy,yaw]`。每 20 个物理周期更新目标，其余保持；观测无噪声。积分限幅 ±10 N·m，条件抗饱和，重置清零。不要在力矩模式外再叠加电机位置 PD。

`rear_mechanism_pd([slider_m,cylinder_rad], [slider_m_s,cylinder_rad_s])` 返回 `[滑块力N,圆柱力矩Nm]`，不是两个电机力矩。滑块传动换算由外部适配层完成。人体外骨骼不需要这两个接口。

## 已验证范围

200 组随机输入下 C 与 TorchScript 原始动作最大误差约 6e-7；1500 步含运动/停止序列下 C 与 NumPy 的力矩最大误差约 6e-6 N·m，并对比 Sim2Sim 的历史与后处理。验证成对输出、±10 限幅、0.8 N·m 步间限速、故障停止、时间戳异常及可选 G1 PID 与仿真参数一致性。

这些是离线计算验证，不代表已验证实机稳定性、标定或实时截止时间。


## 全 NumPy Sim2Sim

本体和外骨骼网络都使用 NumPy 推理，运行时不导入 PyTorch；导出和对比测试仍需 PyTorch。MuJoCo 物理/PID 1000 Hz，本体策略 50 Hz、外骨骼策略 100 Hz。无本体观测噪声，外骨骼直接使用本目录 AssistController，默认策略为已核对的 model_2000.pt。

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v3/sim2sim_numpy.py --vx 0.7 --plot-window 5
```

保留实时滚动曲线、归一化力矩对比、峰值、独立速度滑条；隐藏 MuJoCo 两侧面板。只显示、不保存数据，持续运行，跌倒自动重置，关闭窗口或 Ctrl+C 退出。`--disable-assist` 用于零助力对照。可用 `--weights-dir` 指定另一份完整导出目录；默认直接读取本目录 weights/env.yaml，不依赖旧训练目录。不要传入 TorchScript 的 --gait-policy/--assist-policy。

依赖 NumPy、MuJoCo、Matplotlib、PyYAML、tkinter。入口复用仓库 scripts/mujoco 的显示与仿真代码，需要在当前仓库使用；独立部署核心仍可单独使用。256 组随机本体观测与 TorchScript 对比通过，开启和关闭助力的 GUI 检查均通过且确认未导入 torch；短时检查不代表长时间稳定性验证。

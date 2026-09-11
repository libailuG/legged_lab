# v7 外骨骼 Sim2Real

本目录包含独立 NumPy 控制器、等价 C99 控制器和已导出的部署权重。不包含电机通信；沿用现有实机程序，将编码器读数送入控制器，再将返回值交给驱动。

当前权重来自 `2026-09-10_09-25-22_faster_response/model_500.pt`（500轮）。来源、SHA256、控制参数见 `weights/manifest.json`。这是本次v7响应优化训练的模型。6～10 N·m是训练奖励目标，部署不强制最低6 N·m，也不额外放大输出。

当前包使用保存的训练配置：速度滤波30 ms、启动确认30 ms、上升200 ms、释放250 ms；对侧抬腿释放仍需60 ms确认，候选状态按60 ms归一化。旧2000轮、50 ms滤波部署包已完整备份至 `../g1_assist_exoskeleton_v7_2026-09-09_model1999/`。C生成器会拒绝与当前固定模板不一致的参数，后续若训练参数变化必须同步模板并重新验证。

## 输入与输出

- 100 Hz，每10 ms调用一次。输入为已校准至训练坐标的左右关节角度（rad）与角速度（rad/s），顺序始终为左、右。
- Actor只用本地关节信号、历史输出和控制器状态；不需要足端接触、本体状态或目标行走速度。
- 25帧历史共450维：角度50、角速度50、已处理力矩50、控制器状态300。同一项按时间从旧到新排列。
- 网络输出2个动作，决定单次助力的峰值0～10 N·m与总时长0.6～1.4秒。触发时锁定，后续网络输出不会反复修改正在执行的脉冲。
- 输出 `[T, -T]`，各侧限制±10 N·m，变化率限制80 N·m/s。上升段0.20秒、释放段0.25秒，中间可保持；反向运动、对侧抬腿或静止条件可提前释放。
- 不再用速度门控乘小峰值，但抬腿触发仍有速度阈值，极慢动作仍可能无法触发。
- 训练中的±10°随机零偏不在部署端添加。实机应完成角度零位与方向标定，残余误差由训练鲁棒性覆盖。

## Python接入

从仓库根目录导入，仅外骨骼推理依赖NumPy：

```python
from sim2real.g1_assist_exoskeleton_v7.controller import AssistController

controller = AssistController()
controller.reset([q_left, q_right], [dq_left, dq_right])

# 每10 ms执行，sample_time_s为编码器采样时间（秒）。
torque = controller.step(
    [q_left, q_right], [dq_left, dq_right], timestamp=sample_time_s
)
# 将torque[0]、torque[1]转换到驱动坐标后发给电机。

# 退出时将此返回的零力矩实际发送到驱动。
zero_torque = controller.stop()
```

首次调用填充历史。停用或异常后必须重新reset。非法输入会清空控制状态并抛出异常，调用方应发送零力矩或停机；函数本身不会操作电机。若提供时间戳，相邻间隔必须在5～15 ms，且后续不能省略时间戳。控制器始终按固定10 ms计算，不自动补偿丢帧。

离线演示使用合成关节运动，只打印结果，不写CSV、不连接硬件：

```bash
cd /home/libai/08_amp/legged_lab
python sim2real/g1_assist_exoskeleton_v7/demo.py --seconds 10
```

## C99接入

将 `c/g1_assist_v7_policy.c` 和 `.h` 加入工程并链接libm，权重已嵌入源码，无动态内存分配、文件读取或Python依赖。

使用 `g1_assist_v7_reset`、`g1_assist_v7_step`、`g1_assist_v7_stop`，完整示例见 `c/example_assist.c`。step返回0表示成功、1表示停用或未reset、-1表示输入异常；后两种情况输出为零。C接口不检查时间戳，采样周期与通信看门狗由调用方维护。

```bash
gcc -std=c99 -O2 -Wall -Wextra -Werror \
  sim2real/g1_assist_exoskeleton_v7/c/example_assist.c \
  sim2real/g1_assist_exoskeleton_v7/c/g1_assist_v7_policy.c \
  -lm -o /tmp/g1_assist_v7_example
/tmp/g1_assist_v7_example
```

v7的450维输入、双动作和12维历史状态与v4不兼容，必须成套替换控制器和权重。

## 重新导出与验证

若换检查点，先通过对应训练任务的导出流程生成该检查点的 `exported/policy.pt`。下面脚本逐项核对它与检查点的Actor及归一化参数，一致后才导出：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v7/export_policy.py
python sim2real/g1_assist_exoskeleton_v7/export_policy_c.py
OPENBLAS_NUM_THREADS=1 /home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v7/test_deployment.py
```

export_policy.py默认使用上面的最新500轮模型，可通过 `--run RUN_DIR --checkpoint model_N.pt` 指定其他模型。导出后需要重新生成C源码；不要只替换NPZ而继续使用旧C权重。

已通过5项测试，覆盖网络与归一化、1500步完整历史/脉冲闭环对齐、异常与停用、可选本体PID及本体NumPy推理。C与NumPy力矩最大误差约9.54e-7 N·m。该验证证明与仿真控制实现的一致性，不是实机穿戴效果验证。

`gait_controller.py`、`numpy_gait_policy.py`及gait权重为可选G1本体接口，使用原model_51800、50 Hz策略和1 kHz PID；人体穿戴外骨骼无需调用它们。C核心只包含外骨骼控制。

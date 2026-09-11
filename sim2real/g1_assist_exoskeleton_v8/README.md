# v8 Sim2Real

当前权重：`2026-09-10_13-38-04_motion_guard_cadence/model_499.pt`（500轮）。权重与观测归一化已逐项核对，来源和SHA256见 `weights/manifest.json`。包含NumPy/C99控制器，不包含电机通信。

## 接入

输入左右外骨骼关节角度（rad）、角速度（rad/s），顺序左、右，需校准至训练关节坐标。100 Hz调用，输出 `[T,-T]`（N·m），各侧±10 N·m。局部外骨骼信号以外，不需要足接触、本体状态或目标速度。

```python
from sim2real.g1_assist_exoskeleton_v8.controller import AssistController

controller = AssistController()
controller.reset([q_left, q_right], [dq_left, dq_right])
# 每10 ms调用；输入应为同一采样时刻的数据。
torque = controller.step([q_left, q_right], [dq_left, dq_right], timestamp=sample_time_s)
# 将左右力矩换算到驱动坐标并实际发送。
zero_torque = controller.stop()
```

stop只返回零力矩，不操作电机。异常会清空状态并停用，调用方应实际发送零力矩或停机；再次启动需要reset。若使用时间戳，相邻采样需在5～15 ms且不可中途省略；计算周期始终为10 ms。通信看门狗由调用方负责。

部署不添加训练随机±10°偏置，不额外放大力矩，不写CSV。

## v8机制

- 30 ms滤波、30 ms速度方向确认，还需候选方向下连续相对屈曲位移：首次3°、近期有步态时2.5°。1.5秒无新合格交替事件后恢复首次门槛。
- 从合格交替事件估计步间隔，步频有效后总时长不超过估计间隔85%，预算限制0.3～1.4秒。峰值与时长在脉冲启动时锁定。
- 上升不超过200 ms、释放不超过250 ms，分别不超过总时长45%。峰值按曲线可行性限制，使变化率不超过80 N·m/s；快步可能降低峰值。6～10 N·m是训练目标，不是强制最低出力。
- 提前释放根据实际当前输出缩短拖尾。对侧反向释放仍需60 ms确认。旧脉冲释放到零后才允许新脉冲。
- 位移来自局部关节变化，不能完全分辨主动与被动运动；首步确认会增加启动等待。

Actor650维：25帧角度50、速度50、处理后力矩50、控制状态500。每项按时间从旧到新排列。每帧控制状态包含归一化18维脉冲状态及2维滤波速度；原始候选角度参考在观测中替换为相对位移。C控制器额外保存该位移，保证下一步历史使用前一控制步的状态。

网络结构650→256→64→16→2，ELU。旧v7的450维网络及状态布局不兼容，必须完整替换控制器、权重和头文件。

## 离线演示与C99

```bash
cd /home/libai/08_amp/legged_lab
python sim2real/g1_assist_exoskeleton_v8/demo.py --seconds 10

gcc -std=c99 -O2 -Wall -Wextra -Werror \
  sim2real/g1_assist_exoskeleton_v8/c/example_assist.c \
  sim2real/g1_assist_exoskeleton_v8/c/g1_assist_v8_policy.c \
  -lm -o /tmp/g1_assist_v8_example
/tmp/g1_assist_v8_example
```

演示为合成输入，不驱动电机。C使用 `g1_assist_v8_reset/step/stop`，step返回0成功、1停用/未初始化、-1异常。失败输出零并锁止；无时间戳检测，采样周期由调用方保证。C源码内嵌权重，无文件I/O或动态内存，链接libm。网络参数736712字节float32，推理工作区3624字节，不含历史等控制状态。

## 更新权重及测试

先从目标检查点生成对应运行的 `exported/policy.pt`，再执行：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v8/export_policy.py
python sim2real/g1_assist_exoskeleton_v8/export_policy_c.py
OPENBLAS_NUM_THREADS=1 /home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v8/test_deployment.py
```

export_policy默认使用上述model_499，支持 `--run RUN_DIR --checkpoint model_N.pt`。C生成器要求固定控制参数与当前模板一致；修改训练脉冲逻辑后必须同步部署逻辑并重新验证。

6项测试通过：网络与归一化、1500步完整闭环历史一致性、2100步小摆动/快步测试、异常与停用、本体PID与本体NumPy。C/NumPy基础闭环最大力矩误差约1.19e-6 N·m。测试证明实现一致，不是实机效果验证。

`gait_controller.py`、`numpy_gait_policy.py`和gait权重是可选G1本体接口（model_51800，50 Hz策略/1 kHz PID）。人体穿戴外骨骼无需调用；C核心只含外骨骼控制。

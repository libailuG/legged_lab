# v4 单峰助力 Sim2Real

本目录是独立部署核心，参考此前 sim2real 结构，不接入硬件通信，也不会向电机发送命令。原 v3 目录保留。

默认策略：`2026-09-08_15-42-59_single_pulse_04s/model_499.pt`（500轮训练）。Actor 参数及归一化状态已逐项对比导出的 `policy.pt`。来源、SHA256、控制参数记录于 `weights/manifest.json`。

## 文件

- `controller.py`：完整100 Hz辅助控制器，只依赖NumPy与标准库。
- `pulse.py`：与训练SinglePulse对应的NumPy单峰发生器。
- `numpy_policy.py`：400→256→64→16→1，ELU，包含训练观测归一化。
- `weights/assist_policy.npz`、`assist_policy.pt`：NumPy与TorchScript辅助权重。
- `c/g1_assist_v4_policy.c/.h`：固定权重C99网络、历史和完整脉冲控制器，无堆分配、无文件读写。
- `demo.py`、`c/example_assist.c`：合成编码器输入演示，无硬件操作，不保存数据。
- `gait_controller.py`、`numpy_gait_policy.py`、`weights/gait_policy.*`：可选G1机器人联调接口；人体外骨骼不需要它们。本体策略为51800，50 Hz，PID 1000 Hz，无本体观测噪声。
- `export_policy.py`、`export_policy_c.py`：重新导出与生成C文件。
- `test_deployment.py`：开发验证，需要PyTorch、MuJoCo、gcc，不启动仿真窗口。

## 控制定义

每10 ms采样左右外骨骼关节角度rad、速度rad/s，次序固定[left,right]。输入必须转换到训练关节坐标，角度是绝对关节角，不应直接把穿戴姿态当作零位。硬件层负责电机ID、零位、正方向、减速比及力矩换算。

400维观测按项排列：

1. 25帧左右角度，共50维。
2. 25帧左右角速度，共50维。
3. 25帧左右上一周期处理后的指令力矩，共50维。
4. 25帧10维控制器状态，共250维。

每段最旧到最新。10维状态依次为：归一化脉冲进度、活动方向、峰值/10、释放进度、释放起点/10、已消耗方向、候选方向、确认时长/0.03、左右滤波速度。状态来自软件自身，不是人体或足端传感器。首帧填充角度/速度，力矩和脉冲状态清零。

- 编码器速度做0.05 s低通；较快屈曲腿速度<-0.2 rad/s、左右速度差绝对值>0.3 rad/s，方向持续0.03 s后允许触发。
- Actor输出u∈[-1,1]，仅选择峰值：A=10×(u+1)/2×起始门控。u=0对应半幅；u=-1才是零峰值。方向由编码器规则决定。
- 起始门控使用两侧滤波速度绝对值的最大值，0.15～0.8 rad/s smoothstep。
- 触发时锁定幅值，执行0.4 s sin²单峰。周期中策略动作/门控变化不会反复改峰值。
- 确认停止/反向后，以当前力矩为起点用0.2 s cos²单调释放；释放结束前不启动反方向。
- 同方向不反复触发，需要安静再运动或新的反方向。左右始终[T,-T]，±10 N·m，常规/释放曲线最大斜率约78.54 N·m/s，不超过80。

不接收脚触地、人体髋关节、本体状态或vx，不在实机端重新计算训练奖励。当前版本的脉冲宽度固定，仍需要实测评估时机和体感。

## Python接口

```python
from sim2real.g1_assist_exoskeleton_v4.controller import AssistController

controller = AssistController()
controller.reset([q_left, q_right], [dq_left, dq_right])
# 每10 ms，以同一时刻的两侧编码器输入调用：
torque = controller.step([q_left, q_right], [dq_left, dq_right], timestamp=sample_time_s)
# torque为左右关节N·m；由外部适配层转换并发送。
zero = controller.stop()
```

`enabled=False`/stop返回零并锁定，恢复必须reset；NaN/Inf、形状错误、推理异常会清零内部状态、锁定并抛异常。可选时间戳检测非递增或不在5～15 ms容差内的采样间隔，开启后不得中途省略时间戳。状态机仍按固定10 ms执行，不支持变步长。

stop只修改内存并返回零，不写电机。外部适配层须发送零/失能，处理数据陈旧、超时和断线；独立硬件watchdog处理进程无法响应的情况。每个控制器由单一线程使用。普通输出是完整训练幅值，没有额外输出缩放。

## C99接口

```c
static G1AssistV4Controller controller;
float output_nm[2];
g1_assist_v4_reset(&controller, position_rad, velocity_rad_s);
/* 每10 ms，enabled=1 */
int status = g1_assist_v4_step(&controller, position_rad, velocity_rad_s, 1, output_nm);
g1_assist_v4_stop(&controller, output_nm);
```

reset返回0成功/-1错误；step返回0成功、1禁用/未复位、-1输入或推理错误。故障或禁用时输出零并锁定。C接口的采样时效检查由外部调度器负责。低层policy_forward只做网络推理，集成优先使用完整step。单独pulse历史函数需要传入10维状态，不可沿用v3历史接口。

C仅包含辅助网络及控制器；可选G1本体仍通过Python接口。固定网络参数约480644字节，另外有归一化参数；推理workspace2624字节，完整控制器含历史约6KB，推荐使用static实例。

## 离线运行与编译

```bash
cd /home/libai/08_amp/legged_lab
python sim2real/g1_assist_exoskeleton_v4/demo.py --seconds 10

gcc -std=c99 -O2 -Wall -Wextra -Werror \
  sim2real/g1_assist_exoskeleton_v4/c/example_assist.c \
  sim2real/g1_assist_exoskeleton_v4/c/g1_assist_v4_policy.c \
  -lm -o /tmp/g1_assist_v4_example
/tmp/g1_assist_v4_example

OPENBLAS_NUM_THREADS=1 /home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v4/test_deployment.py
```

## 更新策略

先用v4 Isaac Play导出新检查点，再执行：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_assist_exoskeleton_v4/export_policy.py \
  --run /绝对路径/新训练目录 --checkpoint model_XXXX.pt
python sim2real/g1_assist_exoskeleton_v4/export_policy_c.py
```

导出器验证TorchScript与指定检查点一致。C生成器校验权重哈希，并拒绝与当前固定脉冲参数不符的配置。重新导出会替换本目录权重，之后应重跑测试。v3的150维策略及控制器不能用于v4。

## 验证范围

5项离线验证通过：200组随机观测C/NumPy/Torch网络一致性、1500步运动/停止序列和完整400维历史对比、故障/时间戳/停止锁定、可选G1 PID参数与仿真一致性，以及本体NumPy网络对比。

C与TorchScript原始辅助动作最大误差约7.2e-7；C与NumPy辅助力矩最大误差约9.6e-7 N·m。尚未验证真实电机、传感器标定、实时截止时间或人体体感。

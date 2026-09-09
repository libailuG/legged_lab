# G1 PID + 外骨骼 v2-v2 Sim2Real 控制核心

本目录独立保存策略文件和 Python 部署代码，原有版本不变。
当前完成到“可集成、已离线验证的控制核心”；尚无电机 SDK 适配器，不能直接连接硬件。
仅部署人体外骨骼时只使用 AssistController，不需要 G1 步态策略、速度指令或机器人状态。

## 策略

- G1 步态来自 LeggedLab-Isaac-AMP-G1-assist-v1：
  2026-09-04_20-03-58_rtx5090_bounded_noise_entropy0_remaining_29600/exported/policy.pt。
- 助力来自 v2-v2：
  2026-09-04_17-54-31_lift10_rate80_press4_rate40_motion_gate_resume/exported/policy.pt。
- weights/provenance.json 记录源路径和 SHA256；部署使用目录内副本。
- G1 推理依赖 PyTorch；助力推理仅依赖 NumPy。无需 Isaac Lab 或 MuJoCo。
- test_deployment.py 是仓库内开发验证，需要 MuJoCo 仿真脚本作为对照，但不会启动仿真。

## 助力接口

controllers.py 的 AssistController：
reset(position, velocity) 初始化；step(position, velocity) 每 10 ms 调用一次，
输入始终为 [left, right]，单位 rad 和 rad/s。角度是训练关节坐标系下的绝对关节角，
不能直接以当前穿戴姿态作零位；硬件适配层必须完成零位、减速比、左右方向变换。
禁止添加 vx 门控。

观测：25帧左右位置、25帧左右速度、25帧左右未缩放的平滑请求力矩，各段从旧到新。
关节速度用 tau=0.05s 低通；绝对速度 <=0.15 rad/s 关闭门控，>=0.8 rad/s 完全开启。
策略目标 [-10,+4] Nm；请求力矩向负方向变化最大 80 Nm/s，向正方向最大 40 Nm/s。
这按力矩增量方向选择，包括退回零的过程。

当前 checkpoint 的 Isaac 物理执行器仍为 +/-8 Nm，所以最终硬件输出限制为 [-8,+4] Nm。
这保持已有策略物理限制；不能声称该 checkpoint 已在真实 -10 Nm 执行器上训练。
历史保存限幅前的平滑请求，与当前 Sim2Sim 一致。output_scale 默认 0，仅允许 [0,1]；
输出为 clip(scale * request, -8,4)。更改比例需停止、reset 后开始，避免输出突变。
stop() 和 enabled=False 立即返回零并锁定；须显式 reset 后才能继续。
非有限输入会清空控制器状态并抛异常，上层必须捕获并给硬件零力矩/失能。

## G1 与机构接口

GaitPIDController.step(q,dq,gyro,gravity,command) 每 1 ms 调用，内部每20次执行步态策略。
29关节顺序见 controllers.py 中 POLICY_JOINT_NAMES；姿态输入在 pelvis 局部坐标系，
gravity 为单位重力方向；command=[vx m/s,vy m/s,yaw rad/s]。
PID：Ki=0.1*Kp，积分项限制 +/-10 Nm，条件抗积分饱和；reset 清空积分和上一动作。
返回29维力矩，由硬件适配器映射电机ID。不要再叠加硬件位置模式 PD。
rear_mechanism_pd 接收 [滑块位置m,圆柱角度rad]、对应速度，返回 [N,Nm]。
其目标为零，Kp=[10000,4]，Kd=[20,2]，输出限制 [300N,200Nm]。
真实电机若驱动滑块，需要适配层做丝杆或传动的力到电机力矩转换。

组合调度：硬件层1kHz读取状态并执行 G1 PID/后部PD，每10个周期调用助力核心，
其余周期保持助力输出。关节样本必须来自同一时刻。
Python测试不证明满足真实1kHz截止时间；硬件层需检测超时、样本陈旧及通信故障，
独立执行零力矩/失能。stop方法不能在进程卡住或断网时自动向电机发送指令。

## 离线运行

在仓库根目录：
```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_pid_assist_exoskeleton_v2_v2/test_deployment.py

/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_pid_assist_exoskeleton_v2_v2/replay_assist.py \
  --output /tmp/assist_v2v2_replay.csv --output-scale 0.2
```

replay 默认用合成编码器信号，不驱动电机。可传 --input 读取实测CSV：
time,left_position_rad,right_position_rad,left_velocity_rad_s,right_velocity_rad_s。
时间单位秒，严格100Hz，无缺帧。输出路径必须不存在，防止覆盖历史记录。
无助力硬件操作应禁用输出（enabled=False）；output_scale=0 用于离线或内部策略观察。

## 验证范围

128组随机观测：NumPy与TorchScript网络数值一致；
600步助力：门控、历史、平滑和最终输出与现有Sim2Sim一致；
100步G1 PID：与assist-v1参考控制器一致；停止及NaN输入测试通过。
还未验证真实硬件通信、传感器标定和控制周期。
两策略未联合训练，更换步态后的性能仍需评估。

接入实际电机还需要：硬件SDK/已有通信程序、左右电机ID、单位和零位转换、
力矩或电流发送接口、状态时间戳、故障/急停接口。

## C99 固定权重部署（已生成）

- c/g1_assist_v2_v2_policy.c：56178个固化网络参数及控制逻辑。
- c/g1_assist_v2_v2_policy.h：网络、历史和完整100Hz助力控制器接口。
- c/example_assist.c：可编译运行的离线调用示例，无硬件通信。
- export_assist_policy_c.py：从本目录NPZ重新生成.c/.h。
- test_c_assist_policy.py：编译C99并与NumPy逐周期比较。

本次C转换覆盖左右外骨骼助力策略，与其他Sim2Real目录的C输出范围相同。
G1的29关节步态网络及PID仍在Python控制器中，未包含在这个C文件内。
C99运行无需Python、PyTorch、NPZ、文件系统或动态内存，编译时链接libm。
参数约224712字节，另有归一化参数；控制器状态约2.8KB，建议用static实例。

编译并运行离线示例（项目根目录）：

```bash
gcc -std=c99 -O2 -Wall -Wextra -Werror \
  sim2real/g1_pid_assist_exoskeleton_v2_v2/c/example_assist.c \
  sim2real/g1_pid_assist_exoskeleton_v2_v2/c/g1_assist_v2_v2_policy.c \
  -lm -o /tmp/g1_assist_v2_v2_example
/tmp/g1_assist_v2_v2_example
```

重新生成和测试：

```bash
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_pid_assist_exoskeleton_v2_v2/export_assist_policy_c.py
OPENBLAS_NUM_THREADS=1 /home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  sim2real/g1_pid_assist_exoskeleton_v2_v2/test_c_assist_policy.py
```

调用顺序：g1_assist_v2_v2_reset → 每10ms g1_assist_v2_v2_step →
将motor_nm发送至硬件。step输入position/velocity/enabled，不接收vx。
reset返回0成功、-1参数错误；step返回0成功、1禁用/未复位、-1输入或推理错误。
禁用或错误时输出零、锁定停止，恢复须reset。stop只修改内存，不自动发送电机命令。
每个控制实例使用独立状态，不可同时在多个线程上调用同一实例。
首次调用前必须reset；不要直接修改output_scale，需通过reset重新设置。
低层policy_forward仅是网络推理，不包含输入故障检查，硬件接入优先用完整step接口。

验证结果：200组随机输入最大误差约4.2e-7；3种输出比例、共3000步控制序列
最大输出误差约2.6e-6 Nm；停止锁定、重新复位和NaN输入检查通过。


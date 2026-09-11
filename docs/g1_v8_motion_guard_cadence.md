# v8 位移确认与步频适配

从v7独立创建训练和Isaac Play任务，针对站立小幅摆动误触发与快走脉冲错相。v7及已有部署包不变。本版没有提前预测人的意图，使用外骨骼本地角度、滤波速度与控制历史。

## 触发与站立重新锁止

沿用30 ms速度滤波、30 ms方向确认与原速度阈值，但速度达标不再足以启动。方向候选变化时记录左右当前角度，抬升侧从该参考点连续屈曲至少3°才允许首次触发。有近期有效步态时使用2.5°门槛；1.5秒没有新的合格交替方向事件后，清除步频有效标志，回到首次启动门槛。

门槛使用相对角度变化，不依赖竖直零位。左右±10°偏置仍每回合独立固定，并加在输入局部角度上；策略中额外的位移状态同样为相对量。反向或速度候选中断后重新确认，不累加往返抖动的总路程。

此机制能拒绝已测试的小幅摆动，不是对所有机构松动和自激的保证。超过门槛的被动外骨骼运动仍可能触发，仅靠关节信号无法完全分辨人体意图。门槛会增加首步延迟，尤其对极慢、小步幅动作，需要实测权衡。

## 快步时序与力矩约束

记录相邻合格反向抬升事件的间隔（半个步态周期），只使用0.2～1.2秒的间隔；首个有效间隔直接采用，之后用0.5系数平滑。未获得间隔前，保留原0.6～1.4秒时长选择。

步频有效后，总时长上限为间隔的85%，限制在0.30～1.4秒，与策略选择时长取较小值。上升时间取原200 ms与总时长45%的较小值，释放时间取原250 ms与总时长45%的较小值。每次脉冲开始时锁定时长和峰值，中途不反复修改。

为保持80 N·m/s限制，峰值不能超过 `2 × 80 × min(上升时间,释放时间) / π`，同时不超过10 N·m。因此快走时可能降低峰值来换取正确时序，并非继续追求每次10 N·m。旧脉冲仍需释放至零才允许新脉冲，没有力矩瞬时跳变。

提前释放从当前实际输出幅值开始，释放时长缩短到满足变化率的时间（至少20 ms，且不超过原释放时间），减少小力矩拖尾。下落、对侧反向确认60 ms、静止、超时的释放条件保留。

## 奖励与观测

- 保留有效阶段6～10 N·m目标与末端姿态支撑，以及无支撑下压、下落阻力、平滑、背板稳定等原奖励。
- 完成脉冲峰值目标按该脉冲可实现的峰值上限裁剪，避免要求快步短脉冲实现物理约束之外的峰值。
- 新增 `assist_standing_false_start`，权重-4：训练指令为站立、本体平面速度低于0.15 m/s、两髋速度均低于0.12 rad/s时，惩罚开始脉冲的事件（包括零峰值触发）；按dt归一化。只在训练评价读取本体和指令，不进入Actor。
- 内部状态由10维增为18维，增加相对位移、事件间隔、步频估计和锁定时间等。Actor为650维，Critic为1325维，仍是25帧历史、双动作、100 Hz。
- 必须从头训练，v7检查点及v7部署控制器不能直接用于v8。

新增日志（除原Response/Pulse日志）：

- `Response/displacement_blocked_fraction`：当前速度确认但位移不足的环境比例。
- `Response/cadence_valid_fraction`：步频估计有效的环境比例。
- `Response/peak_cap_start_fraction`：当前开始脉冲中被时序可行上限降低峰值的比例；无开始事件时记0。
- `Response/estimated_step_interval_s`：有效环境的平均估计交替步间隔；无有效环境时记0。

关注延迟日志、实际峰值和释放原因，并结合有效样本数量判断；这些指标并不保证真人步态同步。

## 验证

5项CPU回归测试通过：0.5°/1°、0.5/2/4 Hz小摆动10秒不启动；首次位移门槛与左右独立常量偏置不变性；0.35秒交替步间隔的时序适配和80 N·m/s限幅；停步重新锁止；峰值锁定及同方向不重复触发。32环境1轮Isaac训练和PPO更新完成，650/1325维观测、15项奖励和新增日志成功运行。

合成信号和短训练验证只确认实现可运行及上述性质，不代表快走实机效果已经验证。本次仅完成训练/Isaac Play任务，未生成v8 Sim2Sim或Sim2Real。

## 训练指令

保留vx 0.4～3.0 m/s、站立10%、本体model_51800/PID/无观测噪声。默认1024环境、rollout192、8个mini-batch。

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v2-v8 \
  --num_envs 1024 --max_iterations 10000 --headless \
  --run_name motion_guard_cadence
```

日志：`logs/rsl_rl/g1_assist_exoskeleton_v2_v8_ppo/<时间>_motion_guard_cadence/`。

Play任务：`LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v2-v8`。

## 500轮Sim2Sim

已导出并验证 `2026-09-10_13-38-04_motion_guard_cadence/model_499.pt`。新入口使用650维历史（每帧18维脉冲状态加2维滤波速度），将本地关节角度和速度送入实际v8脉冲模块，保留位移门槛与步频适配。

```bash
cd /home/libai/08_amp/legged_lab
/home/libai/anaconda3/envs/env_isaaclab_2/bin/python \
  scripts/mujoco/sim2sim_g1_assist_exoskeleton_v2_v8.py --vx 0.7 --plot-window 5
```

默认加载上述运行的env.yaml与exported/policy.pt。单机器人、实时滚动曲线、独立速度滑条；不添加随机角度偏置，不保存CSV。关闭任一窗口或Ctrl+C退出。Sim2Real尚未生成。

## Sim2Real部署

现已为上述500轮模型生成 `sim2real/g1_assist_exoskeleton_v8/`，包含独立NumPy与C99、650维权重及来源清单。6项测试通过，含小幅摆动、快步以及完整历史与Sim2Sim对齐；原v7部署保持不变。接入方法见 [v8 Sim2Real说明](../sim2real/g1_assist_exoskeleton_v8/README.md)。

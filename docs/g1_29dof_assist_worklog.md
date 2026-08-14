# G1 29DoF Assist 模型修改与使用记录

本文记录本次对 G1 29DoF 机器人模型提出的需求、执行的操作、最终文件结构和常用运行命令。后续修改模型时，可以用本文核对各版本的区别。

## 1. 相关目录

主要工作目录：

```text
/home/libai/08_amp/legged_lab/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist
```

原始 Unitree ROS 模型目录：

```text
/home/libai/08_amp/unitree_ros/robots/g1_description
```

MuJoCo 启动脚本：

```text
/home/libai/08_amp/legged_lab/scripts/mujoco/play_g1_29dof_assist.py
```

使用的 Conda 环境：

```text
env_isaaclab_2
```

## 2. 需求和操作过程

### 2.1 查找训练模型的播放命令

最初询问了 `LeggedLab-Isaac-AMP-G1-v0` 的播放方法。项目注册了专用 Play 任务：

```bash
python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-Play-v0 \
  --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_amp/2026-08-11_17-41-02/model_17400.pt
```

不指定 `--checkpoint` 时，播放脚本会在对应日志目录中寻找 checkpoint。

### 2.2 精简 URDF 的 collision

操作对象最初位于：

```text
/home/libai/08_amp/unitree_ros/robots/g1_description/g1_29dof_2_assist.urdf
```

需求是删除大部分 link 的 collision，只保留下列碰撞体：

- `torso_link`
- 左右 `hip_pitch_link`
- 左右 `hip_pitch_assist_link`
- 脚部相关碰撞体

该模型没有独立的 `foot_link`。脚底接触点实际定义在左右 `ankle_roll_link` 中，因此保留了这两个 link 中的 4 个球形碰撞点。

处理后保留的有效 collision 如下：

| Link | 每个 link 的 collision 数量 |
|---|---:|
| `left_hip_pitch_link` | 1 |
| `right_hip_pitch_link` | 1 |
| `left_hip_pitch_assist_link` | 2 |
| `right_hip_pitch_assist_link` | 2 |
| `left_ankle_roll_link` | 4 |
| `right_ankle_roll_link` | 4 |
| `torso_link` | 1 |
| 合计 | 15 |

collision 只描述接触几何，不决定机器人质量。

### 2.3 统计原始质量

修改质量前，URDF 中具有 mass 定义的 link 总质量为：

```text
35.31514202 kg
```

其中两个 assist link 各为 `0.1 kg`，合计 `0.2 kg`。

以下空 link 没有 inertial/mass，因此不计入总质量：

- `imu_in_torso`
- `imu_in_pelvis`
- `d435_link`
- `mid360_link`

### 2.4 将总质量提高到 80 kg

需求是按比例提高机器人质量，并同步调整惯量，但排除以下 link：

- 名称包含 `imu` 的 link
- `d435_link`
- 名称包含 `mid360` 的 link
- 名称包含 `assist` 的 link

assist link 的总质量保持 `0.2 kg`，所以其他参与缩放的 link 目标质量为：

```text
80.0 - 0.2 = 79.8 kg
```

实际采用的统一缩放系数为：

```text
2.272523914456889
```

对参与缩放的每个 link，同步进行了以下调整：

```text
new_mass = old_mass × scale
new_inertia = old_inertia × scale
```

惯量张量的六个分量都使用相同比例缩放：

- `ixx`
- `ixy`
- `ixz`
- `iyy`
- `iyz`
- `izz`

缩放后带外骨骼 URDF 总质量为：

```text
80.00000000001538 kg
```

末尾差异是十进制写入和浮点求和误差，可视为准确的 `80 kg`。

### 2.5 将资产复制到 legged_lab

开始时误将文件从 Unitree ROS 目录移动到了 `legged_lab`。随后按要求修正为复制：原目录中的文件和资源已恢复，`legged_lab` 中保留独立副本。

在 `legged_lab` 中建立了：

```text
source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/
```

复制的主要内容包括：

- 4 个模型文件
- 这些模型实际引用的 36 个 STL mesh
- 完整 USD 目录

两边的 USD `config.yaml` 分别指向各自目录中的 URDF，避免交叉引用。

### 2.6 生成 MuJoCo XML

参考原有 `g1_29dof.xml`，生成了带外骨骼的 MuJoCo XML，并同步了：

- 调整后的质量和惯量
- 两个 assist body
- 两个 assist joint
- 两个 assist motor actuator
- assist 可视几何和碰撞盒
- URDF 中保留的碰撞体
- IMU 传感器
- 地面、灯光和可视化场景

由于参考 MuJoCo XML 将一部分固定装饰 link 合并为 geom，XML 中逐个 inertial 的质量求和可能与 URDF 相差约几克。这不代表主要活动刚体的缩放错误。

### 2.7 创建并运行 MuJoCo 启动脚本

创建了：

```text
scripts/mujoco/play_g1_29dof_assist.py
```

脚本会：

- 自动定位项目根目录
- 默认加载 `g1_29dof_assist.xml`
- 检查 XML 是否存在
- 创建 MuJoCo model/data
- 启动被动 viewer
- 按 MuJoCo timestep 推进仿真
- 支持通过 `--realtime-factor` 调整速度

在 `env_isaaclab_2` 中安装了：

```text
mujoco 3.11.0
```

模型成功编译并启动 Viewer。

### 2.8 拆分有无外骨骼版本

原来的 `g1_29dof_assist.urdf/xml` 是带外骨骼版本，因此改名为：

```text
g1_29dof_assist_exoskeleton.urdf
g1_29dof_assist_exoskeleton.xml
```

然后从带外骨骼版本生成新的无外骨骼版本：

```text
g1_29dof_assist.urdf
g1_29dof_assist.xml
```

无外骨骼版本删除了：

- `left_hip_pitch_assist_link`
- `right_hip_pitch_assist_link`
- `left_hip_pitch_assist_joint`
- `right_hip_pitch_assist_joint`
- MuJoCo 中相应的 body、geom 和 motor

两种版本的最终区别：

| 项目 | 无外骨骼 `assist` | 带外骨骼 `assist_exoskeleton` |
|---|---:|---:|
| URDF link | 40 | 42 |
| URDF joint | 39 | 41 |
| URDF 总质量 | 79.8 kg | 80.0 kg |
| MuJoCo `nq` | 36 | 38 |
| MuJoCo `nv` | 35 | 37 |
| MuJoCo actuator `nu` | 29 | 31 |
| MuJoCo body 数量 | 31 | 33 |

这里的 `nq` 包含浮动基座的 7 个广义坐标，`nv` 包含浮动基座的 6 个速度自由度。

### 2.9 从无外骨骼 URDF 生成 USD

使用以下输入生成了新的 USD：

```text
g1_29dof_assist.urdf
```

转换使用 `env_isaaclab_2` 和 Isaac Lab 2.3.1 的 `convert_urdf.py`，主要配置为：

- 浮动基座，即 `fix_base: false`
- 不合并 fixed joints
- joint drive stiffness 为 0
- joint drive damping 为 0
- collision 类型为 `convex_hull`
- 不启用 self-collision

生成的主 USD 是：

```text
usd/g1_29dof_assist/g1_29dof_assist.usd
```

同时生成了：

```text
usd/g1_29dof_assist/configuration/g1_29dof_assist_base.usd
usd/g1_29dof_assist/configuration/g1_29dof_assist_physics.usd
usd/g1_29dof_assist/configuration/g1_29dof_assist_robot.usd
usd/g1_29dof_assist/configuration/g1_29dof_assist_sensor.usd
usd/g1_29dof_assist/config.yaml
```

转换器针对 `imu`、`d435_link` 和 `mid360_link` 发出了没有 mass、visual 或 collider 的警告。这些 link 在 URDF 中本来就是空 link，USD 仍成功生成。

### 2.10 基于新 USD 建立独立 AMP 任务

随后要求模仿以下原始任务：

```text
LeggedLab-Isaac-AMP-G1-v0
LeggedLab-Isaac-AMP-G1-Play-v0
```

建立两个使用新 USD 的任务：

```text
LeggedLab-Isaac-AMP-G1-assist-v0
LeggedLab-Isaac-AMP-G1-assist-Play-v0
```

特别要求重新建立新文件，不影响原始 G1 v0。因此没有修改：

- `config/g1/` 中的原始 G1 环境配置
- `legged_lab/assets/unitree.py` 中的 `UNITREE_G1_29DOF_CFG`
- `LeggedLab-Isaac-AMP-G1-v0` 和 `LeggedLab-Isaac-AMP-G1-Play-v0` 的注册
- 原始 G1 的日志目录 `logs/rsl_rl/g1_amp/`

新建了完全独立的任务配置目录：

```text
source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist/
├── __init__.py
├── g1_assist_amp_env_cfg.py
├── robot_cfg.py
└── agents/
    ├── __init__.py
    └── rsl_rl_ppo_cfg.py
```

各文件职责如下：

| 文件 | 作用 |
|---|---|
| `g1_assist/__init__.py` | 注册训练和播放两个 Gym 任务 |
| `g1_assist_amp_env_cfg.py` | 建立训练和 Play 环境配置，只替换机器人资产 |
| `robot_cfg.py` | 定义新 USD 路径和 assist 专用机器人配置 |
| `agents/rsl_rl_ppo_cfg.py` | 继承 G1 AMP runner 参数，并使用独立实验名 |

新任务使用的 USD：

```text
source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/
usd/g1_29dof_assist/g1_29dof_assist.usd
```

新 runner 的实验名设置为：

```text
g1_assist_amp
```

因此日志和 checkpoint 写入：

```text
logs/rsl_rl/g1_assist_amp/
```

而不会混入原始 G1 的 `logs/rsl_rl/g1_amp/`。

配置加载检查结果：

| 任务 | USD | 默认环境数 | 参考动作重置 | 实验名 |
|---|---|---:|---|---|
| `G1-v0` | 原 G1 USD | 8192 | 开启 | `g1_amp` |
| `G1-Play-v0` | 原 G1 USD | 48 | 关闭 | `g1_amp` |
| `G1-assist-v0` | 新 assist USD | 8192 | 开启 | `g1_assist_amp` |
| `G1-assist-Play-v0` | 新 assist USD | 48 | 关闭 | `g1_assist_amp` |

使用 1 个环境运行了 1 次实际 AMP 训练迭代，验证了：

- USD articulation 可以正常创建
- 29 个动作关节全部加载
- policy observation 为 96 维
- critic observation 为 297 维
- discriminator 和 demonstration observation 均为 `4 × 70`
- Actor 输出维度为 29
- AMP discriminator 可以正常前向和反向传播
- checkpoint 可以正常保存

第一次测试产生了独立测试日志和 `model_0.pt`，位于 `logs/rsl_rl/g1_assist_amp/` 下。

### 2.11 判断增重后是否需要调整 PD

之后询问了 assist 相比原始 G1 增重后是否需要调整 PD。

结论是需要为 assist 建立独立 PD 配置。原因是质量和惯量约增至原来的 `2.27` 倍，而原始 PD 不变时，同样关节误差产生的控制力矩不变，关节加速度和响应速度会明显下降。

可能出现的问题包括：

- 关节动作响应变慢
- 动作跟踪误差增大
- 膝、髋和踝无法及时支撑身体
- 策略输出经常接近最大动作
- effort limit 饱和
- AMP 训练收敛速度下降

同时指出不能只增大 Kp 而不检查 effort limit。控制器即使计算出更大力矩，也会被原始力矩限制截断。

### 2.12 为 assist 建立独立 PD 和 effort limit

最终要求调整 assist 任务的 PD 和 effort limit。修改仅发生在：

```text
source/legged_lab/legged_lab/tasks/locomotion/amp/config/
g1_assist/robot_cfg.py
```

原始 G1 的 `UNITREE_G1_29DOF_CFG` 仍然没有修改。

assist 配置先通过 `deepcopy` 复制原始 G1 配置，再替换 USD 和 actuator 字典，确保新旧任务之间没有共享的可变配置。

原始 G1 将 ankle、waist 和部分 arm joint 放在同一个执行器组，并共用 `25 Nm` 上限。为了避免手臂使用腿部所需的大力矩，新 assist 配置将 29 个关节拆成 5 个互不重叠的 actuator 组。

最终参数如下：

| Actuator 组 | 关节 | Kp | Kd | Effort limit | Velocity limit |
|---|---|---:|---:|---:|---:|
| `assist_hip_pitch_yaw_waist_yaw` | hip pitch/yaw | 180 | 3.0 | 200 Nm | 32 rad/s |
| 同上 | waist yaw | 360 | 7.5 | 200 Nm | 32 rad/s |
| `assist_hip_roll_knee` | hip roll | 180 | 3.0 | 316 Nm | 20 rad/s |
| 同上 | knee | 270 | 6.0 | 316 Nm | 20 rad/s |
| `assist_ankle_waist` | ankle | 72 | 3.0 | 57 Nm | 37 rad/s |
| 同上 | waist roll/pitch | 72 | 7.5 | 57 Nm | 37 rad/s |
| `assist_shoulder_elbow_wrist_roll` | shoulder/elbow/wrist roll | 56 | 1.3 | 38 Nm | 37 rad/s |
| `assist_wrist_pitch_yaw` | wrist pitch/yaw | 56 | 1.3 | 8 Nm | 22 rad/s |

调整原则：

- 承重关节 Kp 约为原值的 `1.8` 倍
- 承重关节 Kd 约为原值的 `1.5` 倍
- 承重关节 effort limit 约按质量比例 `2.27` 倍放大
- 上肢 Kp 约为原值的 `1.4` 倍
- 上肢 Kd 约为原值的 `1.3` 倍
- 上肢 effort limit 采用约 `1.5` 倍的温和增幅
- 所有 actuator 的 armature 保持 `0.01`
- 原始 velocity limit 保持不变

修改后再次使用 1 个环境运行了 1 次 AMP 训练迭代。验证结果：

- 5 个 actuator 组成功初始化
- 29 个动作关节完整覆盖
- 没有 actuator 正则表达式重叠或漏配错误
- 调整后的 PD 和 effort limit 已写入实际运行保存的 `env.yaml`
- AMP rollout、反向传播和 checkpoint 保存均成功
- 原始 G1 配置仍没有代码差异

测试运行中仍会看到 `imu_in_pelvis`、`imu_in_torso`、`d435_link` 和 `mid360_link` 的空 visual reference 警告。这来自生成 USD 时保留的空 link，不是 PD 或任务配置错误，也没有阻止训练运行。

### 2.13 使用训练策略完成 Isaac Sim → MuJoCo Sim2Sim

训练结果使用以下命令在 Isaac Sim 中检查，机器人已经可以行走：

```bash
python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-assist-Play-v0 \
  --num_envs 16 \
  --checkpoint /home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_amp/2026-08-13_13-45-16/model_29400.pt
```

播放脚本导出的 TorchScript 策略为：

```text
/home/libai/08_amp/legged_lab/logs/rsl_rl/g1_assist_amp/
2026-08-13_13-45-16/exported/policy.pt
```

在 `scripts/mujoco/` 中新增了 Sim2Sim 脚本：

```text
scripts/mujoco/sim2sim_g1_29dof_assist.py
```

脚本完成以下映射：

- 加载导出的 TorchScript policy
- 加载 `g1_29dof_assist.xml`
- 构造与 Isaac 训练一致的 96 维 policy observation
- 输出 29 维 joint-position action
- 使用 `action_scale=0.25` 和 Isaac 默认关节姿态生成目标位置
- 使用 assist 专用 Kp、Kd 和 effort limit 计算 MuJoCo torque
- 根据关节名映射 Isaac policy 顺序和 MuJoCo joint/actuator 顺序
- 支持速度命令、键盘控制、自动跌倒复位和无窗口测试

策略接口验证结果：

```text
policy input:  (1, 96)
policy output: (1, 29)
actor observation normalization: Identity
```

实现时发现两个关键 Sim2Sim 差异。

第一，Isaac/PhysX articulation 的实际关节顺序不是 URDF/MuJoCo 的树顺序。训练策略使用的顺序为：

```text
left_hip_pitch, right_hip_pitch, waist_yaw,
left_hip_roll, right_hip_roll, waist_roll,
left_hip_yaw, right_hip_yaw, waist_pitch,
left_knee, right_knee,
left_shoulder_pitch, right_shoulder_pitch,
left_ankle_pitch, right_ankle_pitch,
left_shoulder_roll, right_shoulder_roll,
left_ankle_roll, right_ankle_roll,
left_shoulder_yaw, right_shoulder_yaw,
left_elbow, right_elbow,
left_wrist_roll, right_wrist_roll,
left_wrist_pitch, right_wrist_pitch,
left_wrist_yaw, right_wrist_yaw
```

最初按 MuJoCo 树顺序直接传递 observation/action 时，机器人会立即跌倒。修正后，脚本以 Isaac 顺序组织 policy 输入输出，再通过关节名映射到 MuJoCo。

第二，Isaac actuator 配置对所有活动关节设置了：

```text
armature = 0.01
```

MuJoCo XML 原先没有该 armature。缺少它时，即使只使用 PD 保持默认姿态，模型也会数值不稳定或快速跌倒。Sim2Sim 脚本在加载模型后将 29 个关节的 `dof_armature` 设置为 `0.01`。

此外，MuJoCo 使用：

```text
physics dt = 0.002 s
decimation = 10
policy period = 0.020 s
policy frequency = 50 Hz
```

策略周期仍与 Isaac 训练中的 `0.005 × 4 = 0.020 s` 一致，但更小的 MuJoCo 积分步长提高了高增益模型的稳定性。

最终进行了 30 秒无窗口测试：

```text
command vx:       0.5 m/s
simulated time:   30.0 s
final x position: 15.018 m
final height:     0.737 m
automatic resets: 0
```

测试过程中机器人持续前进，没有跌倒、没有自动复位，也没有出现 MuJoCo 数值发散。随后成功启动了交互式 MuJoCo Viewer。

### 2.14 使用原 29-DoF 策略播放 31-DoF 外骨骼 USD

需求是使用带两个辅助关节的
`g1_29dof_assist_exoskeleton.usd` 替换无外骨骼 USD 进行 Isaac Sim play，继续加载
`model_29400.pt`，并令两个辅助关节不输出扭矩。

为避免影响已有的 `play.py` 和 assist 任务配置，新建了独立脚本：

```text
scripts/rsl_rl/play_g1_assist_exoskeleton.py
```

脚本进行了以下隔离：

1. 将机器人资产替换为 `g1_29dof_assist_exoskeleton.usd`，并保持 self-collision 开启。
2. 原 hip actuator 改为只匹配 5 个实际受控关节，避免其通配表达式误匹配新增辅助关节。
3. 为 `left_hip_pitch_assist_joint` 和 `right_hip_pitch_assist_joint` 设置独立被动 actuator：`stiffness=0`、`damping=0`、`effort_limit_sim=0`。
4. policy action 明确限定为训练时的 29 个关节及原始顺序。
5. policy、critic 和 AMP discriminator 的 joint position/velocity 观测均明确限定为原 29 个关节，避免 checkpoint 输入维度变化。
6. 播放过程中读取 `applied_torque`，如果任一辅助关节输出超过 `1e-6 N·m`，脚本会直接报错。

在 `env_isaaclab_2` 中用 1 个环境进行了 20 步无窗口验证。结果为：

```text
action shape: 29
policy observation shape: 96
critic observation shape: 297
discriminator observation shape: 4 x 70
model output shape: 29
assist actuator torque: 0.000e+00 N.m（连续 20 步）
```

checkpoint 成功加载，测试进程正常退出。USD 加载时仍会报告 imu、D435 和 Mid360 的 visual
reference warning，但不影响 articulation、策略维度和本次零扭矩验证。

### 2.15 建立外骨骼辅助扭矩 PPO 训练任务

新建了两个相互对应的独立任务：

```text
LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0
LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0
```

所有任务代码位于新目录：

```text
source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist_exoskeleton/
```

控制采用分层结构：冻结的 `model_29400.pt` 导出策略继续根据原 96 维输入生成 29
个机器人本体关节位置动作；新 PPO 只生成左右两个 `hip_pitch_assist_joint` 的扭矩。
冻结策略文件为：

```text
logs/rsl_rl/g1_assist_amp/2026-08-13_13-45-16/exported/policy.pt
```

辅助 PPO 的物理扭矩范围按照 URDF effort limit 设为 `[-8, 8] N·m`。Actor 和 Critic
均使用三层 MLP：

```text
150 -> 256 -> 64 -> 16 -> 2    Actor
150 -> 256 -> 64 -> 16 -> 1    Critic
```

控制周期为 `0.001 × 20 = 0.02 s`，即 50 Hz。0.5 秒对应 25 帧，每帧包含左右
辅助关节的 angle、velocity 和 commanded torque 共 6 个值，因此输入维度为
`25 × 6 = 150`。

外骨骼辅助策略曾短暂使用固定的 `(vx, vy, yaw)=(0.7, 0.0, 0.0)` 指令，后续已按要求
取消。训练任务重新继承原 G1-assist 的随机 command 范围；Play 任务恢复为
`vx=(0.5, 3.0)`、`vy=(-0.5, 0.5)`、`yaw=(-1.0, 1.0)`。

大规模训练首次使用 6000 个环境时，ActionTerm reset 中原先的
`default_joint_pos[env_ids, joint_ids]` 会触发 PyTorch 成对高级索引，导致 6000 个环境索引
与 29 个关节索引无法广播。现已改为先选择环境、再选择关节的二维索引，支持任意环境数量。

奖励只有三项：辅助动作变化惩罚、辅助扭矩同向/幅值奖励、非超时终止惩罚。同向奖励
鼓励辅助扭矩接近对应 hip-pitch 电机扭矩；反向扭矩没有有效辅助奖励，超过对应 hip-pitch
扭矩的部分会受到二次惩罚。

终止条件包含原 G1 assist 的高度过低、姿态超过 60 度和 20 秒超时，并增加左右任一
assist joint 与对应 hip-pitch joint 相差超过 4 度时终止。重置时会首先同步两组关节角度，
避免随机初始化直接触发 4 度终止条件。

已在 `env_isaaclab_2` 中完成 1 环境、1 iteration 真实训练测试，并使用生成的
`model_0.pt` 完成 Play 任务加载测试。验证得到：

```text
physics dt: 0.001 s
control dt: 0.02 s
action shape: 2
policy observation shape: 150
critic observation shape: 150
runner: OnPolicyRunner + PPO
```

### 2.16 单机器人 Play、CSV 记录与关节曲线

参考 `/home/libai/06_assist_test/isaaclab/assist/scripts/rsl_rl/play_assist.py`，新建了当前
外骨骼任务专用的单机器人播放脚本：

```text
scripts/rsl_rl/play_g1_assist_exoskeleton_plot.py
```

脚本强制使用 1 个环境，加载
`LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0` 和指定的辅助 PPO checkpoint。
它直接从 articulation 读取左右两组关节的角度、角速度和实际 actuator torque，不依赖
其他项目私有的 `_critic_obs_history`。

实时图按左右腿分为两列，包含四行：hip/assist 角度、assist 与 hip 角度差、hip/assist
角速度、hip/assist 扭矩。角度差图显示正负 4 度终止边界。退出时会保存完整 CSV 和最终
PNG 到 checkpoint 目录下的 `play_analysis/`。

已用 `model_1000.pt` 完成 0.4 秒无窗口测试，CSV 和 PNG 均成功生成。

### 2.17 外骨骼辅助策略 MuJoCo Sim2Sim

为单机器人外骨骼 Play 新建了双策略 MuJoCo Sim2Sim 脚本：

```text
scripts/mujoco/sim2sim_g1_assist_exoskeleton.py
```

脚本加载 `g1_29dof_assist_exoskeleton.xml`，以冻结的原 29 关节策略生成本体 PD 目标，
同时使用 `model_1000.pt` 对应的导出策略生成左右 assist joint 的直接扭矩。辅助观测严格
保持 25 帧、150 维的 oldest-to-newest 排列，扭矩输出范围为正负 8 N·m。MuJoCo 使用
`dt=0.001 s`、`decimation=20`，与 Isaac 训练任务同为 50 Hz 策略频率。

脚本支持 MuJoCo viewer、实时曲线、键盘速度指令、CSV 和最终 PNG。5 秒无窗口测试中，
机器人前进约 2.79 m，未发生高度重置。MuJoCo 中记录到的最大角差约为左 6.10 度、
右 5.81 度，说明碰撞动力学仍存在 Sim2Sim 差异。因此默认只绘制正负 4 度边界而不据此
重置；需要严格复现 Isaac 终止时可传入 `--auto-reset-angle-deg 4`。

### 2.18 同条件对比有无外骨骼的 hip-pitch 机械功率

新建双模型并行对比程序：

```text
scripts/mujoco/compare_g1_hip_torque_exoskeleton.py
```

基线组使用无外骨骼 `g1_29dof_assist.xml` 和冻结行走策略；外骨骼组使用
`g1_29dof_assist_exoskeleton.xml`、同一个冻结行走策略和辅助 PPO。两组具有相同的初始
姿态、速度 command、`dt=0.001 s`、`decimation=20` 和运行时长。程序以 1 kHz 记录左右
机器人本体左右 hip-pitch actuator 的力矩及关节角速度，并按
`P = torque * joint_velocity` 计算带符号的机械功率，生成完整时序 CSV、汇总 CSV 和
对比图。外骨骼 assist actuator 只参与控制，不计算、保存或绘制其功率。

汇总统计默认忽略最初 1 秒启动过程，核心指标为净平均功率、平均正功率、平均负功率、
正/负/净机械能和正/负峰值功率，不再使用平均绝对力矩作为结论。5 秒测试中，
两组最终前进距离分别约 2.766 m 和 2.794 m；左髋平均正功率由 16.435 W 降至
10.413 W（下降 36.64%），右髋由 12.569 W 增至 13.798 W（增加 9.78%）。该结果仅代表
当前 `model_1000.pt`、0.7 m/s 指令和 1--5 秒统计窗口。双侧本体髋电机平均正功率合计由
29.004 W 降至 24.211 W（下降 16.53%）。

## 3. 当前推荐使用的文件

### 无外骨骼模型

```text
g1_29dof_assist.urdf
g1_29dof_assist.xml
usd/g1_29dof_assist/g1_29dof_assist.usd
```

### 带外骨骼模型

```text
g1_29dof_assist_exoskeleton.urdf
g1_29dof_assist_exoskeleton.xml
```

目前只为无外骨骼版本重新生成了以 `g1_29dof_assist` 命名的新 USD。目录中原先的 `usd/g1_29dof.usd` 是早期资产，不应与新生成的 USD 混淆。

### Assist AMP 任务配置

```text
source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_assist/
```

训练任务：

```text
LeggedLab-Isaac-AMP-G1-assist-v0
```

播放任务：

```text
LeggedLab-Isaac-AMP-G1-assist-Play-v0
```

## 4. 常用命令

### 4.1 启动无外骨骼 MuJoCo 模型

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/play_g1_29dof_assist.py
```

### 4.2 启动带外骨骼 MuJoCo 模型

启动脚本支持 `--model` 参数：

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/play_g1_29dof_assist.py \
  --model source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/g1_29dof_assist_exoskeleton.xml
```

### 4.3 调整 MuJoCo 仿真速度

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/play_g1_29dof_assist.py \
  --realtime-factor 0.5
```

`0.5` 表示以约一半实时速度运行，`2.0` 表示目标速度约为实时的两倍。

### 4.4 重新生成无外骨骼 USD

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python /home/libai/00_isaaclab/IsaacLab_2.3.1/scripts/tools/convert_urdf.py \
  source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/g1_29dof_assist.urdf \
  source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof_assist/usd/g1_29dof_assist/g1_29dof_assist.usd \
  --joint-stiffness 0 \
  --joint-damping 0 \
  --headless
```

### 4.5 训练 G1 assist AMP 任务

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-v0 \
  --headless
```

调试时可减少环境数和训练次数：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-v0 \
  --num_envs 1 \
  --max_iterations 1 \
  --headless
```

### 4.6 播放 G1 assist AMP checkpoint

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-assist-Play-v0
```

显式指定 checkpoint：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-assist-Play-v0 \
  --checkpoint /absolute/path/to/model.pt
```

### 4.7 运行 G1 assist Sim2Sim

默认加载本次 `model_29400.pt` 导出的 policy：

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_29dof_assist.py
```

Viewer 键盘控制：

| 按键 | 功能 |
|---|---|
| `W` / `S` | 前进速度增加/减少 0.1 m/s |
| `A` / `D` | 横向速度增加/减少 0.1 m/s |
| `Q` / `E` | 偏航角速度增加/减少 0.1 rad/s |
| `Space` | 所有速度命令归零 |
| `R` | 重置机器人 |

指定初始速度命令：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_29dof_assist.py \
  --vx 1.0 --vy 0.0 --yaw 0.0
```

无窗口快速验证：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_29dof_assist.py \
  --headless --duration 30 --no-realtime --auto-reset-height 0
```

使用其他导出策略：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_29dof_assist.py \
  --policy /absolute/path/to/policy.pt
```

### 4.8 使用外骨骼 USD 播放 model_29400.pt

脚本已经内置任务名、外骨骼 USD 路径和本次 checkpoint 路径，因此直接运行即可：

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/play_g1_assist_exoskeleton.py
```

默认启动 16 个环境。无窗口有限步数验证命令：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/play_g1_assist_exoskeleton.py \
  --headless --num_envs 1 --max_steps 200
```

脚本默认每 200 步报告一次两个辅助关节的最大绝对 actuator torque。可使用
`--torque_report_interval 1` 每步检查，或使用 `--torque_report_interval 0` 关闭打印；
无论是否打印，两个辅助关节仍保持零刚度、零阻尼和零 effort limit。

### 4.9 训练和播放外骨骼辅助扭矩 PPO

训练：

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0 \
  --headless
```

小规模调试：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-v0 \
  --num_envs 1 --max_iterations 1 --headless
```

播放训练后的 checkpoint：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-G1-assist-exoskeleton-Play-v0 \
  --num_envs 16 \
  --checkpoint /absolute/path/to/model_xxx.pt
```

### 4.10 单机器人播放并绘制外骨骼关节曲线

实时播放和绘图：

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/play_g1_assist_exoskeleton_plot.py \
  --checkpoint /absolute/path/to/model_xxx.pt \
  --real-time
```

无窗口运行 20 秒并只保存 CSV/PNG：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/rsl_rl/play_g1_assist_exoskeleton_plot.py \
  --checkpoint /absolute/path/to/model_xxx.pt \
  --duration 20 --headless
```

如果省略 `--checkpoint`，脚本会从 `g1_assist_exoskeleton_ppo` 实验目录自动选择最新
checkpoint。可通过 `--output_dir` 指定其他输出目录。

### 4.11 外骨骼辅助策略 MuJoCo Sim2Sim

启动 MuJoCo viewer、两个策略和实时曲线：

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_assist_exoskeleton.py
```

无窗口快速运行并保存 CSV/PNG：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_assist_exoskeleton.py \
  --headless --duration 20 --no-realtime
```

指定其他辅助导出策略：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_assist_exoskeleton.py \
  --assist-policy /absolute/path/to/exported/policy.pt
```

启用与 Isaac 相同的 4 度角差自动重置：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/sim2sim_g1_assist_exoskeleton.py \
  --auto-reset-angle-deg 4
```

### 4.12 同条件比较有无外骨骼的 hip-pitch 机械功率

默认比较 10 秒，并忽略前 1 秒启动数据：

```bash
cd /home/libai/08_amp/legged_lab

conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/compare_g1_hip_torque_exoskeleton.py
```

指定工况和统计窗口：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/compare_g1_hip_torque_exoskeleton.py \
  --vx 0.7 --vy 0.0 --yaw 0.0 \
  --duration 20 --warmup 2
```

指定其他辅助策略：

```bash
conda run --no-capture-output -n env_isaaclab_2 \
  python scripts/mujoco/compare_g1_hip_torque_exoskeleton.py \
  --assist-policy /absolute/path/to/exported/policy.pt
```

输出包括 `hip_pitch_power_timeseries.csv`、`hip_pitch_power_summary.csv` 和
`hip_pitch_power_comparison.png`。CSV 同时保留力矩和角速度原始值，便于复核
`P = torque * joint_velocity`。

## 5. 后续修改时的注意事项

1. 修改 `mass` 时，应根据物体的变化同步修改 `inertia`；collision 不决定质量。
2. 当前质量缩放假设几何尺寸不变，只是材料密度或附加载荷按比例增加，所以 mass 和 inertia 使用同一倍数。
3. 修改 URDF 后，XML 和 USD 不会自动同步，需要重新生成或手动同步。
4. 无外骨骼版本的总质量为 `79.8 kg`，因为两个各 `0.1 kg` 的外骨骼 link 已被删除。
5. 如果希望无外骨骼版本仍保持 `80 kg`，还需要将缺少的 `0.2 kg` 重新按比例分配到其余 link，并同步惯量。
6. `ankle_roll_link` 中的球形 collision 是脚底接触点，不建议在没有替代接触几何的情况下删除。
7. `usd/g1_29dof_assist/` 是当前无外骨骼模型的新 USD；上一级旧的 `usd/g1_29dof.usd` 来自更早的模型转换。
8. Assist 任务使用独立目录 `config/g1_assist/`。后续调参应修改这里，不应修改 `config/g1/` 或全局 `UNITREE_G1_29DOF_CFG`。
9. PD 增大后应监控关节力矩饱和、动作抖动、接触冲击和训练稳定性。当前参数是基于质量比例的合理起点，不是经过长期训练后的最终最优参数。
10. 如果要对应真实机器人，`effort_limit_sim` 必须重新约束到真实电机和减速器允许的持续/峰值力矩，不能直接使用当前仿真放大值。
11. 如果后续重新生成 USD，应重新运行 1 环境初始化测试，确认 articulation、29 个动作关节和 contact sensor 仍能正常加载。
12. Sim2Sim 的 policy joint 顺序必须使用 Isaac articulation 的实际顺序，不能假设与 URDF/MuJoCo 顺序一致。
13. MuJoCo 中必须保留等效 `armature=0.01`；删除该设置会明显降低高 PD 模型的数值稳定性。
14. MuJoCo 的物理步长采用 `0.002 s`，但策略仍以 50 Hz 运行。不要仅修改 dt 而忘记同步调整 decimation。

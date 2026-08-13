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

## 5. 后续修改时的注意事项

1. 修改 `mass` 时，应根据物体的变化同步修改 `inertia`；collision 不决定质量。
2. 当前质量缩放假设几何尺寸不变，只是材料密度或附加载荷按比例增加，所以 mass 和 inertia 使用同一倍数。
3. 修改 URDF 后，XML 和 USD 不会自动同步，需要重新生成或手动同步。
4. 无外骨骼版本的总质量为 `79.8 kg`，因为两个各 `0.1 kg` 的外骨骼 link 已被删除。
5. 如果希望无外骨骼版本仍保持 `80 kg`，还需要将缺少的 `0.2 kg` 重新按比例分配到其余 link，并同步惯量。
6. `ankle_roll_link` 中的球形 collision 是脚底接触点，不建议在没有替代接触几何的情况下删除。
7. `usd/g1_29dof_assist/` 是当前无外骨骼模型的新 USD；上一级旧的 `usd/g1_29dof.usd` 来自更早的模型转换。

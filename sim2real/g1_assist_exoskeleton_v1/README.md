# G1 v1 外骨骼策略的最小 NumPy 部署

本目录把 v1 外骨骼策略拆成显式的输入归一化、全连接层和 ELU 运算。运行时的
`numpy_assist_policy.py` 不依赖 PyTorch，便于后续逐句改写成单片机 C/C++。

## 网络结构

输入顺序必须保持为 100 个 float32：先放 25 帧左右外骨骼关节速度，再放 25 帧
左右助力指令力矩。每组历史均为从最旧到最新，每帧顺序为左、右。

```text
input[100]
  -> (input - mean) / (std + 0.01)
  -> Linear(100, 256) -> ELU
  -> Linear(256, 64)  -> ELU
  -> Linear(64, 16)   -> ELU
  -> Linear(16, 2)
  -> clip(-1, 1) * 8 N.m
```

线性层采用 `output = input @ weight.T + bias`。权重是 float32，矩阵形状遵循
PyTorch 的 `[输出维数, 输入维数]`。

## 使用

需要 PyTorch 的环境只用于一次性导出和对比：

```bash
/home/libai/anaconda3/envs/env_isaaclab_0/bin/python \
  sim2real/g1_assist_exoskeleton_v1/export_assist_policy_numpy.py

/home/libai/anaconda3/envs/env_isaaclab_0/bin/python \
  sim2real/g1_assist_exoskeleton_v1/test_numpy_assist_policy.py
```

运行 NumPy 外骨骼策略的 MuJoCo 测试：

```bash
/home/libai/anaconda3/envs/env_isaaclab_0/bin/python \
  sim2real/g1_assist_exoskeleton_v1/sim2sim_numpy_assist.py --headless --duration 10 --no-realtime
```

去掉 `--headless` 可打开 Viewer。这个 sim2sim 中 PyTorch 只负责原有的 29 关节
步态策略；100 维外骨骼策略完全由 `NumpyAssistPolicy` 计算。

仿真会在 `sim2real/g1_assist_exoskeleton_v1/output` 保存完整 CSV 和 PNG。PNG 包含左右髋关节/外骨骼的
角度、角度差、速度和执行器力矩。使用 Viewer 时默认同时实时刷新曲线；传入
`--no-live-plot` 可以只在退出时保存最终图片，`--output-dir` 可以修改输出目录。

该 sim2sim 在导入 NumPy 前将尚未配置的 `OPENBLAS_NUM_THREADS` 设为 1，避免很小的
单样本矩阵乘法创建大型线程池。在当前主机上，外骨骼网络单次前向约为 0.1 ms。

## 单片机 C/C++ 使用

`c/g1_assist_policy.c` 已经包含固定网络的全部参数，单片机运行时不需要 `.pt`、
`.npz`、Python、PyTorch 或动态内存。把 `g1_assist_policy.c/.h` 加入 MCU 工程，并
链接提供 `expf()` 的单精度数学库。GCC 类工具链通常需要链接选项 `-lm`：

```bash
gcc -std=c99 -O2 main.c sim2real/g1_assist_exoskeleton_v1/c/g1_assist_policy.c -lm
```

建议把历史和工作区声明成静态或全局变量，避免占用任务栈：

```c
#include "g1_assist_policy.h"

static G1AssistPolicyHistory history;
static G1AssistPolicyWorkspace workspace;
static float observation[G1_ASSIST_POLICY_INPUT_SIZE];
static float previous_torque_nm[2] = {0.0f, 0.0f};
static unsigned int first_policy_step = 1U;

void controller_reset(float left_velocity, float right_velocity)
{
    float initial_velocity[2] = {left_velocity, right_velocity};
    g1_assist_policy_history_reset(&history, initial_velocity);
    previous_torque_nm[0] = 0.0f;
    previous_torque_nm[1] = 0.0f;
    first_policy_step = 1U;
}

/* 每20 ms调用一次，即50 Hz。velocity单位必须是rad/s。 */
void controller_step(float left_velocity, float right_velocity)
{
    float velocity[2] = {left_velocity, right_velocity};
    float torque_nm[2];

    /* 严格复现sim2sim：复位后的第一次推理不追加历史。 */
    if (first_policy_step != 0U) {
        first_policy_step = 0U;
    } else {
        g1_assist_policy_history_append(&history, velocity, previous_torque_nm);
    }
    g1_assist_policy_history_build_observation(&history, observation);
    g1_assist_policy_compute_torque(observation, torque_nm, &workspace);

    previous_torque_nm[0] = torque_nm[0];
    previous_torque_nm[1] = torque_nm[1];
    motor_set_torque_nm(torque_nm[0], torque_nm[1]);
}
```

必须注意：

- 输入速度单位为 `rad/s`，不能使用 `deg/s`。
- 历史顺序为最旧到最新，左右顺序固定为 `[left, right]`。
- 力矩历史保存“上一周期的指令力矩”，不是传感器测得的实际力矩。
- 网络每20 ms运行一次。两个输出已经限幅并换算为左右 `[-8, 8] N.m`。
- `G1AssistPolicyWorkspace` 占1424字节；历史约404字节。
- 固定参数占174312字节 Flash，另有少量程序代码。请确认芯片 Flash 容量足够。
- 如果 RTOS 中有多个任务同时推理，每个并发调用必须使用独立工作区。

上面的示例在复位后的第一个周期是否调用 `history_append()`，应由控制状态机决定。
为了严格复现当前 sim2sim，复位后的第一次推理直接使用 reset 填好的历史；从第二次
推理开始，才追加当前速度和上一次指令力矩。

只有策略重新训练或权重变化时才需要重新生成 C 参数：

```bash
python sim2real/g1_assist_exoskeleton_v1/export_assist_policy_c.py
```

## 文件说明

- `weights/assist_policy_v1.npz`：权重、偏置和输入归一化数组。
- `weights/assist_policy_v1.json`：网络结构和输入顺序说明。
- `numpy_assist_policy.py`：纯 NumPy 前向计算。
- `test_numpy_assist_policy.py`：与 TorchScript 批量数值对比。
- `sim2sim_numpy_assist.py`：纯 NumPy 外骨骼策略的 MuJoCo 测试入口。
- `c/g1_assist_policy.h`：单片机 C99 接口、历史和工作区定义。
- `c/g1_assist_policy.c`：固化的网络参数和完整前向计算。
- `export_assist_policy_c.py`：更换网络后重新生成 C 文件的工具，正常运行时不调用。

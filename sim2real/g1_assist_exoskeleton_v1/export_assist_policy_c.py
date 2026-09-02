#!/usr/bin/env python3
"""Generate standalone C99 source/header files from the fixed NumPy policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights/assist_policy_v1.npz"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def c_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    lines = []
    for start in range(0, flat.size, 8):
        literals = ", ".join(f"{float(value):.9e}f" for value in flat[start : start + 8])
        lines.append(f"    {literals},")
    return f"static const float {name}[{flat.size}] = {{\n" + "\n".join(lines) + "\n};\n"


def main() -> None:
    args = parse_args()
    weights_path = args.weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"NumPy policy weights not found: {weights_path}")

    with np.load(weights_path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}
    expected_shapes = {
        "obs_mean": (100,),
        "obs_std": (100,),
        "weight_0": (256, 100),
        "bias_0": (256,),
        "weight_1": (64, 256),
        "bias_1": (64,),
        "weight_2": (16, 64),
        "bias_2": (16,),
        "weight_3": (2, 16),
        "bias_3": (2,),
    }
    for name, expected in expected_shapes.items():
        if name not in arrays or arrays[name].shape != expected:
            actual = None if name not in arrays else arrays[name].shape
            raise ValueError(f"{name} has shape {actual}, expected {expected}")

    eps = float(np.asarray(arrays["normalizer_eps"], dtype=np.float32))
    header = """/* Auto-generated fixed G1 v1 assist policy interface. */
#ifndef G1_ASSIST_POLICY_H
#define G1_ASSIST_POLICY_H

#ifdef __cplusplus
extern "C" {
#endif

#define G1_ASSIST_POLICY_INPUT_SIZE 100
#define G1_ASSIST_POLICY_OUTPUT_SIZE 2
#define G1_ASSIST_POLICY_HISTORY_LENGTH 25
#define G1_ASSIST_POLICY_TORQUE_LIMIT_NM 8.0f

/*
 * ============================ 使用说明 ============================
 *
 * 1. 将 g1_assist_policy.c 和 g1_assist_policy.h 加入单片机工程。
 *    本文件已固化全部网络参数，运行时不需要 policy.pt、NPZ、Python、
 *    PyTorch或动态内存。g1_assist_policy.c使用expf()，GCC工具链通常
 *    需要链接数学库，例如：
 *
 *        gcc -std=c99 -O2 main.c g1_assist_policy.c -lm
 *
 * 2. 策略必须以50 Hz运行，即每20 ms调用一次。速度单位必须是rad/s，
 *    力矩单位是N.m。左右顺序始终为[左, 右]。
 *
 * 3. 推荐把历史、工作区和观测数组声明为static或全局变量，避免占用
 *    单片机任务栈：
 *
 *        static G1AssistPolicyHistory g_history;
 *        static G1AssistPolicyWorkspace g_workspace;
 *        static float g_observation[G1_ASSIST_POLICY_INPUT_SIZE];
 *        static float g_previous_torque_nm[2] = {0.0f, 0.0f};
 *        static unsigned int g_first_policy_step = 1U;
 *
 * 4. 机器人或控制器复位时调用：
 *
 *        void assist_controller_reset(float left_vel, float right_vel)
 *        {
 *            float initial_velocity[2] = {left_vel, right_vel};
 *            g1_assist_policy_history_reset(&g_history, initial_velocity);
 *            g_previous_torque_nm[0] = 0.0f;
 *            g_previous_torque_nm[1] = 0.0f;
 *            g_first_policy_step = 1U;
 *        }
 *
 * 5. 每20 ms执行一次以下控制流程。复位后的第一次推理直接使用reset
 *    填充的历史；从第二次推理开始，追加当前速度和上一周期指令力矩：
 *
 *        void assist_controller_step(float left_vel, float right_vel)
 *        {
 *            float velocity[2] = {left_vel, right_vel};
 *            float torque_nm[2];
 *
 *            if (g_first_policy_step != 0U) {
 *                g_first_policy_step = 0U;
 *            } else {
 *                g1_assist_policy_history_append(
 *                    &g_history, velocity, g_previous_torque_nm);
 *            }
 *
 *            g1_assist_policy_history_build_observation(
 *                &g_history, g_observation);
 *            g1_assist_policy_compute_torque(
 *                g_observation, torque_nm, &g_workspace);
 *
 *            g_previous_torque_nm[0] = torque_nm[0];
 *            g_previous_torque_nm[1] = torque_nm[1];
 *
 *            // 用实际电机接口替换下面这个示例函数。
 *            motor_set_torque_nm(torque_nm[0], torque_nm[1]);
 *        }
 *
 * 6. g1_assist_policy_compute_torque()输出已经完成动作限幅和8 N.m
 *    缩放，左右输出范围均为[-8, 8] N.m。历史中必须保存上一周期的
 *    指令力矩，而不是电机传感器测量的实际力矩。
 *
 * 7. 100维观测排列为：
 *
 *        observation[0..49]  = 25帧[左速度, 右速度]
 *        observation[50..99] = 25帧[左指令力矩, 右指令力矩]
 *
 *    两段历史均按最旧到最新排列。推荐使用本文件提供的history函数，
 *    不要手工拼接观测。
 *
 * 8. 资源占用（1 KB = 1024字节）：
 *
 *        固定float32参数：174312字节Flash，约170.23 KB
 *        推理工作区：       1424字节RAM，  约1.39 KB
 *        25帧历史：          404字节RAM，  约0.39 KB（32位平台）
 *
 *    一个工作区不能被两个中断或RTOS任务同时使用；并发推理时，
 *    每个调用方必须持有独立工作区。
 *
 * 9. g1_assist_policy_forward()只返回未经限幅的两个网络动作，通常应调用
 *    g1_assist_policy_compute_torque()取得最终N.m力矩指令。
 *
 * =================================================================
 */

/*
 * Caller-owned scratch memory. Declare it static/global on small-stack MCUs.
 * One workspace may not be shared by simultaneous calls.
 */
typedef struct {
    float hidden_256[256];
    float secondary_100[100];
} G1AssistPolicyWorkspace;

/* 25 frames of [left, right], stored internally as a circular buffer. */
typedef struct {
    float velocity[G1_ASSIST_POLICY_HISTORY_LENGTH][2];
    float commanded_torque[G1_ASSIST_POLICY_HISTORY_LENGTH][2];
    unsigned int next_index;
} G1AssistPolicyHistory;

/* Fill all velocity frames with the current velocity and all torques with zero. */
void g1_assist_policy_history_reset(
    G1AssistPolicyHistory *history,
    const float initial_velocity[2]);

/* Append current velocity and the previously commanded assist torque. */
void g1_assist_policy_history_append(
    G1AssistPolicyHistory *history,
    const float velocity[2],
    const float previous_torque_nm[2]);

/* Flatten oldest-to-newest velocity history, followed by torque history. */
void g1_assist_policy_history_build_observation(
    const G1AssistPolicyHistory *history,
    float observation[G1_ASSIST_POLICY_INPUT_SIZE]);

/* Compute the two raw policy actions. No clipping is applied. */
void g1_assist_policy_forward(
    const float observation[G1_ASSIST_POLICY_INPUT_SIZE],
    float action[G1_ASSIST_POLICY_OUTPUT_SIZE],
    G1AssistPolicyWorkspace *workspace);

/* Compute actions, clip each to [-1, 1], then multiply by 8 N.m. */
void g1_assist_policy_compute_torque(
    const float observation[G1_ASSIST_POLICY_INPUT_SIZE],
    float torque_nm[G1_ASSIST_POLICY_OUTPUT_SIZE],
    G1AssistPolicyWorkspace *workspace);

#ifdef __cplusplus
}
#endif

#endif /* G1_ASSIST_POLICY_H */
"""

    arrays_source = "\n".join(
        (
            c_array("OBS_MEAN", arrays["obs_mean"]),
            c_array("OBS_STD", arrays["obs_std"]),
            c_array("WEIGHT_0", arrays["weight_0"]),
            c_array("BIAS_0", arrays["bias_0"]),
            c_array("WEIGHT_1", arrays["weight_1"]),
            c_array("BIAS_1", arrays["bias_1"]),
            c_array("WEIGHT_2", arrays["weight_2"]),
            c_array("BIAS_2", arrays["bias_2"]),
            c_array("WEIGHT_3", arrays["weight_3"]),
            c_array("BIAS_3", arrays["bias_3"]),
        )
    )
    source = f"""/*
 * Auto-generated from {weights_path.name}.
 * Network: normalize -> 100x256 ELU -> 256x64 ELU -> 64x16 ELU -> 16x2.
 */
#include "g1_assist_policy.h"

#include <math.h>
#include <stddef.h>

#define NORMALIZER_EPS {eps:.9e}f

{arrays_source}
void g1_assist_policy_history_reset(
    G1AssistPolicyHistory *history,
    const float initial_velocity[2])
{{
    unsigned int frame;
    unsigned int side;
    if (history == NULL || initial_velocity == NULL) {{
        return;
    }}
    for (frame = 0; frame < G1_ASSIST_POLICY_HISTORY_LENGTH; ++frame) {{
        for (side = 0; side < 2; ++side) {{
            history->velocity[frame][side] = initial_velocity[side];
            history->commanded_torque[frame][side] = 0.0f;
        }}
    }}
    history->next_index = 0;
}}

void g1_assist_policy_history_append(
    G1AssistPolicyHistory *history,
    const float velocity[2],
    const float previous_torque_nm[2])
{{
    unsigned int side;
    unsigned int index;
    if (history == NULL || velocity == NULL || previous_torque_nm == NULL) {{
        return;
    }}
    index = history->next_index;
    for (side = 0; side < 2; ++side) {{
        history->velocity[index][side] = velocity[side];
        history->commanded_torque[index][side] = previous_torque_nm[side];
    }}
    history->next_index = (index + 1U) % G1_ASSIST_POLICY_HISTORY_LENGTH;
}}

void g1_assist_policy_history_build_observation(
    const G1AssistPolicyHistory *history,
    float observation[G1_ASSIST_POLICY_INPUT_SIZE])
{{
    unsigned int frame;
    unsigned int side;
    if (history == NULL || observation == NULL) {{
        return;
    }}
    for (frame = 0; frame < G1_ASSIST_POLICY_HISTORY_LENGTH; ++frame) {{
        unsigned int source =
            (history->next_index + frame) % G1_ASSIST_POLICY_HISTORY_LENGTH;
        for (side = 0; side < 2; ++side) {{
            observation[frame * 2U + side] = history->velocity[source][side];
            observation[50U + frame * 2U + side] =
                history->commanded_torque[source][side];
        }}
    }}
}}

static float elu(float value)
{{
    return value >= 0.0f ? value : expf(value) - 1.0f;
}}

static void linear_elu(
    const float *input,
    size_t input_size,
    float *output,
    size_t output_size,
    const float *weight,
    const float *bias)
{{
    size_t row;
    size_t column;
    for (row = 0; row < output_size; ++row) {{
        float sum = bias[row];
        const float *row_weight = weight + row * input_size;
        for (column = 0; column < input_size; ++column) {{
            sum += row_weight[column] * input[column];
        }}
        output[row] = elu(sum);
    }}
}}

void g1_assist_policy_forward(
    const float observation[G1_ASSIST_POLICY_INPUT_SIZE],
    float action[G1_ASSIST_POLICY_OUTPUT_SIZE],
    G1AssistPolicyWorkspace *workspace)
{{
    size_t index;
    size_t row;
    size_t column;

    if (observation == NULL || action == NULL || workspace == NULL) {{
        return;
    }}

    for (index = 0; index < G1_ASSIST_POLICY_INPUT_SIZE; ++index) {{
        workspace->secondary_100[index] =
            (observation[index] - OBS_MEAN[index]) / (OBS_STD[index] + NORMALIZER_EPS);
    }}

    linear_elu(
        workspace->secondary_100, 100, workspace->hidden_256, 256, WEIGHT_0, BIAS_0);
    linear_elu(
        workspace->hidden_256, 256, workspace->secondary_100, 64, WEIGHT_1, BIAS_1);
    linear_elu(
        workspace->secondary_100, 64, workspace->hidden_256, 16, WEIGHT_2, BIAS_2);

    for (row = 0; row < G1_ASSIST_POLICY_OUTPUT_SIZE; ++row) {{
        float sum = BIAS_3[row];
        const float *row_weight = WEIGHT_3 + row * 16;
        for (column = 0; column < 16; ++column) {{
            sum += row_weight[column] * workspace->hidden_256[column];
        }}
        action[row] = sum;
    }}
}}

void g1_assist_policy_compute_torque(
    const float observation[G1_ASSIST_POLICY_INPUT_SIZE],
    float torque_nm[G1_ASSIST_POLICY_OUTPUT_SIZE],
    G1AssistPolicyWorkspace *workspace)
{{
    float action[G1_ASSIST_POLICY_OUTPUT_SIZE];
    size_t index;

    if (observation == NULL || torque_nm == NULL || workspace == NULL) {{
        return;
    }}
    g1_assist_policy_forward(observation, action, workspace);
    for (index = 0; index < G1_ASSIST_POLICY_OUTPUT_SIZE; ++index) {{
        float clipped = action[index];
        if (clipped > 1.0f) {{
            clipped = 1.0f;
        }} else if (clipped < -1.0f) {{
            clipped = -1.0f;
        }}
        torque_nm[index] = clipped * G1_ASSIST_POLICY_TORQUE_LIMIT_NM;
    }}
}}
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / "g1_assist_policy.h"
    source_path = output_dir / "g1_assist_policy.c"
    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    print(f"Header: {header_path}")
    print(f"Source: {source_path}")
    print("Parameter storage: 174312 bytes of float32 constants")
    print("Workspace: 1424 bytes per inference context")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate standalone C99 source/header for the fixed v2 assist policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights/assist_policy_v2.npz"
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
        "obs_mean": (150,),
        "obs_std": (150,),
        "weight_0": (256, 150),
        "bias_0": (256,),
        "weight_1": (64, 256),
        "bias_1": (64,),
        "weight_2": (16, 64),
        "bias_2": (16,),
        "weight_3": (2, 16),
        "bias_3": (2,),
    }
    for name, expected in expected_shapes.items():
        actual = None if name not in arrays else arrays[name].shape
        if actual != expected:
            raise ValueError(f"{name} has shape {actual}, expected {expected}")

    eps = float(np.asarray(arrays["normalizer_eps"], dtype=np.float32))
    header = """/* Auto-generated G1 assist-exoskeleton v2 policy interface. */
#ifndef G1_ASSIST_V2_POLICY_H
#define G1_ASSIST_V2_POLICY_H

#ifdef __cplusplus
extern "C" {
#endif

#define G1_ASSIST_V2_INPUT_SIZE 150
#define G1_ASSIST_V2_OUTPUT_SIZE 2
#define G1_ASSIST_V2_HISTORY_LENGTH 25
#define G1_ASSIST_V2_POLICY_RATE_HZ 100
#define G1_ASSIST_V2_TORQUE_LIMIT_NM 8.0f
#define G1_ASSIST_V2_MAX_TORQUE_DELTA_NM 0.8f
#define G1_ASSIST_V2_COMMAND_SPEED_DEADZONE_M_S 0.2f
#define G1_ASSIST_V2_COMMAND_SPEED_FULL_M_S 0.7f

/*
 * ========================= 移植与使用说明 =========================
 *
 * 一、文件和编译
 *
 * 1. 将 g1_assist_v2_policy.c 和 g1_assist_v2_policy.h 加入控制器工程。
 *    网络参数已经固化在.c中，运行时不需要policy.pt、NPZ、Python、
 *    PyTorch、文件系统或动态内存。
 *
 * 2. 源码使用C99和expf()。GCC示例：
 *
 *      gcc -std=c99 -O2 controller.c g1_assist_v2_policy.c -lm
 *
 *    C++工程可直接包含本头文件，接口已使用extern "C"。建议使用带单精度
 *    FPU的MCU/CPU，并在目标硬件上实测一次推理的最坏执行时间。
 *
 * 3. 固定float32网络参数约224712字节，通常应由链接脚本放入Flash/ROM。
 *    每个推理实例至少需要：Workspace 1624字节、History约604字节、
 *    observation 600字节，另加少量控制状态。建议均声明为static或全局变量，
 *    不要放在小容量任务栈中。接口不分配内存。
 *
 * 二、时序、单位和观测
 *
 * 1. 策略必须每10 ms调用一次，即100 Hz。若电机力矩内环为1 ms，应在两个
 *    策略周期之间保持最近一次motor_torque_nm，不要每1 ms重复运行策略。
 *
 * 2. 所有二维量的顺序固定为[左, 右]。角度单位是rad，角速度单位是rad/s，
 *    力矩单位是Nm。不要把degree或degree/s直接传入。
 *
 * 3. 150维观测由本文件的history函数自动构造：
 *
 *      observation[0..49]    = 25帧[左角度, 右角度]
 *      observation[50..99]   = 25帧[左角速度, 右角速度]
 *      observation[100..149] = 25帧[左平滑力矩, 右平滑力矩]
 *
 *    三段历史均按最旧到最新排列。角度和速度是外骨骼关节自身的传感量，
 *    不需要人体真实髋关节角度、速度或力矩。力矩历史必须保存上一策略周期
 *    “未乘最终output_scale”的平滑策略力矩，不是电机测得力矩，也不是最终
 *    缩放后的电机指令。
 *
 * 4. policy_step先根据|vx|指令应用0.2到0.7 m/s的smoothstep门控，再把动作
 *    限幅到[-1,1]并换算为[-8,8] Nm，最后将每10 ms最大变化限制为0.8 Nm，
 *    即80 Nm/s。apply_output_scale只在最后写电机前
 *    乘系数，因此不会直接改变policy、obs、历史或平滑过程。最终电机指令
 *    始终限幅在[-8,8] Nm。非法或负output_scale按0处理。
 *
 * 三、最小控制器示例
 *
 *      #include "g1_assist_v2_policy.h"
 *
 *      static G1AssistV2History history;
 *      static G1AssistV2Workspace workspace;
 *      static float observation[G1_ASSIST_V2_INPUT_SIZE];
 *      static float previous_smoothed_nm[2] = {0.0f, 0.0f};
 *      static unsigned int first_policy_step = 1U;
 *      static float assist_output_scale = 0.0f; // 台架测试从0开始
 *
 *      void assist_controller_reset(
 *          const float position_rad[2], const float velocity_rad_s[2])
 *      {
 *          g1_assist_v2_history_reset(
 *              &history, position_rad, velocity_rad_s);
 *          previous_smoothed_nm[0] = 0.0f;
 *          previous_smoothed_nm[1] = 0.0f;
 *          first_policy_step = 1U;
 *          motor_set_torque_nm(0.0f, 0.0f);
 *      }
 *
 *      // 由严格10 ms周期任务调用一次。
 *      void assist_controller_tick_10ms(
 *          const float position_rad[2], const float velocity_rad_s[2],
 *          float forward_command_m_s)
 *      {
 *          float smoothed_nm[2];
 *          float motor_nm[2];
 *
 *          // 复位后的第一次推理直接使用reset填满的25帧历史。
 *          if (first_policy_step != 0U) {
 *              first_policy_step = 0U;
 *          } else {
 *              g1_assist_v2_history_append(
 *                  &history, position_rad, velocity_rad_s,
 *                  previous_smoothed_nm);
 *          }
 *
 *          g1_assist_v2_history_build_observation(
 *              &history, observation);
 *          g1_assist_v2_policy_step(
 *              observation, previous_smoothed_nm, forward_command_m_s,
 *              smoothed_nm, &workspace);
 *
 *          // 历史保存未缩放的平滑策略力矩。
 *          previous_smoothed_nm[0] = smoothed_nm[0];
 *          previous_smoothed_nm[1] = smoothed_nm[1];
 *
 *          // 系数仅在最终电机输入处生效。
 *          g1_assist_v2_apply_output_scale(
 *              smoothed_nm, assist_output_scale, motor_nm);
 *          motor_set_torque_nm(motor_nm[0], motor_nm[1]);
 *      }
 *
 * 四、复位、故障和并发
 *
 * 1. 上电、使能、急停恢复、编码器重新置零或控制状态切换后，都应先调用
 *    history_reset并把previous_smoothed_nm清零，再允许策略输出。
 *
 * 2. 传感器无效、通信超时、策略任务超时或安全状态触发时，应由上层状态机
 *    立即把电机指令置零/失能；恢复前重新reset。此策略接口不替代硬件急停、
 *    电流限制、机械限位、看门狗和通信故障保护。
 *
 * 3. 一个Workspace和History只能由一个推理上下文使用。中断或RTOS多任务
 *    并发推理时，每个实例必须使用独立对象，或由互斥机制保证不重入。
 *
 * 4. 本策略仅输出左右髋部两个助力力矩。后部滑块和圆柱机构的零位PD控制
 *    需要在硬件控制层独立实现。
 *
 * 5. 首次上真实设备前，先空载检查左右方向、编码器零位、单位、100 Hz调度、
 *    力矩限幅、超时归零和急停；output_scale从0开始逐步增加。
 *
 * ================================================================
 */
typedef struct {
    float hidden_256[256];
    float secondary_150[150];
} G1AssistV2Workspace;

typedef struct {
    float position_rad[G1_ASSIST_V2_HISTORY_LENGTH][2];
    float velocity_rad_s[G1_ASSIST_V2_HISTORY_LENGTH][2];
    float smoothed_torque_nm[G1_ASSIST_V2_HISTORY_LENGTH][2];
    unsigned int next_index;
} G1AssistV2History;

/* Fill all 25 position/velocity frames from current sensors and torque with zero. */
void g1_assist_v2_history_reset(
    G1AssistV2History *history,
    const float initial_position_rad[2],
    const float initial_velocity_rad_s[2]);

/* Append one 10 ms sample; torque must be previous unscaled smoothed torque. */
void g1_assist_v2_history_append(
    G1AssistV2History *history,
    const float position_rad[2],
    const float velocity_rad_s[2],
    const float previous_smoothed_torque_nm[2]);

/* Flatten the three circular histories into the exact 150-element policy input. */
void g1_assist_v2_history_build_observation(
    const G1AssistV2History *history,
    float observation[G1_ASSIST_V2_INPUT_SIZE]);

/* Return two raw network actions; no action clipping or torque conversion. */
void g1_assist_v2_policy_forward(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float action[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace);

/* Smooth assist gate: zero through |vx|=0.2 m/s and one from |vx|=0.7 m/s. */
float g1_assist_v2_command_speed_gate(float forward_command_m_s);

/* Apply the speed gate, clip actions, and convert them to target torque. */
void g1_assist_v2_compute_target_torque(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float forward_command_m_s,
    float target_torque_nm[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace);

/* Limit each torque change to +/-0.8 Nm for one 10 ms policy cycle. */
void g1_assist_v2_slew_limit(
    const float previous_torque_nm[2],
    const float target_torque_nm[2],
    float smoothed_torque_nm[2]);

/* Run inference, target conversion and 80 Nm/s slew limiting in one call. */
void g1_assist_v2_policy_step(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    const float previous_torque_nm[2],
    float forward_command_m_s,
    float smoothed_torque_nm[2],
    G1AssistV2Workspace *workspace);

/* Apply final-only deployment gain and clip the actual motor command to +/-8 Nm. */
void g1_assist_v2_apply_output_scale(
    const float smoothed_torque_nm[2],
    float output_scale,
    float motor_torque_nm[2]);

#ifdef __cplusplus
}
#endif
#endif
"""

    arrays_source = "\n".join(
        c_array(name.upper(), arrays[name])
        for name in (
            "obs_mean",
            "obs_std",
            "weight_0",
            "bias_0",
            "weight_1",
            "bias_1",
            "weight_2",
            "bias_2",
            "weight_3",
            "bias_3",
        )
    )
    source = f"""/* Auto-generated from {weights_path.name}. */
#include "g1_assist_v2_policy.h"

#include <math.h>
#include <stddef.h>

#define NORMALIZER_EPS {eps:.9e}f

{arrays_source}
static float clip(float value, float lower, float upper)
{{
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}}

static float elu(float value)
{{
    return value >= 0.0f ? value : expf(value) - 1.0f;
}}

static void linear_elu(
    const float *input, size_t input_size, float *output, size_t output_size,
    const float *weight, const float *bias)
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

void g1_assist_v2_history_reset(
    G1AssistV2History *history,
    const float initial_position_rad[2],
    const float initial_velocity_rad_s[2])
{{
    unsigned int frame;
    unsigned int side;
    if (history == NULL || initial_position_rad == NULL || initial_velocity_rad_s == NULL) return;
    for (frame = 0; frame < G1_ASSIST_V2_HISTORY_LENGTH; ++frame) {{
        for (side = 0; side < 2; ++side) {{
            history->position_rad[frame][side] = initial_position_rad[side];
            history->velocity_rad_s[frame][side] = initial_velocity_rad_s[side];
            history->smoothed_torque_nm[frame][side] = 0.0f;
        }}
    }}
    history->next_index = 0U;
}}

void g1_assist_v2_history_append(
    G1AssistV2History *history,
    const float position_rad[2],
    const float velocity_rad_s[2],
    const float previous_smoothed_torque_nm[2])
{{
    unsigned int side;
    unsigned int index;
    if (history == NULL || position_rad == NULL || velocity_rad_s == NULL || previous_smoothed_torque_nm == NULL) return;
    index = history->next_index;
    for (side = 0; side < 2; ++side) {{
        history->position_rad[index][side] = position_rad[side];
        history->velocity_rad_s[index][side] = velocity_rad_s[side];
        history->smoothed_torque_nm[index][side] = previous_smoothed_torque_nm[side];
    }}
    history->next_index = (index + 1U) % G1_ASSIST_V2_HISTORY_LENGTH;
}}

void g1_assist_v2_history_build_observation(
    const G1AssistV2History *history,
    float observation[G1_ASSIST_V2_INPUT_SIZE])
{{
    unsigned int frame;
    unsigned int side;
    if (history == NULL || observation == NULL) return;
    for (frame = 0; frame < G1_ASSIST_V2_HISTORY_LENGTH; ++frame) {{
        unsigned int source = (history->next_index + frame) % G1_ASSIST_V2_HISTORY_LENGTH;
        for (side = 0; side < 2; ++side) {{
            observation[frame * 2U + side] = history->position_rad[source][side];
            observation[50U + frame * 2U + side] = history->velocity_rad_s[source][side];
            observation[100U + frame * 2U + side] = history->smoothed_torque_nm[source][side];
        }}
    }}
}}

void g1_assist_v2_policy_forward(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float action[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace)
{{
    size_t index;
    size_t row;
    size_t column;
    if (observation == NULL || action == NULL || workspace == NULL) return;
    for (index = 0; index < G1_ASSIST_V2_INPUT_SIZE; ++index) {{
        workspace->secondary_150[index] =
            (observation[index] - OBS_MEAN[index]) / (OBS_STD[index] + NORMALIZER_EPS);
    }}
    linear_elu(workspace->secondary_150, 150, workspace->hidden_256, 256, WEIGHT_0, BIAS_0);
    linear_elu(workspace->hidden_256, 256, workspace->secondary_150, 64, WEIGHT_1, BIAS_1);
    linear_elu(workspace->secondary_150, 64, workspace->hidden_256, 16, WEIGHT_2, BIAS_2);
    for (row = 0; row < G1_ASSIST_V2_OUTPUT_SIZE; ++row) {{
        float sum = BIAS_3[row];
        const float *row_weight = WEIGHT_3 + row * 16U;
        for (column = 0; column < 16; ++column) sum += row_weight[column] * workspace->hidden_256[column];
        action[row] = sum;
    }}
}}

float g1_assist_v2_command_speed_gate(float forward_command_m_s)
{{
    float phase;
    if (!isfinite(forward_command_m_s)) return 0.0f;
    phase = clip(
        (fabsf(forward_command_m_s) - G1_ASSIST_V2_COMMAND_SPEED_DEADZONE_M_S)
            / (G1_ASSIST_V2_COMMAND_SPEED_FULL_M_S - G1_ASSIST_V2_COMMAND_SPEED_DEADZONE_M_S),
        0.0f,
        1.0f);
    return phase * phase * (3.0f - 2.0f * phase);
}}

void g1_assist_v2_compute_target_torque(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float forward_command_m_s,
    float target_torque_nm[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace)
{{
    float action[2];
    float gate;
    size_t side;
    if (observation == NULL || target_torque_nm == NULL || workspace == NULL) return;
    g1_assist_v2_policy_forward(observation, action, workspace);
    gate = g1_assist_v2_command_speed_gate(forward_command_m_s);
    for (side = 0; side < 2; ++side) {{
        target_torque_nm[side] =
            gate * clip(action[side], -1.0f, 1.0f) * G1_ASSIST_V2_TORQUE_LIMIT_NM;
    }}
}}

void g1_assist_v2_slew_limit(
    const float previous_torque_nm[2],
    const float target_torque_nm[2],
    float smoothed_torque_nm[2])
{{
    size_t side;
    if (previous_torque_nm == NULL || target_torque_nm == NULL || smoothed_torque_nm == NULL) return;
    for (side = 0; side < 2; ++side) {{
        float delta = clip(
            target_torque_nm[side] - previous_torque_nm[side],
            -G1_ASSIST_V2_MAX_TORQUE_DELTA_NM,
            G1_ASSIST_V2_MAX_TORQUE_DELTA_NM);
        smoothed_torque_nm[side] = previous_torque_nm[side] + delta;
    }}
}}

void g1_assist_v2_policy_step(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    const float previous_torque_nm[2],
    float forward_command_m_s,
    float smoothed_torque_nm[2],
    G1AssistV2Workspace *workspace)
{{
    float target_torque_nm[2];
    if (observation == NULL || previous_torque_nm == NULL || smoothed_torque_nm == NULL || workspace == NULL) return;
    g1_assist_v2_compute_target_torque(
        observation, forward_command_m_s, target_torque_nm, workspace);
    g1_assist_v2_slew_limit(previous_torque_nm, target_torque_nm, smoothed_torque_nm);
}}

void g1_assist_v2_apply_output_scale(
    const float smoothed_torque_nm[2],
    float output_scale,
    float motor_torque_nm[2])
{{
    size_t side;
    if (smoothed_torque_nm == NULL || motor_torque_nm == NULL) return;
    if (!isfinite(output_scale) || output_scale < 0.0f) output_scale = 0.0f;
    for (side = 0; side < 2; ++side) {{
        motor_torque_nm[side] = clip(
            output_scale * smoothed_torque_nm[side],
            -G1_ASSIST_V2_TORQUE_LIMIT_NM,
            G1_ASSIST_V2_TORQUE_LIMIT_NM);
    }}
}}
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / "g1_assist_v2_policy.h"
    source_path = output_dir / "g1_assist_v2_policy.c"
    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    parameter_count = sum(
        arrays[f"weight_{i}"].size + arrays[f"bias_{i}"].size for i in range(4)
    )
    print(f"Header:            {header_path}")
    print(f"Source:            {source_path}")
    print(f"Parameter storage: {parameter_count * 4} bytes float32")
    print("Workspace:         1624 bytes per inference context")


if __name__ == "__main__":
    main()

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
#define G1_ASSIST_V2_MAX_TORQUE_DELTA_NM 0.4f

/*
 * Call the policy every 10 ms. Units are rad, rad/s and Nm; side order is
 * [left, right]. Observation order is 25 position frames, then 25 velocity
 * frames, then 25 previous smoothed-torque frames, all oldest-to-newest.
 *
 * The output scale is intentionally applied only by
 * g1_assist_v2_apply_output_scale(), after policy inference and slew limiting.
 * It therefore never directly modifies the policy observation or history.
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

void g1_assist_v2_history_reset(
    G1AssistV2History *history,
    const float initial_position_rad[2],
    const float initial_velocity_rad_s[2]);

void g1_assist_v2_history_append(
    G1AssistV2History *history,
    const float position_rad[2],
    const float velocity_rad_s[2],
    const float previous_smoothed_torque_nm[2]);

void g1_assist_v2_history_build_observation(
    const G1AssistV2History *history,
    float observation[G1_ASSIST_V2_INPUT_SIZE]);

void g1_assist_v2_policy_forward(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float action[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace);

void g1_assist_v2_compute_target_torque(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float target_torque_nm[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace);

void g1_assist_v2_slew_limit(
    const float previous_torque_nm[2],
    const float target_torque_nm[2],
    float smoothed_torque_nm[2]);

void g1_assist_v2_policy_step(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    const float previous_torque_nm[2],
    float smoothed_torque_nm[2],
    G1AssistV2Workspace *workspace);

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

void g1_assist_v2_compute_target_torque(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float target_torque_nm[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace)
{{
    float action[2];
    size_t side;
    if (observation == NULL || target_torque_nm == NULL || workspace == NULL) return;
    g1_assist_v2_policy_forward(observation, action, workspace);
    for (side = 0; side < 2; ++side) {{
        target_torque_nm[side] = clip(action[side], -1.0f, 1.0f) * G1_ASSIST_V2_TORQUE_LIMIT_NM;
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
    float smoothed_torque_nm[2],
    G1AssistV2Workspace *workspace)
{{
    float target_torque_nm[2];
    if (observation == NULL || previous_torque_nm == NULL || smoothed_torque_nm == NULL || workspace == NULL) return;
    g1_assist_v2_compute_target_torque(observation, target_torque_nm, workspace);
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

#!/usr/bin/env python3
"""Generate standalone C99 source/header for the fixed v5 assist policy."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights/assist_policy.npz"
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
    manifest = json.loads(weights_path.with_name("manifest.json").read_text())
    expected = dict(dt=.01, torque_limit=10., torque_rate_limit=80., motion_filter_time_constant=.05,
                    motion_speed_deadzone=.15, motion_speed_full=.8, pulse_rise_time=.25, pulse_release_duration=.25, pulse_duration_min=.6, pulse_duration_max=1.4, pulse_confirm_time=.06)
    if manifest["control"] != expected:
        raise ValueError("C template requires the verified v5 control parameters")
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != manifest["npz_sha256"]:
        raise ValueError("Weights hash mismatch")
    with np.load(weights_path, allow_pickle=False) as data:
        arrays = {name: np.array(data[name], copy=True) for name in data.files}

    expected_shapes = {
        "obs_mean": (450,),
        "obs_std": (450,),
        "weight_0": (256, 450),
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
    header = """/* Generated fixed-weight v5 assist policy; C99, no heap or file I/O. */
#ifndef G1_ASSIST_V5_POLICY_H
#define G1_ASSIST_V5_POLICY_H
#ifdef __cplusplus
extern "C" {
#endif
#define G1_ASSIST_V5_INPUT_SIZE 450
#define G1_ASSIST_V5_OUTPUT_SIZE 2
#define G1_ASSIST_V5_HISTORY_LENGTH 25
#define G1_ASSIST_V5_POLICY_RATE_HZ 100
/* Left/right order; radians, rad/s, Nm. Paired torque [T,-T], +/-10 Nm, latched peak/duration, 0.25s rise/release, 0.6-1.4s total. */
typedef struct { float hidden_256[256]; float secondary_450[450]; } G1AssistV5Workspace;
typedef struct {
    float position_rad[25][2], velocity_rad_s[25][2], smoothed_torque_nm[25][2], pulse_state[25][12];
    unsigned int next_index;
} G1AssistV5History;
typedef struct {
    G1AssistV5History history;
    G1AssistV5Workspace workspace;
    float observation[450], previous[2], filtered_velocity[2], pulse[10];
    unsigned int ready, first;
} G1AssistV5Controller;
void g1_assist_v5_history_reset(G1AssistV5History *, const float[2], const float[2]);
void g1_assist_v5_history_append(G1AssistV5History *, const float[2], const float[2], const float[2], const float[12]);
void g1_assist_v5_history_build_observation(const G1AssistV5History *, float[450]);
/* Low-level forward requires finite inputs; returns raw network actions. */
void g1_assist_v5_policy_forward(const float[450], float[2], G1AssistV5Workspace *);
/* reset: fills history. Return 0 success, -1 invalid inputs. */
int g1_assist_v5_reset(G1AssistV5Controller *, const float[2], const float[2]);
/* stop zeroes output and latches disabled until reset; does not write hardware. */
void g1_assist_v5_stop(G1AssistV5Controller *, float[2]);
/* Exactly once per 10 ms. No vx input. Returns 0 success, 1 disabled, -1 fault.
 * On fault/disabled, output is zero and controller is latched stopped.
 * Caller sends output to hardware, handles stale samples and deadlines.
 * State/workspace must be unique to a controller, not shared across threads.
 */
int g1_assist_v5_step(G1AssistV5Controller *, const float[2], const float[2], int, float[2]);
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
#include "g1_assist_v5_policy.h"

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

void g1_assist_v5_history_reset(
    G1AssistV5History *history,
    const float initial_position_rad[2],
    const float initial_velocity_rad_s[2])
{{
    unsigned int frame;
    unsigned int side;
    if (history == NULL || initial_position_rad == NULL || initial_velocity_rad_s == NULL) return;
    for (frame = 0; frame < G1_ASSIST_V5_HISTORY_LENGTH; ++frame) {{
        for (side = 0; side < 2; ++side) {{
            history->position_rad[frame][side] = initial_position_rad[side];
            history->velocity_rad_s[frame][side] = initial_velocity_rad_s[side];
            history->smoothed_torque_nm[frame][side] = 0.0f;
        }}
    }}
    for (frame = 0; frame < 25; ++frame)
        for (side = 0; side < 12; ++side) history->pulse_state[frame][side] = 0.0f;
    history->next_index = 0U;
}}

void g1_assist_v5_history_append(
    G1AssistV5History *history,
    const float position_rad[2],
    const float velocity_rad_s[2],
    const float previous_smoothed_torque_nm[2], const float pulse_state[12])
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
    for (side = 0; side < 12; ++side) history->pulse_state[index][side] = pulse_state[side];
    history->next_index = (index + 1U) % G1_ASSIST_V5_HISTORY_LENGTH;
}}

void g1_assist_v5_history_build_observation(
    const G1AssistV5History *history,
    float observation[G1_ASSIST_V5_INPUT_SIZE])
{{
    unsigned int frame;
    unsigned int side;
    if (history == NULL || observation == NULL) return;
    for (frame = 0; frame < G1_ASSIST_V5_HISTORY_LENGTH; ++frame) {{
        unsigned int source = (history->next_index + frame) % G1_ASSIST_V5_HISTORY_LENGTH;
        for (side = 0; side < 2; ++side) {{
            observation[frame * 2U + side] = history->position_rad[source][side];
            observation[50U + frame * 2U + side] = history->velocity_rad_s[source][side];
            observation[100U + frame * 2U + side] = history->smoothed_torque_nm[source][side];
        }}
        for (side = 0; side < 12; ++side) observation[150U + frame * 12U + side] = history->pulse_state[source][side];
    }}
}}

void g1_assist_v5_policy_forward(
    const float observation[G1_ASSIST_V5_INPUT_SIZE],
    float action[G1_ASSIST_V5_OUTPUT_SIZE],
    G1AssistV5Workspace *workspace)
{{
    size_t index;
    size_t row;
    size_t column;
    if (observation == NULL || action == NULL || workspace == NULL) return;
    for (index = 0; index < G1_ASSIST_V5_INPUT_SIZE; ++index) {{
        workspace->secondary_450[index] =
            (observation[index] - OBS_MEAN[index]) / (OBS_STD[index] + NORMALIZER_EPS);
    }}
    linear_elu(workspace->secondary_450, 450, workspace->hidden_256, 256, WEIGHT_0, BIAS_0);
    linear_elu(workspace->hidden_256, 256, workspace->secondary_450, 64, WEIGHT_1, BIAS_1);
    linear_elu(workspace->secondary_450, 64, workspace->hidden_256, 16, WEIGHT_2, BIAS_2);
    for (row = 0; row < G1_ASSIST_V5_OUTPUT_SIZE; ++row) {{
        float sum = BIAS_3[row];
        const float *row_weight = WEIGHT_3 + row * 16U;
        for (column = 0; column < 16; ++column) sum += row_weight[column] * workspace->hidden_256[column];
        action[row] = sum;
    }}
}}


static float pulse_step(G1AssistV5Controller *c, const float action[2])
{{
    float *s=c->pulse;
    float delta=c->filtered_velocity[0]-c->filtered_velocity[1];
    float direction=(delta<-.12f && c->filtered_velocity[0]<-.08f) ? -1.0f :
        (delta>.12f && c->filtered_velocity[1]<-.08f) ? 1.0f : 0.0f;
    s[7]=direction==s[6] ? fminf(s[7]+.01f,.06f) : .01f;
    s[6]=direction;
    int confirmed=s[7]>=.06f-1e-6f;
    int quiet=fmaxf(fabsf(c->filtered_velocity[0]),fabsf(c->filtered_velocity[1]))<.04f;
    s[9]=quiet ? fminf(s[9]+.01f,.3f) : 0;
    int active=s[1]!=0, releasing=s[3]>0;
    int lowering=(s[1]<0 ? c->filtered_velocity[0] : c->filtered_velocity[1])>.12f;
    int reverse=confirmed && direction!=0 && direction!=s[1];
    int quiet_end=s[9]>=.3f-1e-6f && s[0]>=.5f;
    int timed_end=s[0]>=s[8]-.25f-1e-6f;
    if(active && !releasing && (lowering || reverse || quiet_end || timed_end)) {{
        s[3]=.01f; s[4]=fabsf(c->previous[0]);
    }}
    releasing=s[3]>0;
    if(!active && confirmed && direction==0) s[5]=0;
    if(!active && confirmed && direction!=0 && direction!=s[5]) {{
        s[0]=0; s[1]=direction; s[2]=.5f*(clip(action[0],-1,1)+1)*10;
        s[8]=.6f+.5f*(clip(action[1],-1,1)+1)*(1.4f-.6f);
        s[5]=direction; s[3]=s[4]=0;
    }}
    active=s[1]!=0;
    if(active) s[0]+=.01f;
    float rise=clip(s[0]/.25f,0,1), rp=clip(s[3]/.25f,0,1);
    float wave=releasing ? cosf(1.5707963267948966f*rp) : sinf(1.5707963267948966f*rise);
    float magnitude=(releasing ? s[4] : s[2])*wave*wave;
    float result=active ? s[1]*magnitude : 0;
    int done=active && releasing && s[3]>=.25f-1e-6f;
    if(done) {{ result=0; for(int i=0;i<5;++i) s[i]=0; s[8]=0; }}
    if(releasing && !done) s[3]+=.01f;
    return result;
}}

static int finite_pair(const float *x)
{{
    return x != NULL && isfinite(x[0]) && isfinite(x[1]);
}}
void g1_assist_v5_stop(G1AssistV5Controller *c, float out[2])
{{
    if (out != NULL) out[0] = out[1] = 0.0f;
    if (c == NULL) return;
    c->ready = 0U;
    for (size_t i=0; i<10; ++i) c->pulse[i]=0.0f;
    c->previous[0] = c->previous[1] = 0.0f;
    c->filtered_velocity[0] = c->filtered_velocity[1] = 0.0f;
}}
int g1_assist_v5_reset(G1AssistV5Controller *c, const float p[2], const float v[2])
{{
    if (c == NULL) return -1;
    g1_assist_v5_stop(c, NULL);
    if (!finite_pair(p) || !finite_pair(v)) return -1;
    g1_assist_v5_history_reset(&c->history, p, v);
    c->first = 1U;
    c->ready = 1U;
    return 0;
}}
int g1_assist_v5_step(G1AssistV5Controller *c, const float p[2], const float v[2], int enabled, float out[2])
{{
    float action[2], state[12], output;
    const float scale[10]={{1.4f,1,10,.25f,10,1,1,.06f,1.4f,.3f}};
    size_t side;
    if (c == NULL || out == NULL) {{
        g1_assist_v5_stop(c, out);
        return -1;
    }}
    out[0] = out[1] = 0;
    if (!enabled || !c->ready) {{
        g1_assist_v5_stop(c, out);
        return 1;
    }}
    if (!finite_pair(p) || !finite_pair(v)) {{
        g1_assist_v5_stop(c, out);
        return -1;
    }}
    if (c->first) g1_assist_v5_history_reset(&c->history, p, v);
    else {{
        for (side=0; side<10; ++side) state[side]=c->pulse[side]/scale[side];
        state[10]=c->filtered_velocity[0]; state[11]=c->filtered_velocity[1];
        g1_assist_v5_history_append(&c->history, p, v, c->previous, state);
    }}
    c->first = 0U;
    g1_assist_v5_history_build_observation(&c->history, c->observation);
    g1_assist_v5_policy_forward(c->observation, action, &c->workspace);
    if (!finite_pair(action)) {{
        g1_assist_v5_stop(c, out);
        return -1;
    }}
    for (side = 0; side < 2; ++side) {{
        c->filtered_velocity[side] += (1.0f-expf(-0.01f/0.05f)) * (v[side]-c->filtered_velocity[side]);
    }}
    output = pulse_step(c, action);
    c->previous[0]=output; c->previous[1]=-output;
    out[0]=output; out[1]=-output;
    return 0;
}}

"""

    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / "g1_assist_v5_policy.h"
    source_path = output_dir / "g1_assist_v5_policy.c"
    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    parameter_count = sum(
        arrays[f"weight_{i}"].size + arrays[f"bias_{i}"].size for i in range(4)
    )
    print(f"Header:            {header_path}")
    print(f"Source:            {source_path}")
    print(f"Parameter storage: {parameter_count * 4} bytes float32")
    print("Workspace:         2824 bytes per inference context")


if __name__ == "__main__":
    main()


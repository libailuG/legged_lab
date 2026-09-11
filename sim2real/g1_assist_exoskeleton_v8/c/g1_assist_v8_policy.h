/* Generated fixed-weight v8 assist policy; C99, no heap or file I/O. */
#ifndef G1_ASSIST_V8_POLICY_H
#define G1_ASSIST_V8_POLICY_H
#ifdef __cplusplus
extern "C" {
#endif
#define G1_ASSIST_V8_INPUT_SIZE 650
#define G1_ASSIST_V8_OUTPUT_SIZE 2
#define G1_ASSIST_V8_HISTORY_LENGTH 25
#define G1_ASSIST_V8_POLICY_RATE_HZ 100
/* Left/right order; radians, rad/s, Nm. Paired torque [T,-T], +/-10 Nm, latched peak/duration, displacement-qualified and cadence-adaptive timing. */
typedef struct { float hidden_256[256]; float secondary_650[650]; } G1AssistV8Workspace;
typedef struct {
    float position_rad[25][2], velocity_rad_s[25][2], smoothed_torque_nm[25][2], pulse_state[25][20];
    unsigned int next_index;
} G1AssistV8History;
typedef struct {
    G1AssistV8History history;
    G1AssistV8Workspace workspace;
    float observation[650], previous[2], filtered_velocity[2], pulse[18], displacement[2];
    unsigned int ready, first;
} G1AssistV8Controller;
void g1_assist_v8_history_reset(G1AssistV8History *, const float[2], const float[2]);
void g1_assist_v8_history_append(G1AssistV8History *, const float[2], const float[2], const float[2], const float[20]);
void g1_assist_v8_history_build_observation(const G1AssistV8History *, float[650]);
/* Low-level forward requires finite inputs; returns raw network actions. */
void g1_assist_v8_policy_forward(const float[650], float[2], G1AssistV8Workspace *);
/* reset: fills history. Return 0 success, -1 invalid inputs. */
int g1_assist_v8_reset(G1AssistV8Controller *, const float[2], const float[2]);
/* stop zeroes output and latches disabled until reset; does not write hardware. */
void g1_assist_v8_stop(G1AssistV8Controller *, float[2]);
/* Exactly once per 10 ms. No vx input. Returns 0 success, 1 disabled, -1 fault.
 * On fault/disabled, output is zero and controller is latched stopped.
 * Caller sends output to hardware, handles stale samples and deadlines.
 * State/workspace must be unique to a controller, not shared across threads.
 */
int g1_assist_v8_step(G1AssistV8Controller *, const float[2], const float[2], int, float[2]);
#ifdef __cplusplus
}
#endif
#endif

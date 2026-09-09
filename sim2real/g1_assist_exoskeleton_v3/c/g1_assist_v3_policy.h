/* Generated fixed-weight v3 assist policy; C99, no heap or file I/O. */
#ifndef G1_ASSIST_V3_POLICY_H
#define G1_ASSIST_V3_POLICY_H
#ifdef __cplusplus
extern "C" {
#endif
#define G1_ASSIST_V3_INPUT_SIZE 150
#define G1_ASSIST_V3_OUTPUT_SIZE 1
#define G1_ASSIST_V3_HISTORY_LENGTH 25
#define G1_ASSIST_V3_POLICY_RATE_HZ 100
/* Left/right order; radians, rad/s, Nm. Paired torque [T,-T], +/-10 Nm, shared speed gate, 80 Nm/s slew. */
typedef struct { float hidden_256[256]; float secondary_150[150]; } G1AssistV3Workspace;
typedef struct {
    float position_rad[25][2], velocity_rad_s[25][2], smoothed_torque_nm[25][2];
    unsigned int next_index;
} G1AssistV3History;
typedef struct {
    G1AssistV3History history;
    G1AssistV3Workspace workspace;
    float observation[150], previous[2], filtered_velocity[2];
    unsigned int ready, first;
} G1AssistV3Controller;
void g1_assist_v3_history_reset(G1AssistV3History *, const float[2], const float[2]);
void g1_assist_v3_history_append(G1AssistV3History *, const float[2], const float[2], const float[2]);
void g1_assist_v3_history_build_observation(const G1AssistV3History *, float[150]);
/* Low-level forward requires finite inputs; returns raw network actions. */
void g1_assist_v3_policy_forward(const float[150], float[1], G1AssistV3Workspace *);
/* reset: fills history. Return 0 success, -1 invalid inputs. */
int g1_assist_v3_reset(G1AssistV3Controller *, const float[2], const float[2]);
/* stop zeroes output and latches disabled until reset; does not write hardware. */
void g1_assist_v3_stop(G1AssistV3Controller *, float[2]);
/* Exactly once per 10 ms. No vx input. Returns 0 success, 1 disabled, -1 fault.
 * On fault/disabled, output is zero and controller is latched stopped.
 * Caller sends output to hardware, handles stale samples and deadlines.
 * State/workspace must be unique to a controller, not shared across threads.
 */
int g1_assist_v3_step(G1AssistV3Controller *, const float[2], const float[2], int, float[2]);
#ifdef __cplusplus
}
#endif
#endif

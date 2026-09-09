/* Generated fixed-weight v2-v2 assist policy; C99, no heap or file I/O. */
#ifndef G1_ASSIST_V2_V2_POLICY_H
#define G1_ASSIST_V2_V2_POLICY_H
#ifdef __cplusplus
extern "C" {
#endif
#define G1_ASSIST_V2_V2_INPUT_SIZE 150
#define G1_ASSIST_V2_V2_OUTPUT_SIZE 2
#define G1_ASSIST_V2_V2_HISTORY_LENGTH 25
#define G1_ASSIST_V2_V2_POLICY_RATE_HZ 100
/* Left/right order; radians, rad/s, Nm. Target -10/+4; physical -8/+4. */
typedef struct { float hidden_256[256]; float secondary_150[150]; } G1AssistV2V2Workspace;
typedef struct {
    float position_rad[25][2], velocity_rad_s[25][2], smoothed_torque_nm[25][2];
    unsigned int next_index;
} G1AssistV2V2History;
typedef struct {
    G1AssistV2V2History history;
    G1AssistV2V2Workspace workspace;
    float observation[150], previous[2], filtered_velocity[2], output_scale;
    unsigned int ready, first;
} G1AssistV2V2Controller;
void g1_assist_v2_v2_history_reset(G1AssistV2V2History *, const float[2], const float[2]);
void g1_assist_v2_v2_history_append(G1AssistV2V2History *, const float[2], const float[2], const float[2]);
void g1_assist_v2_v2_history_build_observation(const G1AssistV2V2History *, float[150]);
/* Low-level forward requires finite inputs; returns raw network actions. */
void g1_assist_v2_v2_policy_forward(const float[150], float[2], G1AssistV2V2Workspace *);
/* reset: fills history; scale in [0,1]. Return 0 success, -1 invalid inputs. */
int g1_assist_v2_v2_reset(G1AssistV2V2Controller *, const float[2], const float[2], float);
/* stop zeroes output and latches disabled until reset; does not write hardware. */
void g1_assist_v2_v2_stop(G1AssistV2V2Controller *, float[2]);
/* Exactly once per 10 ms. No vx input. Returns 0 success, 1 disabled, -1 fault.
 * On fault/disabled, output is zero and controller is latched stopped.
 * Caller sends output to hardware, handles stale samples and deadlines.
 * State/workspace must be unique to a controller, not shared across threads.
 */
int g1_assist_v2_v2_step(G1AssistV2V2Controller *, const float[2], const float[2], int, float[2]);
#ifdef __cplusplus
}
#endif
#endif

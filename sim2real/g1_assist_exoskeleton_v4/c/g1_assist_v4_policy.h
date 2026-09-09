/* Generated fixed-weight v4 assist policy; C99, no heap or file I/O. */
#ifndef G1_ASSIST_V4_POLICY_H
#define G1_ASSIST_V4_POLICY_H
#ifdef __cplusplus
extern "C" {
#endif
#define G1_ASSIST_V4_INPUT_SIZE 400
#define G1_ASSIST_V4_OUTPUT_SIZE 1
#define G1_ASSIST_V4_HISTORY_LENGTH 25
#define G1_ASSIST_V4_POLICY_RATE_HZ 100
/* Left/right order; radians, rad/s, Nm. Paired torque [T,-T], +/-10 Nm, latched amplitude, 0.4s single pulse / 0.2s release. */
typedef struct { float hidden_256[256]; float secondary_400[400]; } G1AssistV4Workspace;
typedef struct {
    float position_rad[25][2], velocity_rad_s[25][2], smoothed_torque_nm[25][2], pulse_state[25][10];
    unsigned int next_index;
} G1AssistV4History;
typedef struct {
    G1AssistV4History history;
    G1AssistV4Workspace workspace;
    float observation[400], previous[2], filtered_velocity[2], pulse[8];
    unsigned int ready, first;
} G1AssistV4Controller;
void g1_assist_v4_history_reset(G1AssistV4History *, const float[2], const float[2]);
void g1_assist_v4_history_append(G1AssistV4History *, const float[2], const float[2], const float[2], const float[10]);
void g1_assist_v4_history_build_observation(const G1AssistV4History *, float[400]);
/* Low-level forward requires finite inputs; returns raw network actions. */
void g1_assist_v4_policy_forward(const float[400], float[1], G1AssistV4Workspace *);
/* reset: fills history. Return 0 success, -1 invalid inputs. */
int g1_assist_v4_reset(G1AssistV4Controller *, const float[2], const float[2]);
/* stop zeroes output and latches disabled until reset; does not write hardware. */
void g1_assist_v4_stop(G1AssistV4Controller *, float[2]);
/* Exactly once per 10 ms. No vx input. Returns 0 success, 1 disabled, -1 fault.
 * On fault/disabled, output is zero and controller is latched stopped.
 * Caller sends output to hardware, handles stale samples and deadlines.
 * State/workspace must be unique to a controller, not shared across threads.
 */
int g1_assist_v4_step(G1AssistV4Controller *, const float[2], const float[2], int, float[2]);
#ifdef __cplusplus
}
#endif
#endif

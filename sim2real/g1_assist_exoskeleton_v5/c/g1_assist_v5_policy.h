/* Generated fixed-weight v5 assist policy; C99, no heap or file I/O. */
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

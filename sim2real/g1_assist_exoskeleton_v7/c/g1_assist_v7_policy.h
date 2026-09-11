/* Generated fixed-weight v7 assist policy; C99, no heap or file I/O. */
#ifndef G1_ASSIST_V7_POLICY_H
#define G1_ASSIST_V7_POLICY_H
#ifdef __cplusplus
extern "C" {
#endif
#define G1_ASSIST_V7_INPUT_SIZE 450
#define G1_ASSIST_V7_OUTPUT_SIZE 2
#define G1_ASSIST_V7_HISTORY_LENGTH 25
#define G1_ASSIST_V7_POLICY_RATE_HZ 100
/* Left/right order; radians, rad/s, Nm. Paired torque [T,-T], +/-10 Nm, latched peak/duration, 0.25s rise/release, 0.6-1.4s total. */
typedef struct { float hidden_256[256]; float secondary_450[450]; } G1AssistV7Workspace;
typedef struct {
    float position_rad[25][2], velocity_rad_s[25][2], smoothed_torque_nm[25][2], pulse_state[25][12];
    unsigned int next_index;
} G1AssistV7History;
typedef struct {
    G1AssistV7History history;
    G1AssistV7Workspace workspace;
    float observation[450], previous[2], filtered_velocity[2], pulse[10];
    unsigned int ready, first;
} G1AssistV7Controller;
void g1_assist_v7_history_reset(G1AssistV7History *, const float[2], const float[2]);
void g1_assist_v7_history_append(G1AssistV7History *, const float[2], const float[2], const float[2], const float[12]);
void g1_assist_v7_history_build_observation(const G1AssistV7History *, float[450]);
/* Low-level forward requires finite inputs; returns raw network actions. */
void g1_assist_v7_policy_forward(const float[450], float[2], G1AssistV7Workspace *);
/* reset: fills history. Return 0 success, -1 invalid inputs. */
int g1_assist_v7_reset(G1AssistV7Controller *, const float[2], const float[2]);
/* stop zeroes output and latches disabled until reset; does not write hardware. */
void g1_assist_v7_stop(G1AssistV7Controller *, float[2]);
/* Exactly once per 10 ms. No vx input. Returns 0 success, 1 disabled, -1 fault.
 * On fault/disabled, output is zero and controller is latched stopped.
 * Caller sends output to hardware, handles stale samples and deadlines.
 * State/workspace must be unique to a controller, not shared across threads.
 */
int g1_assist_v7_step(G1AssistV7Controller *, const float[2], const float[2], int, float[2]);
#ifdef __cplusplus
}
#endif
#endif

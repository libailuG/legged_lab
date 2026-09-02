/* Auto-generated G1 assist-exoskeleton v2 policy interface. */
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

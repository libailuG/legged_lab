/* Offline executable example: does not open a bus or drive motors. */
#include "g1_assist_v2_v2_policy.h"
#include <stdio.h>

static G1AssistV2V2Controller controller;
int main(void)
{
    float position[2] = {-0.1f, -0.1f};
    float velocity[2] = {0.0f, 0.0f};
    float motor_nm[2] = {0.0f, 0.0f};
    /* Reset with calibrated sensor values; start with zero output scale. */
    if (g1_assist_v2_v2_reset(&controller, position, velocity, 0.0f) != 0) return 1;

    /* Hardware integration: replace this one offline tick with a 10 ms task.
     * Read current left/right rad and rad/s, call step once, then send motor_nm.
     * On stale samples / missed deadlines, stop and send zero/disable via SDK.
     * A 1 ms motor loop must hold the output between these 10 ms policy ticks.
     */
    if (g1_assist_v2_v2_step(&controller, position, velocity, 1, motor_nm) != 0) return 1;
    printf("Offline output: left=%g Nm, right=%g Nm\n", motor_nm[0], motor_nm[1]);
    g1_assist_v2_v2_stop(&controller, motor_nm);
    return 0;
}


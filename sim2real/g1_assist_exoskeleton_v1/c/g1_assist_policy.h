/* Auto-generated fixed G1 v1 assist policy interface. */
#ifndef G1_ASSIST_POLICY_H
#define G1_ASSIST_POLICY_H

#ifdef __cplusplus
extern "C" {
#endif

#define G1_ASSIST_POLICY_INPUT_SIZE 100
#define G1_ASSIST_POLICY_OUTPUT_SIZE 2
#define G1_ASSIST_POLICY_HISTORY_LENGTH 25
#define G1_ASSIST_POLICY_TORQUE_LIMIT_NM 8.0f

/*
 * ============================ 使用说明 ============================
 *
 * 1. 将 g1_assist_policy.c 和 g1_assist_policy.h 加入单片机工程。
 *    本文件已固化全部网络参数，运行时不需要 policy.pt、NPZ、Python、
 *    PyTorch或动态内存。g1_assist_policy.c使用expf()，GCC工具链通常
 *    需要链接数学库，例如：
 *
 *        gcc -std=c99 -O2 main.c g1_assist_policy.c -lm
 *
 * 2. 策略必须以50 Hz运行，即每20 ms调用一次。速度单位必须是rad/s，
 *    力矩单位是N.m。左右顺序始终为[左, 右]。
 *
 * 3. 推荐把历史、工作区和观测数组声明为static或全局变量，避免占用
 *    单片机任务栈：
 *
 *        static G1AssistPolicyHistory g_history;
 *        static G1AssistPolicyWorkspace g_workspace;
 *        static float g_observation[G1_ASSIST_POLICY_INPUT_SIZE];
 *        static float g_previous_torque_nm[2] = {0.0f, 0.0f};
 *        static unsigned int g_first_policy_step = 1U;
 *
 * 4. 机器人或控制器复位时调用：
 *
 *        void assist_controller_reset(float left_vel, float right_vel)
 *        {
 *            float initial_velocity[2] = {left_vel, right_vel};
 *            g1_assist_policy_history_reset(&g_history, initial_velocity);
 *            g_previous_torque_nm[0] = 0.0f;
 *            g_previous_torque_nm[1] = 0.0f;
 *            g_first_policy_step = 1U;
 *        }
 *
 * 5. 每20 ms执行一次以下控制流程。复位后的第一次推理直接使用reset
 *    填充的历史；从第二次推理开始，追加当前速度和上一周期指令力矩：
 *
 *        void assist_controller_step(float left_vel, float right_vel)
 *        {
 *            float velocity[2] = {left_vel, right_vel};
 *            float torque_nm[2];
 *
 *            if (g_first_policy_step != 0U) {
 *                g_first_policy_step = 0U;
 *            } else {
 *                g1_assist_policy_history_append(
 *                    &g_history, velocity, g_previous_torque_nm);
 *            }
 *
 *            g1_assist_policy_history_build_observation(
 *                &g_history, g_observation);
 *            g1_assist_policy_compute_torque(
 *                g_observation, torque_nm, &g_workspace);
 *
 *            g_previous_torque_nm[0] = torque_nm[0];
 *            g_previous_torque_nm[1] = torque_nm[1];
 *
 *            // 用实际电机接口替换下面这个示例函数。
 *            motor_set_torque_nm(torque_nm[0], torque_nm[1]);
 *        }
 *
 * 6. g1_assist_policy_compute_torque()输出已经完成动作限幅和8 N.m
 *    缩放，左右输出范围均为[-8, 8] N.m。历史中必须保存上一周期的
 *    指令力矩，而不是电机传感器测量的实际力矩。
 *
 * 7. 100维观测排列为：
 *
 *        observation[0..49]  = 25帧[左速度, 右速度]
 *        observation[50..99] = 25帧[左指令力矩, 右指令力矩]
 *
 *    两段历史均按最旧到最新排列。推荐使用本文件提供的history函数，
 *    不要手工拼接观测。
 *
 * 8. 资源占用（1 KB = 1024字节）：
 *
 *        固定float32参数：174312字节Flash，约170.23 KB
 *        推理工作区：       1424字节RAM，  约1.39 KB
 *        25帧历史：          404字节RAM，  约0.39 KB（32位平台）
 *
 *    一个工作区不能被两个中断或RTOS任务同时使用；并发推理时，
 *    每个调用方必须持有独立工作区。
 *
 * 9. g1_assist_policy_forward()只返回未经限幅的两个网络动作，通常应调用
 *    g1_assist_policy_compute_torque()取得最终N.m力矩指令。
 *
 * =================================================================
 */

/*
 * Caller-owned scratch memory. Declare it static/global on small-stack MCUs.
 * One workspace may not be shared by simultaneous calls.
 */
typedef struct {
    float hidden_256[256];
    float secondary_100[100];
} G1AssistPolicyWorkspace;

/* 25 frames of [left, right], stored internally as a circular buffer. */
typedef struct {
    float velocity[G1_ASSIST_POLICY_HISTORY_LENGTH][2];
    float commanded_torque[G1_ASSIST_POLICY_HISTORY_LENGTH][2];
    unsigned int next_index;
} G1AssistPolicyHistory;

/* Fill all velocity frames with the current velocity and all torques with zero. */
void g1_assist_policy_history_reset(
    G1AssistPolicyHistory *history,
    const float initial_velocity[2]);

/* Append current velocity and the previously commanded assist torque. */
void g1_assist_policy_history_append(
    G1AssistPolicyHistory *history,
    const float velocity[2],
    const float previous_torque_nm[2]);

/* Flatten oldest-to-newest velocity history, followed by torque history. */
void g1_assist_policy_history_build_observation(
    const G1AssistPolicyHistory *history,
    float observation[G1_ASSIST_POLICY_INPUT_SIZE]);

/* Compute the two raw policy actions. No clipping is applied. */
void g1_assist_policy_forward(
    const float observation[G1_ASSIST_POLICY_INPUT_SIZE],
    float action[G1_ASSIST_POLICY_OUTPUT_SIZE],
    G1AssistPolicyWorkspace *workspace);

/* Compute actions, clip each to [-1, 1], then multiply by 8 N.m. */
void g1_assist_policy_compute_torque(
    const float observation[G1_ASSIST_POLICY_INPUT_SIZE],
    float torque_nm[G1_ASSIST_POLICY_OUTPUT_SIZE],
    G1AssistPolicyWorkspace *workspace);

#ifdef __cplusplus
}
#endif

#endif /* G1_ASSIST_POLICY_H */

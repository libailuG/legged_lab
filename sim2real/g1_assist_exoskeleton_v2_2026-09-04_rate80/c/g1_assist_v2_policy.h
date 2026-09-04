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
#define G1_ASSIST_V2_MAX_TORQUE_DELTA_NM 0.8f
#define G1_ASSIST_V2_COMMAND_DEADZONE_M_S 0.2f
#define G1_ASSIST_V2_COMMAND_FULL_M_S 0.7f

/*
 * ========================= 移植与使用说明 =========================
 *
 * 一、文件和编译
 *
 * 1. 将 g1_assist_v2_policy.c 和 g1_assist_v2_policy.h 加入控制器工程。
 *    网络参数已经固化在.c中，运行时不需要policy.pt、NPZ、Python、
 *    PyTorch、文件系统或动态内存。
 *
 * 2. 源码使用C99和expf()。GCC示例：
 *
 *      gcc -std=c99 -O2 controller.c g1_assist_v2_policy.c -lm
 *
 *    C++工程可直接包含本头文件，接口已使用extern "C"。建议使用带单精度
 *    FPU的MCU/CPU，并在目标硬件上实测一次推理的最坏执行时间。
 *
 * 3. 固定float32网络参数约224712字节，通常应由链接脚本放入Flash/ROM。
 *    每个推理实例至少需要：Workspace 1624字节、History约604字节、
 *    observation 600字节，另加少量控制状态。建议均声明为static或全局变量，
 *    不要放在小容量任务栈中。接口不分配内存。
 *
 * 二、时序、单位和观测
 *
 * 1. 策略必须每10 ms调用一次，即100 Hz。若电机力矩内环为1 ms，应在两个
 *    策略周期之间保持最近一次motor_torque_nm，不要每1 ms重复运行策略。
 *
 * 2. 所有二维量的顺序固定为[左, 右]。角度单位是rad，角速度单位是rad/s，
 *    力矩单位是Nm。不要把degree或degree/s直接传入。
 *
 * 3. 150维观测由本文件的history函数自动构造：
 *
 *      observation[0..49]    = 25帧[左角度, 右角度]
 *      observation[50..99]   = 25帧[左角速度, 右角速度]
 *      observation[100..149] = 25帧[左平滑力矩, 右平滑力矩]
 *
 *    三段历史均按最旧到最新排列。角度和速度是外骨骼关节自身的传感量，
 *    不需要人体真实髋关节角度、速度或力矩。力矩历史必须保存上一策略周期
 *    “未乘最终output_scale”的平滑策略力矩，不是电机测得力矩，也不是最终
 *    缩放后的电机指令。
 *
 * 4. policy_step先把动作限幅到[-1,1]并换算为[-8,8] Nm，再将每10 ms
 *    最大变化限制为0.8 Nm，即80 Nm/s。apply_output_scale只在最后写电机前
 *    乘系数，因此不会直接改变policy、obs、历史或平滑过程。最终电机指令
 *    始终限幅在[-8,8] Nm。非法或负output_scale按0处理。
 *
 * 三、最小控制器示例
 *
 *      #include "g1_assist_v2_policy.h"
 *
 *      static G1AssistV2History history;
 *      static G1AssistV2Workspace workspace;
 *      static float observation[G1_ASSIST_V2_INPUT_SIZE];
 *      static float previous_smoothed_nm[2] = {0.0f, 0.0f};
 *      static unsigned int first_policy_step = 1U;
 *      static float assist_output_scale = 0.0f; // 台架测试从0开始
 *
 *      void assist_controller_reset(
 *          const float position_rad[2], const float velocity_rad_s[2])
 *      {
 *          g1_assist_v2_history_reset(
 *              &history, position_rad, velocity_rad_s);
 *          previous_smoothed_nm[0] = 0.0f;
 *          previous_smoothed_nm[1] = 0.0f;
 *          first_policy_step = 1U;
 *          motor_set_torque_nm(0.0f, 0.0f);
 *      }
 *
 *      // 由严格10 ms周期任务调用一次。
 *      void assist_controller_tick_10ms(
 *          const float position_rad[2], const float velocity_rad_s[2],
 *          float commanded_vx_m_s)
 *      {
 *          float smoothed_nm[2];
 *          float motor_nm[2];
 *
 *          // 复位后的第一次推理直接使用reset填满的25帧历史。
 *          if (first_policy_step != 0U) {
 *              first_policy_step = 0U;
 *          } else {
 *              g1_assist_v2_history_append(
 *                  &history, position_rad, velocity_rad_s,
 *                  previous_smoothed_nm);
 *          }
 *
 *          g1_assist_v2_history_build_observation(
 *              &history, observation);
 *          g1_assist_v2_policy_step(
 *              observation, commanded_vx_m_s,
 *              previous_smoothed_nm, smoothed_nm, &workspace);
 *
 *          // 历史保存未缩放的平滑策略力矩。
 *          previous_smoothed_nm[0] = smoothed_nm[0];
 *          previous_smoothed_nm[1] = smoothed_nm[1];
 *
 *          // 系数仅在最终电机输入处生效。
 *          g1_assist_v2_apply_output_scale(
 *              smoothed_nm, assist_output_scale, motor_nm);
 *          motor_set_torque_nm(motor_nm[0], motor_nm[1]);
 *      }
 *
 * 四、复位、故障和并发
 *
 * 1. 上电、使能、急停恢复、编码器重新置零或控制状态切换后，都应先调用
 *    history_reset并把previous_smoothed_nm清零，再允许策略输出。
 *
 * 2. 传感器无效、通信超时、策略任务超时或安全状态触发时，应由上层状态机
 *    立即把电机指令置零/失能；恢复前重新reset。此策略接口不替代硬件急停、
 *    电流限制、机械限位、看门狗和通信故障保护。
 *
 * 3. 一个Workspace和History只能由一个推理上下文使用。中断或RTOS多任务
 *    并发推理时，每个实例必须使用独立对象，或由互斥机制保证不重入。
 *
 * 4. 本策略仅输出左右髋部两个助力力矩。后部滑块和圆柱机构的零位PD控制
 *    需要在硬件控制层独立实现。
 *
 * 5. 首次上真实设备前，先空载检查左右方向、编码器零位、单位、100 Hz调度、
 *    力矩限幅、超时归零和急停；output_scale从0开始逐步增加。
 *
 * ================================================================
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

/* Fill all 25 position/velocity frames from current sensors and torque with zero. */
void g1_assist_v2_history_reset(
    G1AssistV2History *history,
    const float initial_position_rad[2],
    const float initial_velocity_rad_s[2]);

/* Append one 10 ms sample; torque must be previous unscaled smoothed torque. */
void g1_assist_v2_history_append(
    G1AssistV2History *history,
    const float position_rad[2],
    const float velocity_rad_s[2],
    const float previous_smoothed_torque_nm[2]);

/* Flatten the three circular histories into the exact 150-element policy input. */
void g1_assist_v2_history_build_observation(
    const G1AssistV2History *history,
    float observation[G1_ASSIST_V2_INPUT_SIZE]);

/* Return two raw network actions; no action clipping or torque conversion. */
void g1_assist_v2_policy_forward(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float action[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace);

/* Smooth training-matched assist gate: 0 at |vx|<=0.2, 1 at |vx|>=0.7 m/s. */
float g1_assist_v2_command_speed_gate(float forward_command_m_s);

/* Clip actions, apply the vx gate, and convert to target torque in [-8,8] Nm. */
void g1_assist_v2_compute_target_torque(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float forward_command_m_s,
    float target_torque_nm[G1_ASSIST_V2_OUTPUT_SIZE],
    G1AssistV2Workspace *workspace);

/* Limit each torque change to +/-0.8 Nm for one 10 ms policy cycle. */
void g1_assist_v2_slew_limit(
    const float previous_torque_nm[2],
    const float target_torque_nm[2],
    float smoothed_torque_nm[2]);

/* Run inference, target conversion and 80 Nm/s slew limiting in one call. */
void g1_assist_v2_policy_step(
    const float observation[G1_ASSIST_V2_INPUT_SIZE],
    float forward_command_m_s,
    const float previous_torque_nm[2],
    float smoothed_torque_nm[2],
    G1AssistV2Workspace *workspace);

/* Apply final-only deployment gain and clip the actual motor command to +/-8 Nm. */
void g1_assist_v2_apply_output_scale(
    const float smoothed_torque_nm[2],
    float output_scale,
    float motor_torque_nm[2]);

#ifdef __cplusplus
}
#endif
#endif

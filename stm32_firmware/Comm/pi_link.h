#ifndef PI_LINK_H
#define PI_LINK_H

#include "stm32f4xx_hal.h"
#include <stdint.h>

#define PI_FRAME_HEADER          0xA5
#define PI_FRAME_FOOTER          0x5A

/* Command codes from Pi -> STM32 */
#define PI_CMD_HEARTBEAT         0x01
#define PI_CMD_ACTION            0x02

/* Command codes from STM32 -> Pi */
#define PI_CMD_TELEMETRY_IMU     0x81
#define PI_CMD_TELEMETRY_JOINTS  0x82
#define PI_CMD_TELEMETRY_DIAG    0x83
#define PI_CMD_TELEMETRY_FAULT   0x8F

typedef struct {
    uint8_t mode_request;    /* 0=INIT, 1=STAND, 2=ACTIVE, 3=CLIMB, 4=FAULT */
    float target_velocity;   /* Target forward velocity (m/s) */
    float target_yaw_rate;   /* Target yaw rate (rad/s) */
    float target_leg_d0;     /* Target virtual leg length (m) */
    uint32_t last_heartbeat_ms;
} Pi_Command_Heartbeat_t;

typedef struct {
    float delta_torque_l;    /* Left residual torque command (N-m) */
    float delta_torque_r;    /* Right residual torque command (N-m) */
    float target_q[4];       /* Target hip joint angles (rad): [LF, RF, LB, RB] */
    uint32_t last_action_ms;
} Pi_Command_Action_t;

/* Global instance variables representing state */
extern Pi_Command_Heartbeat_t g_pi_cmd_heartbeat;
extern Pi_Command_Action_t g_pi_cmd_action;

/**
 * @brief Initializes the Pi communication bridge structures.
 */
void pi_link_init(void);

/**
 * @brief Parses an incoming raw byte stream for valid Pi link packets.
 *        Designed to be called from the USART IDLE line ISR or a background task.
 * 
 * @param buf Pointer to the raw bytes buffer.
 * @param len Length of the buffer to parse.
 * @return int Number of successfully parsed packets.
 */
int pi_link_parse_packet(const uint8_t *buf, uint16_t len);

/**
 * @brief Packages and transmits IMU telemetry (Roll, Pitch, Yaw, Gyro) to the Pi.
 * 
 * @param huart USART handle.
 * @param roll Pitch/Roll Euler angles in radians.
 * @param pitch 
 * @param yaw 
 * @param gx Gyro rates in rad/s.
 * @param gy 
 * @param gz 
 * @return int 0 on success, -1 on failure.
 */
int pi_link_send_imu(UART_HandleTypeDef *huart, float roll, float pitch, float yaw, float gx, float gy, float gz);

/**
 * @brief Packages and transmits joint state telemetry to the Pi.
 * 
 * @param huart USART handle.
 * @param wheel_l_pos Left wheel angle (rad).
 * @param wheel_l_vel Left wheel velocity (rad/s).
 * @param wheel_l_tau Left wheel feedback torque (N-m).
 * @param wheel_r_pos Right wheel angle (rad).
 * @param wheel_r_vel Right wheel velocity (rad/s).
 * @param wheel_r_tau Right wheel feedback torque (N-m).
 * @param servo_pos Array of 4 servo angles (rad): [LF, RF, LB, RB].
 * @param servo_vel Array of 4 servo velocities (rad/s).
 * @param servo_cur Array of 4 servo currents (A).
 * @return int 0 on success, -1 on failure.
 */
int pi_link_send_joints(UART_HandleTypeDef *huart,
                        float wheel_l_pos, float wheel_l_vel, float wheel_l_tau,
                        float wheel_r_pos, float wheel_r_vel, float wheel_r_tau,
                        const float *servo_pos, const float *servo_vel, const float *servo_cur);

/**
 * @brief Packages and transmits diagnostics telemetry to the Pi.
 * 
 * @param huart USART handle.
 * @param battery_mv Battery voltage in millivolts.
 * @param max_temp_c Maximum detected temperature across servos/sensors.
 * @param error_mask Bitmask of diagnostic errors.
 * @return int 0 on success, -1 on failure.
 */
int pi_link_send_diag(UART_HandleTypeDef *huart, uint16_t battery_mv, uint8_t max_temp_c, uint8_t error_mask);

/**
 * @brief Packages and transmits a critical fault packet to the Pi.
 * 
 * @param huart USART handle.
 * @param fault_code Code representing the active shutdown fault.
 * @return int 0 on success, -1 on failure.
 */
int pi_link_send_fault(UART_HandleTypeDef *huart, uint8_t fault_code);

#endif /* PI_LINK_H */

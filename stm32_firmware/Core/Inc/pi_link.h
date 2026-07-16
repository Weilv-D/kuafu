#ifndef PI_LINK_H
#define PI_LINK_H

#include "stm32f4xx_hal.h"
#include <stdint.h>

#define PI_FRAME_HEADER          0xA5
#define PI_FRAME_FOOTER          0x5A
#define PI_PROTOCOL_VERSION       1
#define PI_MAX_PAYLOAD            64

/* Command codes from Pi -> STM32 */
#define PI_CMD_HEARTBEAT         0x01
#define PI_CMD_ACTION            0x02
#define PI_CMD_HELLO             0x03

/* Command codes from STM32 -> Pi */
#define PI_CMD_TELEMETRY_IMU     0x81
#define PI_CMD_TELEMETRY_JOINTS  0x82
#define PI_CMD_TELEMETRY_DIAG    0x83
#define PI_CMD_TELEMETRY_HEALTH  0x84
#define PI_CMD_TELEMETRY_FAULT   0x8F
#define PI_HEALTH_PAYLOAD_SIZE     46U

typedef struct {
    uint32_t fault_mask;
    uint8_t mode;
    uint8_t reset_cause;
    uint16_t imu_age_ms;
    uint16_t wheel_l_age_ms;
    uint16_t wheel_r_age_ms;
    uint16_t servo_age_ms[4];
    uint16_t imu_errors;
    uint16_t wheel_l_errors;
    uint16_t wheel_r_errors;
    uint16_t servo_errors[4];
    /* DDSM wheel error breakdown (timeout / checksum / protocol) for diagnostics. */
    uint16_t wheel_l_timeout_errors;
    uint16_t wheel_l_checksum_errors;
    uint16_t wheel_l_protocol_errors;
    uint16_t wheel_r_timeout_errors;
    uint16_t wheel_r_checksum_errors;
    uint16_t wheel_r_protocol_errors;
} Pi_HealthTelemetry_t;

typedef struct {
    uint8_t mode_request;    /* 0=INIT, 1=STAND, 2=ACTIVE, 3=CLIMB, 4=FAULT */
    float target_velocity;   /* Target forward velocity (m/s) */
    float target_yaw_rate;   /* Target yaw rate (rad/s) */
    float target_leg_d0;     /* Target virtual leg length (m) */
    uint32_t last_heartbeat_ms;
} Pi_Command_Heartbeat_t;

typedef struct {
    float delta_torque_common; /* Common wheel residual, normalized [-1,1] */
    float delta_torque_yaw;    /* Yaw wheel residual, normalized [-1,1] */
    float qx_l;                /* Left foot Qx residual, normalized [-1,1] */
    float d0_l;                /* Left D0 residual, normalized [-1,1] */
    float qx_r;                /* Right foot Qx residual, normalized [-1,1] */
    float d0_r;                /* Right D0 residual, normalized [-1,1] */
    uint32_t last_action_ms;
} Pi_Command_Action_t;

/* Global instance variables representing state */
extern Pi_Command_Heartbeat_t g_pi_cmd_heartbeat;
extern Pi_Command_Action_t g_pi_cmd_action;

uint8_t pi_link_is_compatible(void);
uint8_t pi_link_heartbeat_fresh(void);
uint8_t pi_link_action_fresh(void);

/**
 * @brief Initializes the Pi communication bridge structures.
 */
void pi_link_init(void);

/* Freshness fallbacks are intentionally separate: stale action removes only the
 * learned residual, while stale heartbeat also zeros high-level velocity/yaw and
 * leaves the 250Hz position-hold baseline active. */
void pi_link_clear_action(void);
void pi_link_enter_hold(void);

/**
 * @brief Feeds arbitrary UART byte chunks into the versioned streaming decoder.
 *        Frame: A5 | version | type | length | seq:u16 | timestamp_ms:u32 |
 *        payload | CRC8/MAXIM | 5A. Partial DMA-IDLE chunks are retained.
 * 
 * @param buf Pointer to the raw bytes buffer.
 * @param len Length of the buffer to parse.
 * @return int Number of successfully parsed packets.
 */
int pi_link_parse_packet(const uint8_t *buf, uint16_t len);
void pi_link_on_tx_complete(UART_HandleTypeDef *huart);
void pi_link_on_tx_error(UART_HandleTypeDef *huart);

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
 * @param battery_mv Battery voltage in millivolts; zero means unavailable/not fitted.
 * @param max_temp_c Maximum detected temperature across servos/sensors.
 * @param error_mask Bitmask of diagnostic errors.
 * @return int 0 on success, -1 on failure.
 */
int pi_link_send_diag(UART_HandleTypeDef *huart, uint16_t battery_mv, uint8_t max_temp_c, uint8_t error_mask);
void pi_link_encode_health_payload(uint8_t payload[PI_HEALTH_PAYLOAD_SIZE],
                                   const Pi_HealthTelemetry_t *health);
int pi_link_send_health(UART_HandleTypeDef *huart,
                        const Pi_HealthTelemetry_t *health);

/**
 * @brief Packages and transmits a critical fault packet to the Pi.
 * 
 * @param huart USART handle.
 * @param fault_code Code representing the active shutdown fault.
 * @return int 0 on success, -1 on failure.
 */
int pi_link_send_fault(UART_HandleTypeDef *huart, uint8_t fault_code);

#endif /* PI_LINK_H */

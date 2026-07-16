#ifndef DDSM315_H
#define DDSM315_H

#include "stm32f4xx_hal.h"
#include "device_health.h"
#include <stdint.h>

#define DDSM_MODE_CURRENT        1
#define DDSM_MODE_SPEED          2
#define DDSM_MODE_POSITION       3
#define DDSM_MODE_DISABLE        9

typedef struct {
    uint8_t id;
    uint8_t mode;
    float torque;            /* Measured feedback torque (N-m) */
    float velocity_rads;     /* Measured feedback velocity (rad/s) */
    float position_rad;      /* Measured feedback single-turn angle (rad, [0, 2pi]) */
    uint8_t error_code;
    DeviceHealth_t health;
} DDSM_State_t;

/**
 * @brief Sends a command to set the motor torque (current loop).
 * 
 * @param huart USART handle to send the packet.
 * @param id Motor ID.
 * @param torque_nm Target torque in N-m (clamped to [-1.1, 1.1]).
 * @return int 0 on success, -1 on failure.
 */
int ddsm_set_torque(UART_HandleTypeDef *huart, uint8_t id, float torque_nm);

/**
 * @brief Sends a command to set the motor speed (speed loop).
 * 
 * @param huart USART handle to send the packet.
 * @param id Motor ID.
 * @param rpm Target speed in RPM.
 * @return int 0 on success, -1 on failure.
 */
int ddsm_set_speed(UART_HandleTypeDef *huart, uint8_t id, float rpm);

/**
 * @brief Sends enable/disable command to the motor.
 * 
 * @param huart USART handle.
 * @param id Motor ID.
 * @param enable 1 to enable, 0 to disable.
 * @return int 0 on success, -1 on failure.
 */
int ddsm_set_enable(UART_HandleTypeDef *huart, uint8_t id, uint8_t enable);

/**
 * @brief Changes the motor control mode.
 * 
 * @param huart USART handle.
 * @param id Motor ID.
 * @param mode Target mode (DDSM_MODE_CURRENT, DDSM_MODE_SPEED, DDSM_MODE_POSITION).
 * @return int 0 on success, -1 on failure.
 */
int ddsm_set_mode(UART_HandleTypeDef *huart, uint8_t id, uint8_t mode);

/**
 * @brief Parses a received 10-byte feedback frame and updates motor state.
 * 
 * @param rx_buf Pointer to 10-byte buffer.
 * @param state Out: State structure to update.
 * @return int 0 on success, -1 on CRC/format failure, -2 on wrong ID.
 */
int ddsm_parse_feedback(const uint8_t *rx_buf, DDSM_State_t *state);

#endif /* DDSM315_H */

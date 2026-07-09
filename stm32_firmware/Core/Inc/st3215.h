#ifndef ST3215_H
#define ST3215_H

#include "stm32f4xx_hal.h"
#include <stdint.h>

#define ST_INST_PING             0x01
#define ST_INST_READ             0x02
#define ST_INST_WRITE            0x03
#define ST_INST_REG_WRITE        0x04
#define ST_INST_REG_ACTION       0x05
#define ST_INST_SYNC_READ        0x82
#define ST_INST_SYNC_WRITE       0x83

#define ST_REG_ID                5
#define ST_REG_MODE              33
#define ST_REG_TORQUE_ENABLE     40
#define ST_REG_ACC               41
#define ST_REG_GOAL_POSITION_L   42
#define ST_REG_GOAL_TIME_L       44
#define ST_REG_GOAL_SPEED_L      46
#define ST_REG_PRESENT_POSITION_L 56
#define ST_REG_PRESENT_SPEED_L   58
#define ST_REG_PRESENT_LOAD_L    60
#define ST_REG_PRESENT_VOLTAGE   62
#define ST_REG_PRESENT_TEMPERATURE 63
#define ST_REG_PRESENT_CURRENT_L 69

typedef struct {
    uint8_t id;
    float position_rad;      /* Present position in radians */
    float velocity_rads;     /* Present velocity in rad/s */
    float load;              /* Present load/torque ratio */
    float temperature_c;     /* Present temperature in Celsius */
    float voltage;           /* Present voltage in Volts */
    float current_a;         /* Present current in Amperes */
    uint8_t is_online;       /* 1 if online, 0 if offline */
    uint8_t consecutive_failures;
} ST3215_State_t;

/**
 * @brief Sends a SyncWrite command to set the targets of multiple servos simultaneously.
 * 
 * @param huart USART handle.
 * @param ids Array of servo IDs.
 * @param count Number of servos in the array.
 * @param positions_ticks Array of target positions in ticks (0 to 4095).
 * @param speeds_ticks Array of target speeds in ticks (0 to 4000).
 * @param accels Array of target accelerations (0 to 254).
 * @return int 0 on success, -1 on failure.
 */
int st3215_sync_write_pos(UART_HandleTypeDef *huart,
                          const uint8_t *ids,
                          uint8_t count,
                          const int16_t *positions_ticks,
                          const uint16_t *speeds_ticks,
                          const uint8_t *accels);

/**
 * @brief Reads present state data from a single servo and updates the state structure.
 * 
 * @param huart USART handle.
 * @param id Servo ID.
 * @param state Out: State structure to update.
 * @return int 0 on success, -1 on timeout/communication error, -2 on CRC mismatch.
 */
int st3215_read_state(UART_HandleTypeDef *huart, uint8_t id, ST3215_State_t *state);

/**
 * @brief Enables or disables torque for a single servo.
 * 
 * @param huart USART handle.
 * @param id Servo ID.
 * @param enable 1 to enable, 0 to disable.
 * @return int 0 on success, -1 on failure.
 */
int st3215_set_torque_enable(UART_HandleTypeDef *huart, uint8_t id, uint8_t enable);

#endif /* ST3215_H */

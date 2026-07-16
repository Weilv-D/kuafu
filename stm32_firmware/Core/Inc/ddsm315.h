#ifndef DDSM315_H
#define DDSM315_H

#include "stm32f4xx_hal.h"
#include "device_health.h"
#include <stdint.h>

#define DDSM_MODE_CURRENT        1U
#define DDSM_MODE_SPEED          2U
#define DDSM_MODE_POSITION       3U
#define DDSM_MODE_DISABLE        9U
#define DDSM_FRAME_SIZE         10U
#define DDSM_TRANSACTION_TIMEOUT_MS 3U

typedef struct {
    uint8_t id;
    uint8_t mode;
    float torque;
    float velocity_rads;
    float position_rad;
    uint8_t error_code;
    DeviceHealth_t health;
} DDSM_State_t;

typedef enum {
    DDSM_BUS_IDLE = 0,
    DDSM_BUS_TX = 1,
    DDSM_BUS_RX = 2
} DDSM_BusPhase_t;

typedef struct {
    UART_HandleTypeDef *huart;
    DDSM_State_t *target;
    uint8_t tx[DDSM_FRAME_SIZE];
    uint8_t rx[DDSM_FRAME_SIZE];
    uint32_t deadline_ms;
    DDSM_BusPhase_t phase;
} DDSM_Bus_t;

void ddsm_build_torque(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float torque_nm);
void ddsm_build_speed(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float rpm);
void ddsm_build_enable(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t enable);
void ddsm_build_mode(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t mode);
void ddsm_build_query(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id);
int ddsm_parse_feedback(const uint8_t packet[DDSM_FRAME_SIZE], DDSM_State_t *state);

void ddsm_bus_init(DDSM_Bus_t *bus, UART_HandleTypeDef *huart);
uint8_t ddsm_bus_is_idle(const DDSM_Bus_t *bus);
int ddsm_bus_submit(DDSM_Bus_t *bus,
                    DDSM_State_t *target,
                    const uint8_t packet[DDSM_FRAME_SIZE],
                    uint32_t now_ms);
int ddsm_bus_queue_torque(DDSM_Bus_t *bus, DDSM_State_t *target,
                          float torque_nm, uint32_t now_ms);
int ddsm_bus_queue_enable(DDSM_Bus_t *bus, DDSM_State_t *target,
                          uint8_t enable, uint32_t now_ms);
int ddsm_bus_queue_mode(DDSM_Bus_t *bus, DDSM_State_t *target,
                        uint8_t mode, uint32_t now_ms);
int ddsm_bus_queue_query(DDSM_Bus_t *bus, DDSM_State_t *target,
                         uint32_t now_ms);
void ddsm_bus_step(DDSM_Bus_t *bus, uint32_t now_ms);
void ddsm_bus_on_tx_complete(DDSM_Bus_t *bus);
void ddsm_bus_on_rx_complete(DDSM_Bus_t *bus, uint32_t now_ms);
void ddsm_bus_on_uart_error(DDSM_Bus_t *bus);

#endif

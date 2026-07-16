#ifndef ST3215_H
#define ST3215_H

#include "stm32f4xx_hal.h"
#include "device_health.h"
#include <stdint.h>

#define ST_INST_READ               0x02U
#define ST_INST_WRITE              0x03U
#define ST_INST_SYNC_WRITE         0x83U
#define ST_REG_TORQUE_ENABLE       40U
#define ST_REG_ACC                 41U
#define ST_REG_PRESENT_POSITION_L  56U
#define ST_STATE_DATA_SIZE         15U
#define ST_STATE_FRAME_SIZE        21U
#define ST_MAX_PACKET_SIZE         96U
#define ST_REPLY_TIMEOUT_MS         3U

typedef struct {
    uint8_t id;
    uint16_t position_tick;
    float position_rad;      /* Raw actuator coordinate, not the shared joint frame. */
    float velocity_rads;
    float load;
    float temperature_c;
    float voltage;
    float current_a;
    DeviceHealth_t health;
} ST3215_State_t;

typedef enum {
    ST_BUS_IDLE = 0,
    ST_BUS_TX_ONLY = 1,
    ST_BUS_TX_READ = 2,
    ST_BUS_WAIT_REPLY = 3
} ST3215_BusPhase_t;

typedef struct {
    UART_HandleTypeDef *huart;
    ST3215_State_t *target;
    uint8_t tx[ST_MAX_PACKET_SIZE];
    uint8_t tx_len;
    uint8_t echo_candidate[ST_MAX_PACKET_SIZE];
    uint8_t echo_len;
    uint8_t frame[ST_MAX_PACKET_SIZE];
    uint8_t frame_len;
    uint8_t expected_frame_len;
    uint8_t rx_byte;
    uint8_t offline_after;
    uint32_t deadline_ms;
    ST3215_BusPhase_t phase;
} ST3215_Bus_t;

int st3215_build_sync_write(uint8_t *packet, uint8_t capacity,
                            const uint8_t *ids, uint8_t count,
                            const int16_t *positions_ticks,
                            const uint16_t *speeds_ticks,
                            const uint8_t *accels);
int st3215_build_torque(uint8_t packet[8], uint8_t id, uint8_t enable);
int st3215_build_read_state(uint8_t packet[8], uint8_t id);
int st3215_parse_state_frame(const uint8_t *frame, uint8_t frame_len,
                             ST3215_State_t *state);

void st3215_bus_init(ST3215_Bus_t *bus, UART_HandleTypeDef *huart);
uint8_t st3215_bus_is_idle(const ST3215_Bus_t *bus);
int st3215_bus_queue_sync_write(ST3215_Bus_t *bus,
                                const uint8_t *ids, uint8_t count,
                                const int16_t *positions_ticks,
                                const uint16_t *speeds_ticks,
                                const uint8_t *accels);
int st3215_bus_queue_torque(ST3215_Bus_t *bus, uint8_t id, uint8_t enable);
int st3215_bus_queue_read(ST3215_Bus_t *bus, ST3215_State_t *target,
                          uint8_t offline_after, uint32_t now_ms);
void st3215_bus_step(ST3215_Bus_t *bus, uint32_t now_ms);
void st3215_bus_on_tx_complete(ST3215_Bus_t *bus);
void st3215_bus_on_rx_byte(ST3215_Bus_t *bus, uint32_t now_ms);
void st3215_bus_on_uart_error(ST3215_Bus_t *bus);

#endif

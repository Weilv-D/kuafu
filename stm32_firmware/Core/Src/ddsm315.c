#include "ddsm315.h"
#include "crc8.h"
#include "pin_config.h"

#include <stddef.h>
#include <string.h>

#define RPM_TO_RADS (2.0f * 3.14159265f / 60.0f)
#define POS_TO_RAD  (2.0f * 3.14159265f / 32768.0f)
#define DDSM_OFFLINE_AFTER 3U

static uint8_t deadline_reached(uint32_t now_ms, uint32_t deadline_ms) {
    return (uint8_t)((int32_t)(now_ms - deadline_ms) >= 0);
}

static void finish_failure(DDSM_Bus_t *bus, DeviceFailure_t failure) {
    if (bus == NULL) return;
    if (bus->target != NULL) {
        device_health_mark_failure(&bus->target->health, failure, DDSM_OFFLINE_AFTER);
    }
    bus->target = NULL;
    bus->phase = DDSM_BUS_IDLE;
}

static void arm_rx(DDSM_Bus_t *bus) {
    if (bus != NULL && bus->huart != NULL) {
        (void)HAL_UART_Receive_IT(bus->huart, &bus->rx_byte, 1U);
    }
}

void ddsm_build_torque(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float torque_nm) {
    int16_t raw;
    if (torque_nm > DDSM_MAX_TORQUE_NM) torque_nm = DDSM_MAX_TORQUE_NM;
    if (torque_nm < -DDSM_MAX_TORQUE_NM) torque_nm = -DDSM_MAX_TORQUE_NM;
    raw = (int16_t)(torque_nm * DDSM_TORQUE_TO_RAW);
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0x64U;
    packet[2] = (uint8_t)((uint16_t)raw >> 8);
    packet[3] = (uint8_t)raw;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_speed(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, float rpm) {
    int16_t raw = (int16_t)(rpm * 10.0f);
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0x64U;
    packet[2] = (uint8_t)((uint16_t)raw >> 8);
    packet[3] = (uint8_t)raw;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_enable(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t enable) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0xA0U;
    packet[2] = enable ? 0x08U : 0x09U;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_mode(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id, uint8_t mode) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0xA0U;
    packet[9] = mode;
}

void ddsm_build_query(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = id;
    packet[1] = 0x74U;
    packet[9] = crc8_calculate(packet, 9U);
}

void ddsm_build_set_id(uint8_t packet[DDSM_FRAME_SIZE], uint8_t id) {
    memset(packet, 0, DDSM_FRAME_SIZE);
    packet[0] = 0xAAU;
    packet[1] = 0x55U;
    packet[2] = 0x53U;
    packet[3] = id;
    packet[9] = crc8_calculate(packet, 9U);
}

int ddsm_parse_feedback(const uint8_t packet[DDSM_FRAME_SIZE], DDSM_State_t *state) {
    int16_t raw_current;
    int16_t raw_speed;
    uint16_t raw_position;
    if (packet == NULL || state == NULL) return -3;
    if (crc8_calculate(packet, 9U) != packet[9]) return -1;
    if (packet[0] != state->id) return -2;
    state->mode = packet[1];
    raw_current = (int16_t)(((uint16_t)packet[2] << 8) | packet[3]);
    raw_speed = (int16_t)(((uint16_t)packet[4] << 8) | packet[5]);
    raw_position = (uint16_t)(((uint16_t)packet[6] << 8) | packet[7]);
    state->torque = (float)raw_current * DDSM_RAW_TO_TORQUE;
    state->velocity_rads = (float)raw_speed * RPM_TO_RADS;
    state->position_rad = (float)raw_position * POS_TO_RAD;
    state->error_code = packet[8];
    return 0;
}

void ddsm_bus_init(DDSM_Bus_t *bus, UART_HandleTypeDef *huart) {
    if (bus == NULL) return;
    memset(bus, 0, sizeof(*bus));
    bus->huart = huart;
    bus->phase = DDSM_BUS_IDLE;
    arm_rx(bus);
}

uint8_t ddsm_bus_is_idle(const DDSM_Bus_t *bus) {
    return (uint8_t)(bus != NULL && bus->phase == DDSM_BUS_IDLE);
}

int ddsm_bus_submit(DDSM_Bus_t *bus,
                    DDSM_State_t *target,
                    const uint8_t packet[DDSM_FRAME_SIZE],
                    uint32_t now_ms) {
    if (bus == NULL || target == NULL || packet == NULL || bus->huart == NULL) return -1;
    if (bus->phase != DDSM_BUS_IDLE) return -2;
    memcpy(bus->tx, packet, DDSM_FRAME_SIZE);
    bus->target = target;
    bus->deadline_ms = now_ms + DDSM_TRANSACTION_TIMEOUT_MS;
    bus->phase = DDSM_BUS_TX;
    bus->rx_len = 0U;
    if (HAL_UART_Transmit_IT(bus->huart, bus->tx, DDSM_FRAME_SIZE) != HAL_OK) {
        finish_failure(bus, DEVICE_FAILURE_PROTOCOL);
        return -3;
    }
    return 0;
}

int ddsm_bus_queue_torque(DDSM_Bus_t *bus, DDSM_State_t *target,
                          float torque_nm, uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_torque(packet, target->id, torque_nm);
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

int ddsm_bus_queue_enable(DDSM_Bus_t *bus, DDSM_State_t *target,
                          uint8_t enable, uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_enable(packet, target->id, enable);
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

int ddsm_bus_queue_mode(DDSM_Bus_t *bus, DDSM_State_t *target,
                        uint8_t mode, uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_mode(packet, target->id, mode);
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

int ddsm_bus_queue_query(DDSM_Bus_t *bus, DDSM_State_t *target,
                         uint32_t now_ms) {
    uint8_t packet[DDSM_FRAME_SIZE];
    if (target == NULL) return -1;
    ddsm_build_query(packet, target->id);
    return ddsm_bus_submit(bus, target, packet, now_ms);
}

void ddsm_bus_step(DDSM_Bus_t *bus, uint32_t now_ms) {
    if (bus == NULL || bus->phase == DDSM_BUS_IDLE) return;
    if (deadline_reached(now_ms, bus->deadline_ms)) {
        finish_failure(bus, DEVICE_FAILURE_TIMEOUT);
        bus->rx_len = 0U;
    }
}

void ddsm_bus_on_tx_complete(DDSM_Bus_t *bus) {
    if (bus == NULL || bus->phase != DDSM_BUS_TX) return;
    bus->phase = DDSM_BUS_RX;
}

void ddsm_bus_on_rx_byte(DDSM_Bus_t *bus, uint32_t now_ms) {
    int result;
    if (bus == NULL) return;
    if (bus->rx_len < DDSM_FRAME_SIZE) {
        bus->rx[bus->rx_len++] = bus->rx_byte;
    }
    if (bus->rx_len == DDSM_FRAME_SIZE) {
        if (memcmp(bus->rx, bus->tx, DDSM_FRAME_SIZE) == 0) {
            bus->rx_len = 0U;
        } else if ((bus->phase == DDSM_BUS_TX || bus->phase == DDSM_BUS_RX) &&
                   bus->target != NULL && bus->rx[0] == bus->target->id &&
                   crc8_calculate(bus->rx, 9U) == bus->rx[9]) {
            result = ddsm_parse_feedback(bus->rx, bus->target);
            if (result == 0) {
                device_health_mark_valid(&bus->target->health, now_ms);
                bus->target = NULL;
                bus->phase = DDSM_BUS_IDLE;
                bus->rx_len = 0U;
            } else {
                finish_failure(bus, result == -1 ? DEVICE_FAILURE_CHECKSUM
                                                 : DEVICE_FAILURE_PROTOCOL);
                bus->rx_len = 0U;
            }
        } else if ((bus->phase == DDSM_BUS_TX || bus->phase == DDSM_BUS_RX) &&
                   bus->target != NULL && bus->rx[0] == bus->target->id) {
            /* An auto-direction adapter can leave a partial echo directly in
             * front of the motor reply. A target-ID byte at the start of one
             * invalid window is therefore not a transaction boundary. Record
             * the bad candidate and keep sliding until a complete frame or
             * the bounded transaction deadline is reached. */
            device_health_mark_failure(&bus->target->health,
                                       DEVICE_FAILURE_CHECKSUM,
                                       DDSM_OFFLINE_AFTER);
            memmove(bus->rx, &bus->rx[1], DDSM_FRAME_SIZE - 1U);
            bus->rx_len = DDSM_FRAME_SIZE - 1U;
        } else {
            memmove(bus->rx, &bus->rx[1], DDSM_FRAME_SIZE - 1U);
            bus->rx_len = DDSM_FRAME_SIZE - 1U;
        }
    }
    arm_rx(bus);
}

void ddsm_bus_on_uart_error(DDSM_Bus_t *bus) {
    if (bus == NULL) return;
    if (bus->phase != DDSM_BUS_IDLE) {
        finish_failure(bus, DEVICE_FAILURE_PROTOCOL);
    }
    bus->rx_len = 0U;
    arm_rx(bus);
}

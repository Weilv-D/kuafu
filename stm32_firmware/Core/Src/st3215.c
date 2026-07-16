#include "st3215.h"

#include <stddef.h>
#include <string.h>

#define TICK_TO_RAD ((2.0f * 3.14159265f) / 4096.0f)

static uint8_t checksum(const uint8_t *body, uint8_t size) {
    uint8_t sum = 0U;
    uint8_t i;
    for (i = 0U; i < size; ++i) sum = (uint8_t)(sum + body[i]);
    return (uint8_t)~sum;
}

int st3215_build_sync_write(uint8_t *packet, uint8_t capacity,
                            const uint8_t *ids, uint8_t count,
                            const int16_t *positions_ticks,
                            const uint16_t *speeds_ticks,
                            const uint8_t *accels) {
    uint8_t i;
    uint8_t total;
    if (packet == NULL || ids == NULL || positions_ticks == NULL ||
        speeds_ticks == NULL || accels == NULL || count == 0U || count > 10U) return -1;
    total = (uint8_t)(8U * count + 8U);
    if (capacity < total) return -2;
    packet[0] = 0xFFU; packet[1] = 0xFFU; packet[2] = 0xFEU;
    packet[3] = (uint8_t)(8U * count + 4U);
    packet[4] = ST_INST_SYNC_WRITE;
    packet[5] = ST_REG_ACC;
    packet[6] = 7U;
    for (i = 0U; i < count; ++i) {
        int32_t clamped = positions_ticks[i];
        uint16_t pos;
        uint8_t p = (uint8_t)(7U + i * 8U);
        if (clamped < 0) clamped = 0;
        if (clamped > 4095) clamped = 4095;
        pos = (uint16_t)clamped;
        packet[p] = ids[i]; packet[p + 1U] = accels[i];
        packet[p + 2U] = (uint8_t)pos; packet[p + 3U] = (uint8_t)(pos >> 8);
        packet[p + 4U] = 0U; packet[p + 5U] = 0U;
        packet[p + 6U] = (uint8_t)speeds_ticks[i];
        packet[p + 7U] = (uint8_t)(speeds_ticks[i] >> 8);
    }
    packet[total - 1U] = checksum(&packet[2], (uint8_t)(total - 3U));
    return total;
}

int st3215_build_torque(uint8_t packet[8], uint8_t id, uint8_t enable) {
    if (packet == NULL) return -1;
    packet[0] = 0xFFU; packet[1] = 0xFFU; packet[2] = id; packet[3] = 4U;
    packet[4] = ST_INST_WRITE; packet[5] = ST_REG_TORQUE_ENABLE;
    packet[6] = enable ? 1U : 0U;
    packet[7] = checksum(&packet[2], 5U);
    return 8;
}

int st3215_build_read_state(uint8_t packet[8], uint8_t id) {
    if (packet == NULL) return -1;
    packet[0] = 0xFFU; packet[1] = 0xFFU; packet[2] = id; packet[3] = 4U;
    packet[4] = ST_INST_READ; packet[5] = ST_REG_PRESENT_POSITION_L;
    packet[6] = ST_STATE_DATA_SIZE;
    packet[7] = checksum(&packet[2], 5U);
    return 8;
}

int st3215_parse_state_frame(const uint8_t *frame, uint8_t frame_len,
                             ST3215_State_t *state) {
    uint16_t value;
    int16_t signed_value;
    if (frame == NULL || state == NULL || frame_len != ST_STATE_FRAME_SIZE) return -1;
    if (frame[0] != 0xFFU || frame[1] != 0xFFU || frame[3] != 17U) return -1;
    if (frame[2] != state->id) return -2;
    if (checksum(&frame[2], (uint8_t)(frame_len - 3U)) != frame[frame_len - 1U]) return -3;
    value = (uint16_t)(((uint16_t)frame[6] << 8) | frame[5]);
    state->position_tick = value;
    state->position_rad = (float)value * TICK_TO_RAD;
    value = (uint16_t)(((uint16_t)frame[8] << 8) | frame[7]);
    signed_value = (int16_t)value;
    state->velocity_rads = (float)signed_value * TICK_TO_RAD;
    value = (uint16_t)(((uint16_t)frame[10] << 8) | frame[9]);
    signed_value = (value & 0x0400U) ? -(int16_t)(value & 0x03FFU) : (int16_t)(value & 0x03FFU);
    state->load = (float)signed_value / 1000.0f;
    state->voltage = (float)frame[11] * 0.1f;
    state->temperature_c = (float)frame[12];
    value = (uint16_t)(((uint16_t)frame[19] << 8) | frame[18]);
    state->current_a = (float)(int16_t)value * 0.0065f;
    return 0;
}

static void arm_rx(ST3215_Bus_t *bus) {
    if (bus != NULL && bus->huart != NULL) {
        (void)HAL_UART_Receive_IT(bus->huart, &bus->rx_byte, 1U);
    }
}

static void reset_parser(ST3215_Bus_t *bus) {
    bus->frame_len = 0U;
    bus->expected_frame_len = 0U;
}

static void finish_read_failure(ST3215_Bus_t *bus, DeviceFailure_t failure) {
    if (bus->target != NULL) {
        device_health_mark_failure(&bus->target->health, failure, bus->offline_after);
    }
    bus->target = NULL;
    bus->phase = ST_BUS_IDLE;
    reset_parser(bus);
}

static void accept_stream_byte(ST3215_Bus_t *bus, uint8_t byte, uint32_t now_ms) {
    int result;
    if (bus->frame_len == 0U) {
        if (byte == 0xFFU) bus->frame[bus->frame_len++] = byte;
        return;
    }
    if (bus->frame_len == 1U) {
        if (byte == 0xFFU) bus->frame[bus->frame_len++] = byte;
        else bus->frame_len = 0U;
        return;
    }
    if (bus->frame_len >= ST_MAX_PACKET_SIZE) {
        reset_parser(bus);
        return;
    }
    bus->frame[bus->frame_len++] = byte;
    if (bus->frame_len == 4U) {
        uint16_t expected = (uint16_t)bus->frame[3] + 4U;
        if (expected < 6U || expected > ST_MAX_PACKET_SIZE) {
            reset_parser(bus);
            if (byte == 0xFFU) bus->frame[bus->frame_len++] = byte;
            return;
        }
        bus->expected_frame_len = (uint8_t)expected;
    }
    if (bus->expected_frame_len != 0U && bus->frame_len == bus->expected_frame_len) {
        if ((bus->phase == ST_BUS_TX_READ || bus->phase == ST_BUS_WAIT_REPLY) &&
            bus->target != NULL && bus->frame[2] == bus->target->id) {
            result = st3215_parse_state_frame(bus->frame, bus->frame_len, bus->target);
            if (result == 0) {
                device_health_mark_valid(&bus->target->health, now_ms);
                bus->target = NULL;
                bus->phase = ST_BUS_IDLE;
            } else {
                finish_read_failure(bus, result == -3 ? DEVICE_FAILURE_CHECKSUM
                                                      : DEVICE_FAILURE_PROTOCOL);
            }
        }
        reset_parser(bus);
    }
}

static void consume_with_echo_filter(ST3215_Bus_t *bus, uint8_t byte, uint32_t now_ms) {
    uint8_t i;
    if (bus->echo_len < bus->tx_len && byte == bus->tx[bus->echo_len]) {
        bus->echo_candidate[bus->echo_len++] = byte;
        if (bus->echo_len == bus->tx_len) bus->echo_len = 0U;
        return;
    }
    for (i = 0U; i < bus->echo_len; ++i) {
        accept_stream_byte(bus, bus->echo_candidate[i], now_ms);
    }
    bus->echo_len = 0U;
    accept_stream_byte(bus, byte, now_ms);
}

static int start_tx(ST3215_Bus_t *bus, uint8_t len, ST3215_BusPhase_t phase) {
    if (bus == NULL || bus->huart == NULL || bus->phase != ST_BUS_IDLE) return -1;
    bus->tx_len = len;
    bus->echo_len = 0U;
    bus->phase = phase;
    if (HAL_UART_Transmit_IT(bus->huart, bus->tx, len) != HAL_OK) {
        bus->phase = ST_BUS_IDLE;
        return -2;
    }
    return 0;
}

void st3215_bus_init(ST3215_Bus_t *bus, UART_HandleTypeDef *huart) {
    if (bus == NULL) return;
    memset(bus, 0, sizeof(*bus));
    bus->huart = huart;
    bus->phase = ST_BUS_IDLE;
    arm_rx(bus);
}

uint8_t st3215_bus_is_idle(const ST3215_Bus_t *bus) {
    return (uint8_t)(bus != NULL && bus->phase == ST_BUS_IDLE);
}

int st3215_bus_queue_sync_write(ST3215_Bus_t *bus,
                                const uint8_t *ids, uint8_t count,
                                const int16_t *positions_ticks,
                                const uint16_t *speeds_ticks,
                                const uint8_t *accels) {
    int len;
    if (bus == NULL || bus->phase != ST_BUS_IDLE) return -1;
    len = st3215_build_sync_write(bus->tx, ST_MAX_PACKET_SIZE, ids, count,
                                  positions_ticks, speeds_ticks, accels);
    if (len < 0) return len;
    return start_tx(bus, (uint8_t)len, ST_BUS_TX_ONLY);
}

int st3215_bus_queue_torque(ST3215_Bus_t *bus, uint8_t id, uint8_t enable) {
    int len;
    if (bus == NULL || bus->phase != ST_BUS_IDLE) return -1;
    len = st3215_build_torque(bus->tx, id, enable);
    return len < 0 ? len : start_tx(bus, (uint8_t)len, ST_BUS_TX_ONLY);
}

int st3215_bus_queue_read(ST3215_Bus_t *bus, ST3215_State_t *target,
                          uint8_t offline_after, uint32_t now_ms) {
    int len;
    if (bus == NULL || target == NULL || bus->phase != ST_BUS_IDLE) return -1;
    len = st3215_build_read_state(bus->tx, target->id);
    if (len < 0) return len;
    bus->target = target;
    bus->offline_after = offline_after;
    bus->deadline_ms = now_ms + ST_REPLY_TIMEOUT_MS;
    reset_parser(bus);
    if (start_tx(bus, (uint8_t)len, ST_BUS_TX_READ) != 0) {
        bus->target = NULL;
        return -2;
    }
    return 0;
}

void st3215_bus_step(ST3215_Bus_t *bus, uint32_t now_ms) {
    if (bus == NULL) return;
    if ((bus->phase == ST_BUS_TX_READ || bus->phase == ST_BUS_WAIT_REPLY) &&
        (int32_t)(now_ms - bus->deadline_ms) >= 0) {
        finish_read_failure(bus, DEVICE_FAILURE_TIMEOUT);
    }
}

void st3215_bus_on_tx_complete(ST3215_Bus_t *bus) {
    if (bus == NULL) return;
    if (bus->phase == ST_BUS_TX_ONLY) bus->phase = ST_BUS_IDLE;
    else if (bus->phase == ST_BUS_TX_READ) bus->phase = ST_BUS_WAIT_REPLY;
}

void st3215_bus_on_rx_byte(ST3215_Bus_t *bus, uint32_t now_ms) {
    if (bus == NULL) return;
    consume_with_echo_filter(bus, bus->rx_byte, now_ms);
    arm_rx(bus);
}

void st3215_bus_on_uart_error(ST3215_Bus_t *bus) {
    if (bus == NULL) return;
    if (bus->phase == ST_BUS_TX_READ || bus->phase == ST_BUS_WAIT_REPLY) {
        finish_read_failure(bus, DEVICE_FAILURE_PROTOCOL);
    } else if (bus->phase == ST_BUS_TX_ONLY) {
        bus->phase = ST_BUS_IDLE;
    }
    arm_rx(bus);
}

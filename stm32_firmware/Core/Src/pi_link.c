#include "pi_link.h"
#include "crc8.h"
#include "kuafu_generated.h"
#include "pin_config.h"
#include <string.h>
#include <math.h>

Pi_Command_Heartbeat_t g_pi_cmd_heartbeat;
Pi_Command_Action_t g_pi_cmd_action;

static uint16_t tx_sequence = 0;
static uint16_t last_rx_sequence = 0;
static uint8_t have_rx_sequence = 0;
static uint8_t link_compatible = 0;
static uint8_t rx_stream[2 * (12 + PI_MAX_PAYLOAD)];
static uint16_t rx_stream_len = 0;

#define PI_TX_QUEUE_DEPTH 4U
#define PI_MAX_FRAME_SIZE (12U + PI_MAX_PAYLOAD)
static uint8_t tx_frames[PI_TX_QUEUE_DEPTH][PI_MAX_FRAME_SIZE];
static uint8_t tx_lengths[PI_TX_QUEUE_DEPTH];
static uint8_t tx_head = 0U;
static uint8_t tx_tail = 0U;
static uint8_t tx_count = 0U;
static uint8_t tx_active = 0U;
static UART_HandleTypeDef *tx_uart = NULL;

static int16_t read_i16_be(const uint8_t *bytes) {
    return (int16_t)(((uint16_t)bytes[0] << 8) | bytes[1]);
}

static void write_i16_be(uint8_t *bytes, int16_t value) {
    bytes[0] = (uint8_t)((value >> 8) & 0xff);
    bytes[1] = (uint8_t)(value & 0xff);
}

static int16_t quantize_i16(float value, float scale) {
    float scaled = value * scale;
    if (scaled > 32767.0f) return 32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)scaled;
}

static uint8_t sequence_is_new(uint16_t sequence) {
    uint16_t delta = (uint16_t)(sequence - last_rx_sequence);
    return !have_rx_sequence || (delta != 0u && delta < 0x8000u);
}

void pi_link_init(void) {
    memset(&g_pi_cmd_heartbeat, 0, sizeof(g_pi_cmd_heartbeat));
    memset(&g_pi_cmd_action, 0, sizeof(g_pi_cmd_action));
    g_pi_cmd_heartbeat.target_leg_d0 = 0.058f;
    tx_sequence = 0;
    have_rx_sequence = 0;
    link_compatible = 0;
    rx_stream_len = 0;
    tx_head = 0U;
    tx_tail = 0U;
    tx_count = 0U;
    tx_active = 0U;
    tx_uart = NULL;
}

void pi_link_clear_action(void) {
    __disable_irq();
    memset(&g_pi_cmd_action, 0, sizeof(g_pi_cmd_action));
    __enable_irq();
}

void pi_link_enter_hold(void) {
    __disable_irq();
    memset(&g_pi_cmd_action, 0, sizeof(g_pi_cmd_action));
    g_pi_cmd_heartbeat.target_velocity = 0.0f;
    g_pi_cmd_heartbeat.target_yaw_rate = 0.0f;
    __enable_irq();
}

uint8_t pi_link_is_compatible(void) { return link_compatible; }

uint8_t pi_link_heartbeat_fresh(void) {
    return g_pi_cmd_heartbeat.last_heartbeat_ms > 0 &&
           HAL_GetTick() - g_pi_cmd_heartbeat.last_heartbeat_ms <= SAFETY_HEARTBEAT_MS;
}

uint8_t pi_link_action_fresh(void) {
    return g_pi_cmd_action.last_action_ms > 0 &&
           HAL_GetTick() - g_pi_cmd_action.last_action_ms <= SAFETY_ACTION_MS;
}

static int parse_payload(uint8_t type, const uint8_t *payload, uint8_t payload_len) {
    if (type == PI_CMD_HELLO) {
        if (payload_len != 16 || memcmp(payload, KUAFU_MODEL_HASH, 16) != 0) {
            link_compatible = 0;
            return 0;
        }
        memset(&g_pi_cmd_heartbeat, 0, sizeof(g_pi_cmd_heartbeat));
        memset(&g_pi_cmd_action, 0, sizeof(g_pi_cmd_action));
        g_pi_cmd_heartbeat.target_leg_d0 = D0_MIN_MM * 0.001f;
        link_compatible = 1;
        return 1;
    }
    if (type == PI_CMD_HEARTBEAT) {
        if (!link_compatible) return 0;
        if (payload_len != 7) return 0;
        int16_t raw_v = read_i16_be(&payload[1]);
        int16_t raw_w = read_i16_be(&payload[3]);
        int16_t raw_d0 = read_i16_be(&payload[5]);
        float vx = (float)raw_v / 1000.0f;
        float wz = (float)raw_w / 1000.0f;
        float d0_mm = (float)raw_d0;
        float d0_max = (fabsf(vx) > D0_GATE_V_THRESH || fabsf(wz) > D0_GATE_W_THRESH)
                           ? D0_GATE_MAX_HIGH : D0_MAX_MM;
        if (payload[0] > 4 || vx < -0.5f || vx > 0.5f || wz < -1.0f || wz > 1.0f ||
            d0_mm < D0_MIN_MM || d0_mm > d0_max) return 0;
        g_pi_cmd_heartbeat.mode_request = payload[0];
        g_pi_cmd_heartbeat.target_velocity = vx;
        g_pi_cmd_heartbeat.target_yaw_rate = wz;
        g_pi_cmd_heartbeat.target_leg_d0 = d0_mm / 1000.0f;
        g_pi_cmd_heartbeat.last_heartbeat_ms = HAL_GetTick();
        return 1;
    }
    if (type == PI_CMD_ACTION) {
        if (!link_compatible) return 0;
        if (payload_len != 12) return 0;
        int16_t action[6];
        for (int i = 0; i < 6; ++i) action[i] = read_i16_be(&payload[2 * i]);
        for (int i = 0; i < 6; ++i) {
            if (action[i] < -10000 || action[i] > 10000) return 0;
        }
        g_pi_cmd_action.delta_torque_common = (float)action[0] / 10000.0f;
        g_pi_cmd_action.delta_torque_yaw = (float)action[1] / 10000.0f;
        g_pi_cmd_action.qx_l = (float)action[2] / 10000.0f;
        g_pi_cmd_action.d0_l = (float)action[3] / 10000.0f;
        g_pi_cmd_action.qx_r = (float)action[4] / 10000.0f;
        g_pi_cmd_action.d0_r = (float)action[5] / 10000.0f;
        g_pi_cmd_action.last_action_ms = HAL_GetTick();
        return 1;
    }
    return 0;
}

int pi_link_parse_packet(const uint8_t *buf, uint16_t len) {
    if (len == 0) return 0;
    if (len > sizeof(rx_stream) - rx_stream_len) {
        rx_stream_len = 0;  /* Lost framing: resynchronize at the next header. */
        if (len > sizeof(rx_stream)) return 0;
    }
    memcpy(&rx_stream[rx_stream_len], buf, len);
    rx_stream_len += len;

    int parsed = 0;
    uint16_t offset = 0;
    while (rx_stream_len - offset >= 12u) {
        if (rx_stream[offset] != PI_FRAME_HEADER) {
            ++offset;
            continue;
        }
        uint8_t version = rx_stream[offset + 1];
        uint8_t type = rx_stream[offset + 2];
        uint8_t payload_len = rx_stream[offset + 3];
        uint16_t frame_len = (uint16_t)(12u + payload_len);
        if (version != PI_PROTOCOL_VERSION || payload_len > PI_MAX_PAYLOAD) {
            ++offset;
            continue;
        }
        if (rx_stream_len - offset < frame_len) break;  /* Retain fragment. */
        if (rx_stream[offset + frame_len - 1] != PI_FRAME_FOOTER ||
            crc8_calculate(&rx_stream[offset + 1], (uint16_t)(9u + payload_len)) !=
                rx_stream[offset + frame_len - 2]) {
            ++offset;
            continue;
        }
        uint16_t sequence = (uint16_t)(((uint16_t)rx_stream[offset + 4] << 8) | rx_stream[offset + 5]);
        /* A validated HELLO starts a new Pi session and deliberately resets the
         * sequence window; every other frame is monotonic within that session. */
        uint8_t is_hello = type == PI_CMD_HELLO;
        if ((is_hello || sequence_is_new(sequence)) && parse_payload(type, &rx_stream[offset + 10], payload_len)) {
            last_rx_sequence = sequence;
            have_rx_sequence = 1;
            ++parsed;
        }
        offset += frame_len;
    }
    if (offset > 0) {
        memmove(rx_stream, &rx_stream[offset], rx_stream_len - offset);
        rx_stream_len -= offset;
    }
    return parsed;
}

static int start_next_tx(void) {
    if (tx_active || tx_count == 0U || tx_uart == NULL) return 0;
    tx_active = 1U;
    if (HAL_UART_Transmit_IT(tx_uart, tx_frames[tx_head], tx_lengths[tx_head]) != HAL_OK) {
        tx_active = 0U;
        return -1;
    }
    return 0;
}

static void write_u16_be(uint8_t *bytes, uint16_t value) {
    bytes[0] = (uint8_t)(value >> 8);
    bytes[1] = (uint8_t)value;
}

static void write_u32_be(uint8_t *bytes, uint32_t value) {
    bytes[0] = (uint8_t)(value >> 24);
    bytes[1] = (uint8_t)(value >> 16);
    bytes[2] = (uint8_t)(value >> 8);
    bytes[3] = (uint8_t)value;
}

void pi_link_on_tx_complete(UART_HandleTypeDef *huart) {
    if (!tx_active || huart != tx_uart) return;
    tx_head = (uint8_t)((tx_head + 1U) % PI_TX_QUEUE_DEPTH);
    --tx_count;
    tx_active = 0U;
    (void)start_next_tx();
}

void pi_link_on_tx_error(UART_HandleTypeDef *huart) {
    if (huart != tx_uart || tx_count == 0U) return;
    tx_head = (uint8_t)((tx_head + 1U) % PI_TX_QUEUE_DEPTH);
    --tx_count;
    tx_active = 0U;
    (void)start_next_tx();
}

static int pi_link_transmit(UART_HandleTypeDef *huart, uint8_t type,
                            const uint8_t *payload, uint8_t payload_len) {
    uint8_t frame[12 + PI_MAX_PAYLOAD];
    uint16_t sequence = tx_sequence++;
    uint32_t timestamp = HAL_GetTick();
    frame[0] = PI_FRAME_HEADER;
    frame[1] = PI_PROTOCOL_VERSION;
    frame[2] = type;
    frame[3] = payload_len;
    frame[4] = (uint8_t)(sequence >> 8);
    frame[5] = (uint8_t)sequence;
    frame[6] = (uint8_t)(timestamp >> 24);
    frame[7] = (uint8_t)(timestamp >> 16);
    frame[8] = (uint8_t)(timestamp >> 8);
    frame[9] = (uint8_t)timestamp;
    if (payload_len > 0) memcpy(&frame[10], payload, payload_len);
    frame[10 + payload_len] = crc8_calculate(&frame[1], (uint16_t)(9u + payload_len));
    frame[11 + payload_len] = PI_FRAME_FOOTER;
    __disable_irq();
    if (tx_count >= PI_TX_QUEUE_DEPTH) {
        __enable_irq();
        return -1;
    }
    memcpy(tx_frames[tx_tail], frame, (uint16_t)(12U + payload_len));
    tx_lengths[tx_tail] = (uint8_t)(12U + payload_len);
    tx_tail = (uint8_t)((tx_tail + 1U) % PI_TX_QUEUE_DEPTH);
    ++tx_count;
    tx_uart = huart;
    if (start_next_tx() != 0) {
        tx_tail = (uint8_t)((tx_tail + PI_TX_QUEUE_DEPTH - 1U) % PI_TX_QUEUE_DEPTH);
        --tx_count;
        __enable_irq();
        return -1;
    }
    __enable_irq();
    return 0;
}

int pi_link_send_imu(UART_HandleTypeDef *huart, float roll, float pitch, float yaw,
                     float gx, float gy, float gz) {
    uint8_t payload[12];
    float values[6] = {roll, pitch, yaw, gx, gy, gz};
    for (int i = 0; i < 6; ++i) write_i16_be(&payload[2 * i], quantize_i16(values[i], 1000.0f));
    return pi_link_transmit(huart, PI_CMD_TELEMETRY_IMU, payload, sizeof(payload));
}

int pi_link_send_joints(UART_HandleTypeDef *huart,
                        float wheel_l_pos, float wheel_l_vel, float wheel_l_tau,
                        float wheel_r_pos, float wheel_r_vel, float wheel_r_tau,
                        const float *servo_pos, const float *servo_vel, const float *servo_cur) {
    uint8_t payload[36];
    int16_t values[18];
    values[0] = quantize_i16(wheel_l_pos, 1000.0f);
    values[1] = quantize_i16(wheel_l_vel, WHEEL_SPEED_SCALE);
    values[2] = quantize_i16(wheel_l_tau, 10000.0f);
    values[3] = quantize_i16(wheel_r_pos, 1000.0f);
    values[4] = quantize_i16(wheel_r_vel, WHEEL_SPEED_SCALE);
    values[5] = quantize_i16(wheel_r_tau, 10000.0f);
    for (int i = 0; i < 4; ++i) {
        values[6 + 3 * i] = quantize_i16(servo_pos[i], 1000.0f);
        values[7 + 3 * i] = quantize_i16(servo_vel[i], 1000.0f);
        values[8 + 3 * i] = quantize_i16(servo_cur[i], 1000.0f);
    }
    for (int i = 0; i < 18; ++i) write_i16_be(&payload[2 * i], values[i]);
    return pi_link_transmit(huart, PI_CMD_TELEMETRY_JOINTS, payload, sizeof(payload));
}

int pi_link_send_diag(UART_HandleTypeDef *huart, uint16_t battery_mv, uint8_t max_temp_c, uint8_t error_mask) {
    uint8_t payload[4] = {
        (uint8_t)(battery_mv >> 8), (uint8_t)battery_mv, max_temp_c, error_mask
    };
    return pi_link_transmit(huart, PI_CMD_TELEMETRY_DIAG, payload, sizeof(payload));
}

void pi_link_encode_health_payload(uint8_t payload[PI_HEALTH_PAYLOAD_SIZE],
                                   const Pi_HealthTelemetry_t *health) {
    uint8_t offset = 0U;
    uint8_t i;
    if (payload == NULL || health == NULL) return;
    write_u32_be(&payload[offset], health->fault_mask); offset += 4U;
    payload[offset++] = health->mode;
    payload[offset++] = health->reset_cause;
    write_u16_be(&payload[offset], health->imu_age_ms); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_l_age_ms); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_r_age_ms); offset += 2U;
    for (i = 0U; i < 4U; ++i) {
        write_u16_be(&payload[offset], health->servo_age_ms[i]); offset += 2U;
    }
    write_u16_be(&payload[offset], health->imu_errors); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_l_errors); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_r_errors); offset += 2U;
    for (i = 0U; i < 4U; ++i) {
        write_u16_be(&payload[offset], health->servo_errors[i]); offset += 2U;
    }
    write_u16_be(&payload[offset], health->wheel_l_timeout_errors); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_l_checksum_errors); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_l_protocol_errors); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_r_timeout_errors); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_r_checksum_errors); offset += 2U;
    write_u16_be(&payload[offset], health->wheel_r_protocol_errors); offset += 2U;
}

int pi_link_send_health(UART_HandleTypeDef *huart,
                        const Pi_HealthTelemetry_t *health) {
    uint8_t payload[PI_HEALTH_PAYLOAD_SIZE];
    if (health == NULL) return -1;
    pi_link_encode_health_payload(payload, health);
    return pi_link_transmit(huart, PI_CMD_TELEMETRY_HEALTH,
                            payload, PI_HEALTH_PAYLOAD_SIZE);
}

int pi_link_send_fault(UART_HandleTypeDef *huart, uint8_t fault_code) {
    return pi_link_transmit(huart, PI_CMD_TELEMETRY_FAULT, &fault_code, 1);
}

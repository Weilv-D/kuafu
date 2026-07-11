#include "ddsm315.h"
#include "crc8.h"
#include "pin_config.h"

#define RPM_TO_RADS         (2.0f * 3.14159265f / 60.0f)
#define POS_TO_RAD          (2.0f * 3.14159265f / 32768.0f)

/* Helper function to transmit packet via UART in blocking mode */
static int ddsm_send_packet(UART_HandleTypeDef *huart, const uint8_t *packet, uint16_t size) {
    if (HAL_UART_Transmit(huart, (uint8_t *)packet, size, 10) != HAL_OK) {
        return -1;
    }
    return 0;
}

int ddsm_set_torque(UART_HandleTypeDef *huart, uint8_t id, float torque_nm) {
    uint8_t packet[10];

    /* Clamp torque command to safety limits */
    if (torque_nm > DDSM_MAX_TORQUE_NM)  torque_nm = DDSM_MAX_TORQUE_NM;
    if (torque_nm < -DDSM_MAX_TORQUE_NM) torque_nm = -DDSM_MAX_TORQUE_NM;

    /* Scale torque to raw current command */
    int16_t raw_current = (int16_t)(torque_nm * DDSM_TORQUE_TO_RAW);

    packet[0] = id;
    packet[1] = 0x64; /* Speed/torque command control word */
    packet[2] = (uint8_t)((raw_current >> 8) & 0xFF);
    packet[3] = (uint8_t)(raw_current & 0xFF);
    packet[4] = 0x00;
    packet[5] = 0x00;
    packet[6] = 0x00; /* Acceleration time (0 = direct) */
    packet[7] = 0x00;
    packet[8] = 0x00;
    packet[9] = crc8_calculate(packet, 9);

    return ddsm_send_packet(huart, packet, 10);
}

int ddsm_set_speed(UART_HandleTypeDef *huart, uint8_t id, float rpm) {
    uint8_t packet[10];

    /* Target speed in 0.1 RPM units */
    int16_t raw_speed = (int16_t)(rpm * 10.0f);

    packet[0] = id;
    packet[1] = 0x64;
    packet[2] = (uint8_t)((raw_speed >> 8) & 0xFF);
    packet[3] = (uint8_t)(raw_speed & 0xFF);
    packet[4] = 0x00;
    packet[5] = 0x00;
    packet[6] = 0x00;
    packet[7] = 0x00;
    packet[8] = 0x00;
    packet[9] = crc8_calculate(packet, 9);

    return ddsm_send_packet(huart, packet, 10);
}

int ddsm_set_enable(UART_HandleTypeDef *huart, uint8_t id, uint8_t enable) {
    uint8_t packet[10];

    packet[0] = id;
    packet[1] = 0xA0; /* Mode/enable control word */
    packet[2] = enable ? 0x08 : 0x09; /* 0x08 = Enable, 0x09 = Disable */
    packet[3] = 0x00;
    packet[4] = 0x00;
    packet[5] = 0x00;
    packet[6] = 0x00;
    packet[7] = 0x00;
    packet[8] = 0x00;
    packet[9] = crc8_calculate(packet, 9);

    return ddsm_send_packet(huart, packet, 10);
}

int ddsm_set_mode(UART_HandleTypeDef *huart, uint8_t id, uint8_t mode) {
    uint8_t packet[10];

    packet[0] = id;
    packet[1] = 0xA0;
    packet[2] = 0x00;
    packet[3] = 0x00;
    packet[4] = 0x00;
    packet[5] = 0x00;
    packet[6] = 0x00;
    packet[7] = 0x00;
    packet[8] = 0x00;
    packet[9] = mode; /* The 10th byte represents the target mode, no CRC8! */

    return ddsm_send_packet(huart, packet, 10);
}

int ddsm_parse_feedback(const uint8_t *rx_buf, DDSM_State_t *state) {
    /* 1. Calculate and verify CRC-8/MAXIM */
    uint8_t calculated_crc = crc8_calculate(rx_buf, 9);
    if (calculated_crc != rx_buf[9]) {
        return -1; /* CRC error */
    }

    /* 2. Verify motor ID */
    if (rx_buf[0] != state->id) {
        return -2; /* Wrong ID */
    }

    /* 3. Parse feedback values */
    state->mode = rx_buf[1];

    /* Torque Current (two's complement int16) */
    int16_t raw_current = (int16_t)(((uint16_t)rx_buf[2] << 8) | rx_buf[3]);
    state->torque = (float)raw_current * DDSM_RAW_TO_TORQUE;

    /* Speed (two's complement int16, in RPM) */
    int16_t raw_speed = (int16_t)(((uint16_t)rx_buf[4] << 8) | rx_buf[5]);
    state->velocity_rads = (float)raw_speed * RPM_TO_RADS;

    /* Position (unsigned 16-bit single-turn, range 0 to 32767) */
    uint16_t raw_position = ((uint16_t)rx_buf[6] << 8) | rx_buf[7];
    state->position_rad = (float)raw_position * POS_TO_RAD;

    state->error_code = rx_buf[8];
    state->last_update_ms = HAL_GetTick();

    return 0;
}

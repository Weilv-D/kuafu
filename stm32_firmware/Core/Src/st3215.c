#include "st3215.h"
#include "pin_config.h"

#define TICK_TO_RAD          ((2.0f * 3.14159265f) / 4096.0f)

int st3215_sync_write_pos(UART_HandleTypeDef *huart,
                          const uint8_t *ids,
                          uint8_t count,
                          const int16_t *positions_ticks,
                          const uint16_t *speeds_ticks,
                          const uint8_t *accels) {
    if (count == 0 || count > 10) return -1;

    uint8_t length = 8 * count + 4;
    uint8_t tx_buf[128];

    tx_buf[0] = 0xFF;
    tx_buf[1] = 0xFF;
    tx_buf[2] = 0xFE; /* Broadcast ID */
    tx_buf[3] = length;
    tx_buf[4] = ST_INST_SYNC_WRITE;
    tx_buf[5] = ST_REG_ACC;
    tx_buf[6] = 7; /* Data length per servo (ACC + POS_L + POS_H + TIME_L + TIME_H + SPD_L + SPD_H) */

    uint8_t sum = 0xFE + length + ST_INST_SYNC_WRITE + ST_REG_ACC + 7;

    for (uint8_t i = 0; i < count; i++) {
        /* ST3215 goal position is an unsigned 16-bit value in [0, 4095].
         * The caller may pass out-of-range ints (e.g. before center-offset
         * calibration); clamp to the valid range rather than sign-encoding. */
        int32_t pos_clamp = positions_ticks[i];
        if (pos_clamp < 0) pos_clamp = 0;
        if (pos_clamp > 4095) pos_clamp = 4095;
        uint16_t pos = (uint16_t)pos_clamp;

        uint16_t speed = speeds_ticks[i];
        uint8_t acc = accels[i];

        uint8_t p_idx = 7 + i * 8;
        tx_buf[p_idx + 0] = ids[i];
        tx_buf[p_idx + 1] = acc;
        tx_buf[p_idx + 2] = (uint8_t)(pos & 0xFF);
        tx_buf[p_idx + 3] = (uint8_t)((pos >> 8) & 0xFF);
        tx_buf[p_idx + 4] = 0; /* Goal Time L */
        tx_buf[p_idx + 5] = 0; /* Goal Time H */
        tx_buf[p_idx + 6] = (uint8_t)(speed & 0xFF);
        tx_buf[p_idx + 7] = (uint8_t)((speed >> 8) & 0xFF);

        sum += ids[i] + acc + (uint8_t)(pos & 0xFF) + (uint8_t)((pos >> 8) & 0xFF) +
               (uint8_t)(speed & 0xFF) + (uint8_t)((speed >> 8) & 0xFF);
    }

    tx_buf[7 + count * 8] = ~sum;

    /* Transmit the packet (8 * count + 8 bytes total) */
    if (HAL_UART_Transmit(huart, tx_buf, 8 * count + 8, 10) != HAL_OK) {
        return -1;
    }
    return 0;
}

int st3215_read_state(UART_HandleTypeDef *huart, uint8_t id, ST3215_State_t *state) {
    uint8_t tx_buf[8];
    /* Response: 0xFF 0xFF [ID] [Length=17] [Error] [Data(15)] [Checksum] = 21 bytes */
    uint8_t rx_buf[21];

    uint8_t read_len = 15;
    tx_buf[0] = 0xFF;
    tx_buf[1] = 0xFF;
    tx_buf[2] = id;
    tx_buf[3] = 4; /* Length */
    tx_buf[4] = ST_INST_READ;
    tx_buf[5] = ST_REG_PRESENT_POSITION_L; /* 56 */
    tx_buf[6] = read_len;
    tx_buf[7] = ~(id + 4 + ST_INST_READ + ST_REG_PRESENT_POSITION_L + read_len);

    /* Send read query */
    if (HAL_UART_Transmit(huart, tx_buf, 8, 5) != HAL_OK) {
        return -1;
    }

    /* Full-duplex: discard any loopback / leftover RX bytes before the servo's
     * reply arrives on the RX line. */
    __HAL_UART_CLEAR_OREFLAG(huart);
    __HAL_UART_FLUSH_DRREGISTER(huart);

    /* Receive response (21 bytes) */
    if (HAL_UART_Receive(huart, rx_buf, 21, 2) != HAL_OK) {
        return -1; /* Timeout or RX error */
    }

    /* Verify response format */
    if (rx_buf[0] != 0xFF || rx_buf[1] != 0xFF || rx_buf[2] != id || rx_buf[3] != 17) {
        return -1; /* Invalid packet format */
    }

    /* Verify Checksum */
    uint8_t sum = id + 17 + rx_buf[4];
    for (uint8_t i = 0; i < read_len; i++) {
        sum += rx_buf[5 + i];
    }
    if ((uint8_t)(~sum) != rx_buf[20]) {
        return -2; /* Checksum error */
    }

    /* Parse register block (offset relative to reg 56) */
    /* Reg 56/57: Present Position (unsigned, [0, 4095]) */
    uint16_t raw_pos = ((uint16_t)rx_buf[5 + 1] << 8) | rx_buf[5 + 0];
    state->position_tick = raw_pos;
    state->position_rad = (float)(raw_pos - SERVO_CENTER_TICKS) * TICK_TO_RAD;

    /* Reg 58/59: Present Speed (two's complement int16, bit15 = direction) */
    uint16_t u16_speed = ((uint16_t)rx_buf[5 + 3] << 8) | rx_buf[5 + 2];
    int16_t raw_speed = (int16_t)u16_speed;
    state->velocity_rads = (float)raw_speed * TICK_TO_RAD;

    /* Reg 60/61: Present Load (bit10 = sign, bits 0-9 = magnitude 0..1000) */
    uint16_t u16_load = ((uint16_t)rx_buf[5 + 5] << 8) | rx_buf[5 + 4];
    int16_t raw_load = (u16_load & 0x0400) ? -(int16_t)(u16_load & 0x03FF) : (int16_t)(u16_load & 0x03FF);
    state->load = (float)raw_load / 1000.0f; /* 1000 = 100% max load */

    /* Reg 62: Present Voltage */
    state->voltage = (float)rx_buf[5 + 6] * 0.1f; /* Scale factor 0.1V */

    /* Reg 63: Present Temperature */
    state->temperature_c = (float)rx_buf[5 + 7];

    /* Reg 69/70: Present Current (two's complement int16) */
    uint16_t u16_current = ((uint16_t)rx_buf[5 + 14] << 8) | rx_buf[5 + 13];
    int16_t raw_current = (int16_t)u16_current;
    state->current_a = (float)raw_current * 0.0065f; /* 6.5mA per LSB */

    state->id = id;
    return 0;
}

int st3215_set_torque_enable(UART_HandleTypeDef *huart, uint8_t id, uint8_t enable) {
    uint8_t tx_buf[8];
    tx_buf[0] = 0xFF;
    tx_buf[1] = 0xFF;
    tx_buf[2] = id;
    tx_buf[3] = 4;
    tx_buf[4] = ST_INST_WRITE;
    tx_buf[5] = ST_REG_TORQUE_ENABLE;
    tx_buf[6] = enable ? 1 : 0;
    tx_buf[7] = ~(id + 4 + ST_INST_WRITE + ST_REG_TORQUE_ENABLE + (enable ? 1 : 0));

    if (HAL_UART_Transmit(huart, tx_buf, 8, 5) != HAL_OK) {
        return -1;
    }
    return 0;
}

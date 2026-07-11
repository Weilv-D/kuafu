#include "pi_link.h"
#include "crc8.h"
#include <string.h>

Pi_Command_Heartbeat_t g_pi_cmd_heartbeat;
Pi_Command_Action_t g_pi_cmd_action;

static uint8_t tx_sequence = 0;

void pi_link_init(void) {
    memset(&g_pi_cmd_heartbeat, 0, sizeof(g_pi_cmd_heartbeat));
    memset(&g_pi_cmd_action, 0, sizeof(g_pi_cmd_action));
    /* Initialize leg length target to standard dwell pose (58mm = 0.058m) */
    g_pi_cmd_heartbeat.target_leg_d0 = 0.058f; 
}

int pi_link_parse_packet(const uint8_t *buf, uint16_t len) {
    if (len < 7) {
        return 0; /* Too short to be a valid frame */
    }

    int parsed_count = 0;
    uint16_t i = 0;

    /* Scan buffer for frame start */
    while (i <= len - 7) {
        if (buf[i] == PI_FRAME_HEADER) {
            uint8_t cmd = buf[i + 1];
            uint8_t seq = buf[i + 2];
            uint16_t payload_len = 0;

            if (cmd == PI_CMD_HEARTBEAT) {
                payload_len = 7;
            } else if (cmd == PI_CMD_ACTION) {
                payload_len = 12;
            } else {
                /* Unknown command, skip this header byte and look for next one */
                i++;
                continue;
            }

            /* Check if we have received the full frame in the buffer */
            if (i + 3 + payload_len + 2 > len) {
                /* Full packet not here yet, wait for more data */
                break;
            }

            const uint8_t *payload = &buf[i + 3];
            uint8_t rx_crc = buf[i + 3 + payload_len];
            uint8_t rx_footer = buf[i + 3 + payload_len + 1];

            /* Validate footer */
            if (rx_footer != PI_FRAME_FOOTER) {
                i++;
                continue;
            }

            /* Validate CRC-8 (covers CMD, SEQ, and Payload) */
            uint8_t calculated_crc = crc8_calculate(&buf[i + 1], 2 + payload_len);
            if (calculated_crc != rx_crc) {
                i++;
                continue; /* CRC mismatch */
            }

            /* Successfully verified packet. Parse payload. */
            if (cmd == PI_CMD_HEARTBEAT) {
                g_pi_cmd_heartbeat.mode_request = payload[0];

                int16_t raw_v = (int16_t)(((uint16_t)payload[1] << 8) | payload[2]);
                g_pi_cmd_heartbeat.target_velocity = (float)raw_v / 1000.0f;

                int16_t raw_w = (int16_t)(((uint16_t)payload[3] << 8) | payload[4]);
                g_pi_cmd_heartbeat.target_yaw_rate = (float)raw_w / 1000.0f;

                int16_t raw_d0 = (int16_t)(((uint16_t)payload[5] << 8) | payload[6]);
                g_pi_cmd_heartbeat.target_leg_d0 = (float)raw_d0 / 1000.0f;

                g_pi_cmd_heartbeat.last_heartbeat_ms = HAL_GetTick();
                (void)seq; /* Unused */

            } else if (cmd == PI_CMD_ACTION) {
                int16_t raw_tau_l = (int16_t)(((uint16_t)payload[0] << 8) | payload[1]);
                g_pi_cmd_action.delta_torque_l = (float)raw_tau_l / 10000.0f;

                int16_t raw_tau_r = (int16_t)(((uint16_t)payload[2] << 8) | payload[3]);
                g_pi_cmd_action.delta_torque_r = (float)raw_tau_r / 10000.0f;

                for (int j = 0; j < 4; j++) {
                    int16_t raw_q = (int16_t)(((uint16_t)payload[4 + j * 2] << 8) | payload[5 + j * 2]);
                    g_pi_cmd_action.target_q[j] = (float)raw_q / 10000.0f;
                }

                g_pi_cmd_action.last_action_ms = HAL_GetTick();
            }

            parsed_count++;
            i += 3 + payload_len + 2; /* Move past parsed frame */
        } else {
            i++;
        }
    }

    return parsed_count;
}

static int pi_link_transmit(UART_HandleTypeDef *huart, uint8_t cmd, const uint8_t *payload, uint16_t payload_len) {
    uint8_t tx_buf[64];
    tx_buf[0] = PI_FRAME_HEADER;
    tx_buf[1] = cmd;
    tx_buf[2] = tx_sequence++;
    
    if (payload_len > 0 && payload != NULL) {
        memcpy(&tx_buf[3], payload, payload_len);
    }
    
    uint8_t crc = crc8_calculate(&tx_buf[1], 2 + payload_len);
    tx_buf[3 + payload_len] = crc;
    tx_buf[3 + payload_len + 1] = PI_FRAME_FOOTER;

    if (HAL_UART_Transmit(huart, tx_buf, 5 + payload_len, 20) != HAL_OK) {
        return -1;
    }
    return 0;
}

int pi_link_send_imu(UART_HandleTypeDef *huart, float roll, float pitch, float yaw, float gx, float gy, float gz) {
    uint8_t payload[12];
    int16_t values[6];

    values[0] = (int16_t)(roll * 10000.0f);
    values[1] = (int16_t)(pitch * 10000.0f);
    values[2] = (int16_t)(yaw * 10000.0f);
    values[3] = (int16_t)(gx * 10000.0f);
    values[4] = (int16_t)(gy * 10000.0f);
    values[5] = (int16_t)(gz * 10000.0f);

    for (int i = 0; i < 6; i++) {
        payload[i * 2]     = (uint8_t)((values[i] >> 8) & 0xFF);
        payload[i * 2 + 1] = (uint8_t)(values[i] & 0xFF);
    }

    return pi_link_transmit(huart, PI_CMD_TELEMETRY_IMU, payload, 12);
}

int pi_link_send_joints(UART_HandleTypeDef *huart,
                        float wheel_l_pos, float wheel_l_vel, float wheel_l_tau,
                        float wheel_r_pos, float wheel_r_vel, float wheel_r_tau,
                        const float *servo_pos, const float *servo_vel, const float *servo_cur) {
    uint8_t payload[36];
    int16_t values[18];

    values[0] = (int16_t)(wheel_l_pos * 10000.0f);
    values[1] = (int16_t)(wheel_l_vel * 10000.0f);
    values[2] = (int16_t)(wheel_l_tau * 10000.0f);

    values[3] = (int16_t)(wheel_r_pos * 10000.0f);
    values[4] = (int16_t)(wheel_r_vel * 10000.0f);
    values[5] = (int16_t)(wheel_r_tau * 10000.0f);

    for (int i = 0; i < 4; i++) {
        values[6 + i * 3 + 0] = (int16_t)(servo_pos[i] * 10000.0f);
        values[6 + i * 3 + 1] = (int16_t)(servo_vel[i] * 10000.0f);
        values[6 + i * 3 + 2] = (int16_t)(servo_cur[i] * 10000.0f);
    }

    for (int i = 0; i < 18; i++) {
        payload[i * 2]     = (uint8_t)((values[i] >> 8) & 0xFF);
        payload[i * 2 + 1] = (uint8_t)(values[i] & 0xFF);
    }

    return pi_link_transmit(huart, PI_CMD_TELEMETRY_JOINTS, payload, 36);
}

int pi_link_send_diag(UART_HandleTypeDef *huart, uint16_t battery_mv, uint8_t max_temp_c, uint8_t error_mask) {
    uint8_t payload[4];

    payload[0] = (uint8_t)((battery_mv >> 8) & 0xFF);
    payload[1] = (uint8_t)(battery_mv & 0xFF);
    payload[2] = max_temp_c;
    payload[3] = error_mask;

    return pi_link_transmit(huart, PI_CMD_TELEMETRY_DIAG, payload, 4);
}

int pi_link_send_fault(UART_HandleTypeDef *huart, uint8_t fault_code) {
    uint8_t payload[1];
    payload[0] = fault_code;

    return pi_link_transmit(huart, PI_CMD_TELEMETRY_FAULT, payload, 1);
}

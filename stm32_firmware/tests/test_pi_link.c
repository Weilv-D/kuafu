#include "pi_link.h"
#include "pi_transport.h"
#include "crc8.h"
#include "kuafu_generated.h"
#include "test_support.h"

#include <string.h>

static uint16_t build_frame(uint8_t *frame, uint8_t version, uint8_t type,
                            uint16_t sequence, const uint8_t *payload,
                            uint8_t payload_len) {
    uint16_t total = (uint16_t)(12U + payload_len);
    frame[0] = PI_FRAME_HEADER; frame[1] = version; frame[2] = type;
    frame[3] = payload_len; frame[4] = (uint8_t)(sequence >> 8);
    frame[5] = (uint8_t)sequence;
    frame[6] = 0U; frame[7] = 0U; frame[8] = 0U; frame[9] = 1U;
    if (payload_len > 0U) memcpy(&frame[10], payload, payload_len);
    frame[10U + payload_len] = crc8_calculate(&frame[1], (uint16_t)(9U + payload_len));
    frame[11U + payload_len] = PI_FRAME_FOOTER;
    return total;
}

static void put_i16(uint8_t *bytes, int16_t value) {
    bytes[0] = (uint8_t)((uint16_t)value >> 8);
    bytes[1] = (uint8_t)value;
}

static void make_heartbeat(uint8_t payload[7], uint8_t mode,
                           int16_t velocity, int16_t yaw, int16_t d0_mm) {
    payload[0] = mode;
    put_i16(&payload[1], velocity);
    put_i16(&payload[3], yaw);
    put_i16(&payload[5], d0_mm);
}

void run_pi_link_tests(void) {
    uint8_t frame[96];
    uint8_t second[96];
    uint8_t joined[192];
    uint8_t heartbeat[7];
    uint8_t action[12];
    uint8_t bad_hash[16];
    uint16_t len;
    uint16_t len2;
    int i;
    UART_HandleTypeDef uart;

    test_set_time_ms(100U);
    pi_link_init();
    memset(bad_hash, '0', sizeof(bad_hash));
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_HELLO, 1U,
                      bad_hash, 16U);
    TEST_EQ_INT(0, pi_link_parse_packet(frame, len));
    TEST_TRUE(!pi_link_is_compatible());

    len = build_frame(frame, 2U, PI_CMD_HELLO, 2U,
                      (const uint8_t *)KUAFU_MODEL_HASH, 16U);
    len2 = build_frame(second, PI_PROTOCOL_VERSION, PI_CMD_HELLO, 10U,
                       (const uint8_t *)KUAFU_MODEL_HASH, 16U);
    memcpy(joined, frame, len); memcpy(&joined[len], second, len2);
    TEST_EQ_INT(1, pi_link_parse_packet(joined, (uint16_t)(len + len2)));
    TEST_TRUE(pi_link_is_compatible());

    make_heartbeat(heartbeat, 1U, 100, -200, 58);
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_HEARTBEAT, 11U,
                      heartbeat, sizeof(heartbeat));
    TEST_EQ_INT(0, pi_link_parse_packet(frame, 5U));
    TEST_EQ_INT(1, pi_link_parse_packet(&frame[5], (uint16_t)(len - 5U)));
    TEST_NEAR(0.1f, g_pi_cmd_heartbeat.target_velocity, 0.0001f);
    TEST_NEAR(-0.2f, g_pi_cmd_heartbeat.target_yaw_rate, 0.0001f);
    TEST_TRUE(pi_link_heartbeat_fresh());
    TEST_EQ_INT(0, pi_link_parse_packet(frame, len));

    frame[len - 2U] ^= 1U;
    TEST_EQ_INT(0, pi_link_parse_packet(frame, len));

    make_heartbeat(heartbeat, 1U, 600, 0, 58);
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_HEARTBEAT, 12U,
                      heartbeat, sizeof(heartbeat));
    TEST_EQ_INT(0, pi_link_parse_packet(frame, len));

    for (i = 0; i < 6; ++i) put_i16(&action[2 * i], (int16_t)(i & 1 ? -10000 : 10000));
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_ACTION, 12U,
                      action, sizeof(action));
    TEST_EQ_INT(1, pi_link_parse_packet(frame, len));
    TEST_NEAR(1.0f, g_pi_cmd_action.delta_torque_common, 0.0001f);
    TEST_NEAR(-1.0f, g_pi_cmd_action.delta_torque_yaw, 0.0001f);

    put_i16(action, 10001);
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_ACTION, 13U,
                      action, sizeof(action));
    TEST_EQ_INT(0, pi_link_parse_packet(frame, len));
    put_i16(action, 5000);
    test_set_time_ms(350U);
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_ACTION, 13U,
                      action, sizeof(action));
    TEST_EQ_INT(1, pi_link_parse_packet(frame, len));
    test_set_time_ms(400U);
    TEST_TRUE(!pi_link_heartbeat_fresh());
    TEST_TRUE(pi_link_action_fresh());
    pi_link_clear_action();
    TEST_TRUE(!pi_link_action_fresh());
    pi_link_enter_hold();
    TEST_NEAR(0.0f, g_pi_cmd_heartbeat.target_velocity, 0.0001f);

    /* A new HELLO resets the sequence window; 65535 -> 0 is a valid wrap. */
    test_set_time_ms(500U);
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_HELLO, 65534U,
                      (const uint8_t *)KUAFU_MODEL_HASH, 16U);
    TEST_EQ_INT(1, pi_link_parse_packet(frame, len));
    make_heartbeat(heartbeat, 1U, 0, 0, 58);
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_HEARTBEAT, 65535U,
                      heartbeat, sizeof(heartbeat));
    len2 = build_frame(second, PI_PROTOCOL_VERSION, PI_CMD_ACTION, 0U,
                       action, sizeof(action));
    joined[0] = 0x22U; joined[1] = 0xA4U;
    memcpy(&joined[2], frame, len); memcpy(&joined[2U + len], second, len2);
    TEST_EQ_INT(2, pi_link_parse_packet(joined, (uint16_t)(2U + len + len2)));
    TEST_EQ_INT(0, pi_link_parse_packet(frame, len));

    test_uart_reset();
    pi_link_init();
    TEST_EQ_INT(0, pi_link_send_diag(&uart, 0U, 40U, 0U));
    TEST_EQ_INT(0, pi_link_send_fault(&uart, 3U));
    TEST_EQ_INT(1, (int)test_uart_tx_count());
    pi_link_on_tx_complete(&uart);
    TEST_EQ_INT(2, (int)test_uart_tx_count());
    pi_link_on_tx_complete(&uart);
}

void run_pi_transport_tests(void) {
    uint8_t ring[64] = {0};
    uint8_t frame[96];
    uint8_t heartbeat[7];
    PiTransport_t transport;
    uint16_t len;

    pi_link_init();
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_HELLO, 200U,
                      (const uint8_t *)KUAFU_MODEL_HASH, 16U);
    pi_transport_init(&transport, ring, sizeof(ring));
    memcpy(ring, frame, 20U);
    TEST_EQ_INT(0, pi_transport_poll(&transport, (uint16_t)(sizeof(ring) - 20U)));
    memcpy(&ring[20], &frame[20], (size_t)(len - 20U));
    TEST_EQ_INT(1, pi_transport_poll(&transport,
                                     (uint16_t)(sizeof(ring) - len)));
    TEST_TRUE(pi_link_is_compatible());

    /* Place a complete heartbeat across the physical end of the DMA ring. */
    make_heartbeat(heartbeat, 1U, 50, 0, 58);
    len = build_frame(frame, PI_PROTOCOL_VERSION, PI_CMD_HEARTBEAT, 201U,
                      heartbeat, sizeof(heartbeat));
    transport.read_index = 60U;
    memcpy(&ring[60], frame, 4U);
    memcpy(ring, &frame[4], (size_t)(len - 4U));
    TEST_EQ_INT(1, pi_transport_poll(&transport,
                                     (uint16_t)(sizeof(ring) - (len - 4U))));
    TEST_NEAR(0.05f, g_pi_cmd_heartbeat.target_velocity, 0.0001f);
}

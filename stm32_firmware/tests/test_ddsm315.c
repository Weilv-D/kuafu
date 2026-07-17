#include "ddsm315.h"
#include "crc8.h"
#include "test_support.h"

#include <string.h>

static void make_feedback(uint8_t frame[DDSM_FRAME_SIZE], uint8_t id) {
    memset(frame, 0, DDSM_FRAME_SIZE);
    frame[0] = id;
    frame[1] = DDSM_MODE_CURRENT;
    frame[2] = 0xEAU; frame[3] = 0xABU; /* signed negative current */
    frame[4] = 0xFFU; frame[5] = 0x9CU; /* -100 rpm */
    frame[6] = 0x40U; frame[7] = 0x00U; /* pi radians */
    frame[8] = 0x12U;
    frame[9] = crc8_calculate(frame, 9U);
}

static void feed_bus(DDSM_Bus_t *bus, const uint8_t *bytes,
                     uint8_t count, uint32_t now_ms) {
    uint8_t i;
    for (i = 0U; i < count; ++i) {
        test_uart_supply_rx(&bytes[i], 1U);
        ddsm_bus_on_rx_byte(bus, now_ms);
    }
}

void run_ddsm315_tests(void) {
    uint8_t packet[DDSM_FRAME_SIZE];
    uint8_t frame[DDSM_FRAME_SIZE];
    DDSM_State_t state;
    DDSM_State_t right;
    DDSM_Bus_t bus;
    UART_HandleTypeDef uart;

    memset(&state, 0, sizeof(state));
    state.id = 1U;
    device_health_init(&state.health);
    memset(&right, 0, sizeof(right));
    right.id = 2U;
    device_health_init(&right.health);

    ddsm_build_torque(packet, 1U, 100.0f);
    TEST_EQ_U8(1U, packet[0]);
    TEST_EQ_U8(0x64U, packet[1]);
    TEST_EQ_U8(crc8_calculate(packet, 9U), packet[9]);
    TEST_EQ_INT((int)(1.1f * 5461.17f),
                (int)(int16_t)(((uint16_t)packet[2] << 8) | packet[3]));

    ddsm_build_torque(packet, 1U, -100.0f);
    TEST_EQ_INT((int)(-1.1f * 5461.17f),
                (int)(int16_t)(((uint16_t)packet[2] << 8) | packet[3]));

    ddsm_build_query(packet, 2U);
    TEST_EQ_U8(2U, packet[0]);
    TEST_EQ_U8(0x74U, packet[1]);
    TEST_EQ_U8(crc8_calculate(packet, 9U), packet[9]);

    ddsm_build_set_id(packet, 2U);
    TEST_EQ_U8(0xAAU, packet[0]);
    TEST_EQ_U8(0x55U, packet[1]);
    TEST_EQ_U8(0x53U, packet[2]);
    TEST_EQ_U8(2U, packet[3]);
    TEST_EQ_U8(0x92U, packet[9]);

    make_feedback(frame, 1U);
    TEST_EQ_INT(0, ddsm_parse_feedback(frame, &state));
    TEST_TRUE(state.torque < 0.0f);
    TEST_NEAR(-1.04719755f, state.velocity_rads, 0.0001f);
    TEST_NEAR(3.14159265f, state.position_rad, 0.0001f);
    TEST_EQ_U8(0x12U, state.error_code);
    frame[9] ^= 1U;
    TEST_EQ_INT(-1, ddsm_parse_feedback(frame, &state));
    make_feedback(frame, 2U);
    TEST_EQ_INT(-2, ddsm_parse_feedback(frame, &state));

    test_uart_reset();
    ddsm_bus_init(&bus, &uart);
    TEST_TRUE(ddsm_bus_is_idle(&bus));
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.25f, 10U));
    TEST_EQ_INT(DDSM_BUS_TX, bus.phase);
    TEST_EQ_INT(1, (int)test_uart_tx_count());
    TEST_EQ_INT(DDSM_FRAME_SIZE, test_uart_last_tx_size());
    TEST_EQ_INT(-2, ddsm_bus_queue_torque(&bus, &state, 0.0f, 10U));

    ddsm_bus_on_tx_complete(&bus);
    TEST_EQ_INT(DDSM_BUS_RX, bus.phase);
    feed_bus(&bus, bus.tx, DDSM_FRAME_SIZE, 10U); /* self-echo */
    make_feedback(frame, 1U);
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 11U);
    TEST_TRUE(ddsm_bus_is_idle(&bus));
    TEST_TRUE(state.health.online);
    TEST_EQ_INT(11, (int)state.health.last_valid_ms);

    /* The serialized owner accepts the opposite motor after the first slot. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &right, -0.25f, 12U));
    ddsm_bus_on_tx_complete(&bus);
    make_feedback(frame, 2U);
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 13U);
    TEST_TRUE(right.health.online);

    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 20U));
    ddsm_bus_step(&bus, 21U);
    TEST_EQ_INT(DDSM_BUS_TX, bus.phase);
    ddsm_bus_step(&bus, 24U);
    TEST_EQ_INT(DDSM_BUS_TX, bus.phase);
    ddsm_bus_step(&bus, 27U);
    TEST_EQ_INT(DDSM_BUS_TX, bus.phase);
    ddsm_bus_step(&bus, 28U);
    TEST_TRUE(ddsm_bus_is_idle(&bus));
    TEST_EQ_INT(0, (int)test_uart_abort_count());
    TEST_EQ_INT(1, (int)state.health.timeout_count);

    /* A valid transaction after timeout restores online health. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 30U));
    ddsm_bus_on_tx_complete(&bus);
    make_feedback(frame, 1U);
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 31U);
    TEST_EQ_INT(0, (int)state.health.consecutive_failures);
    TEST_TRUE(state.health.online);

    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 40U));
    ddsm_bus_on_tx_complete(&bus);
    make_feedback(frame, 1U);
    frame[9] ^= 1U;
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 41U);
    TEST_EQ_INT(1, (int)state.health.checksum_count);
    TEST_EQ_INT(DDSM_BUS_RX, bus.phase);
    make_feedback(frame, 1U);
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 42U);
    TEST_TRUE(ddsm_bus_is_idle(&bus));
    TEST_EQ_INT(0, (int)state.health.consecutive_failures);
}

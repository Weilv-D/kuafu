#include "ddsm315.h"
#include "crc8.h"
#include "test_support.h"

#include <math.h>
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

    /* DDSM315 Protocol-3 mode frame: A0 header, mode VALUE in byte[9], NO CRC.
     * byte[2] must remain the sub-command slot (0x00 here), never the mode. */
    ddsm_build_mode(packet, 1U, DDSM_MODE_CURRENT);
    TEST_EQ_U8(1U, packet[0]);
    TEST_EQ_U8(0xA0U, packet[1]);
    TEST_EQ_U8(0x00U, packet[2]);
    TEST_EQ_U8(0x01U, packet[9]); /* mode value, not a CRC */

    ddsm_build_mode(packet, 2U, DDSM_MODE_SPEED);
    TEST_EQ_U8(2U, packet[0]);
    TEST_EQ_U8(0x02U, packet[9]);
    TEST_EQ_U8(0x00U, packet[2]);

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

    /* Single outstanding transaction starts immediately. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.25f, 10U));
    TEST_EQ_INT(DDSM_BUS_TX, bus.phase);
    TEST_EQ_INT(1, (int)test_uart_tx_count());
    TEST_EQ_INT(DDSM_FRAME_SIZE, test_uart_last_tx_size());
    TEST_EQ_INT(DDSM_TX_STATUS_ACTIVE, (int)ddsm_bus_status(&bus));

    /* A busy bus queues (up to depth 4) instead of dropping the command;
     * overflow is reported explicitly and counted. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 10U));
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 10U));
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 10U));
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 10U));
    TEST_EQ_INT(4, (int)ddsm_bus_queue_depth(&bus));
    TEST_EQ_INT(-2, ddsm_bus_queue_torque(&bus, &state, 0.0f, 10U));
    TEST_EQ_INT(1, (int)bus.queue_overflow_count);

    /* Drain all five transactions.  A completed frame auto-starts the next
     * pending one, so the bus reports idle only when the queue empties. */
    for (int i = 0; i < 5; ++i) {
        ddsm_bus_on_tx_complete_at(&bus, 11U);
        TEST_EQ_INT(DDSM_BUS_RX, bus.phase);
        feed_bus(&bus, bus.tx, DDSM_FRAME_SIZE, 11U); /* adapter self-echo */
        make_feedback(frame, 1U);
        feed_bus(&bus, frame, DDSM_FRAME_SIZE, 12U);
    }
    TEST_TRUE(ddsm_bus_is_idle(&bus));
    TEST_EQ_INT(0, (int)ddsm_bus_queue_depth(&bus));
    TEST_EQ_INT(5, (int)test_uart_tx_count());
    TEST_TRUE(state.health.online);
    TEST_EQ_INT(12, (int)state.health.last_valid_ms);
    TEST_EQ_INT(DDSM_TX_STATUS_FEEDBACK_VALID, (int)ddsm_bus_status(&bus));
    TEST_EQ_INT(DDSM_MODE_CURRENT, (int)ddsm_bus_mode_feedback(&bus));

    /* The serialized owner accepts the opposite motor in the next slot. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &right, -0.25f, 13U));
    ddsm_bus_on_tx_complete_at(&bus, 13U);
    feed_bus(&bus, bus.tx, DDSM_FRAME_SIZE, 13U);
    make_feedback(frame, 2U);
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 14U);
    TEST_TRUE(right.health.online);
    TEST_TRUE(ddsm_bus_is_idle(&bus));

    /* No reply in time: bounded transaction deadline, health timeout, idle. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 20U));
    ddsm_bus_step(&bus, 21U);
    TEST_EQ_INT(DDSM_BUS_TX, bus.phase);
    ddsm_bus_step(&bus, 27U);
    TEST_EQ_INT(DDSM_BUS_TX, bus.phase);
    ddsm_bus_step(&bus, 32U);  /* 12 ms transaction deadline */
    TEST_TRUE(ddsm_bus_is_idle(&bus));
    TEST_EQ_INT(0, (int)test_uart_abort_count());
    TEST_EQ_INT(1, (int)state.health.timeout_count);
    TEST_EQ_INT(DDSM_TX_STATUS_TIMEOUT, (int)ddsm_bus_status(&bus));

    /* A valid transaction after timeout restores online health. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 30U));
    ddsm_bus_on_tx_complete_at(&bus, 30U);
    make_feedback(frame, 1U);
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 31U);
    TEST_EQ_INT(0, (int)state.health.consecutive_failures);
    TEST_TRUE(state.health.online);

    /* Corrupt CRC slides the window instead of failing the transaction. */
    TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 40U));
    ddsm_bus_on_tx_complete_at(&bus, 40U);
    make_feedback(frame, 1U);
    frame[9] ^= 1U;
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 41U);
    TEST_EQ_INT(1, (int)state.health.checksum_count);
    TEST_EQ_INT(DDSM_BUS_RX, bus.phase);
    make_feedback(frame, 1U);
    feed_bus(&bus, frame, DDSM_FRAME_SIZE, 42U);
    TEST_TRUE(ddsm_bus_is_idle(&bus));
    TEST_EQ_INT(0, (int)state.health.consecutive_failures);

    /* The TX snapshot reports what actually reached the wire, not an ACK. */
    {
        const DDSM_TxSnapshot_t *snap = ddsm_bus_get_last_tx(&bus);
        TEST_TRUE(snap != NULL);
        TEST_EQ_INT(1, (int)snap->id);
        TEST_EQ_INT(40, (int)snap->queued_ms);
        TEST_EQ_INT(40, (int)snap->tx_complete_ms);
        TEST_EQ_INT(42, (int)snap->finished_ms);
        TEST_EQ_INT(DDSM_TX_STATUS_FEEDBACK_VALID, (int)snap->status);
    }

    /* Queuing a no-reply control frame while a reply-expecting transaction
     * is in flight must not disturb the active transaction: the expectation
     * travels with the queued frame itself (start_pending re-arms it), so a
     * late enable/mode can never collapse the in-flight reply window and
     * cost a freshness timeout. */
    {
        test_uart_reset();
        memset(&state, 0, sizeof(state));
        state.id = 1U;
        device_health_init(&state.health);
        ddsm_bus_init(&bus, &uart);

        TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.25f, 60U));
        TEST_EQ_INT(1, (int)bus.expect_reply);
        TEST_EQ_INT(0, ddsm_bus_queue_enable(&bus, &state, 1U, 60U));
        TEST_EQ_INT(1, (int)ddsm_bus_queue_depth(&bus));
        TEST_EQ_INT(1, (int)bus.expect_reply); /* active transaction intact */

        ddsm_bus_on_tx_complete_at(&bus, 61U);
        TEST_EQ_INT(DDSM_BUS_RX, bus.phase);   /* still waiting for the reply */
        feed_bus(&bus, bus.tx, DDSM_FRAME_SIZE, 61U); /* adapter self-echo */
        make_feedback(frame, 1U);
        feed_bus(&bus, frame, DDSM_FRAME_SIZE, 62U);
        TEST_TRUE(state.health.online);
        TEST_EQ_INT(0, (int)ddsm_bus_queue_depth(&bus));
        TEST_EQ_INT(DDSM_BUS_TX, bus.phase);   /* enable already auto-started */

        /* The queued enable runs as a no-reply transaction. */
        TEST_EQ_INT(0, (int)bus.expect_reply);
        ddsm_bus_on_tx_complete_at(&bus, 63U);
        TEST_TRUE(ddsm_bus_is_idle(&bus));
    }

    /* Gate-closed dispatch on an ENABLED wheel sends an explicit zero-torque
     * frame, not a read query: the DDSM315 current loop holds its last
     * setpoint, so a query-only policy would leave the pre-closure torque
     * driving the robot until the TILT fault backstop.  Pin both properties
     * the scheduler relies on: the zero frame is byte-exact, and it keeps
     * the same reply/freshness channel as every other torque frame. */
    {
        test_uart_reset();
        memset(&state, 0, sizeof(state));
        state.id = 1U;
        device_health_init(&state.health);
        ddsm_bus_init(&bus, &uart);

        ddsm_build_torque(packet, 1U, 0.0f);
        TEST_EQ_U8(0x64U, packet[1]);
        TEST_EQ_INT(0, (int)(int16_t)(((uint16_t)packet[2] << 8) | packet[3]));
        TEST_EQ_U8(crc8_calculate(packet, 9U), packet[9]);

        TEST_EQ_INT(0, ddsm_bus_queue_torque(&bus, &state, 0.0f, 70U));
        TEST_EQ_INT(1, (int)bus.expect_reply);
        ddsm_bus_on_tx_complete_at(&bus, 70U);
        make_feedback(frame, 1U);
        feed_bus(&bus, frame, DDSM_FRAME_SIZE, 71U);
        TEST_TRUE(state.health.online);
        TEST_TRUE(ddsm_bus_is_idle(&bus));
        TEST_EQ_INT(DDSM_TX_STATUS_FEEDBACK_VALID, (int)ddsm_bus_status(&bus));
    }

    /* A non-finite torque command decodes as zero (coast), never as an
     * undefined float-to-int cast or a saturated command: NaN compares false
     * against every clamp bound, so the encoder must reject it explicitly. */
    {
        uint8_t nan_packet[DDSM_FRAME_SIZE];
        memset(nan_packet, 0, sizeof(nan_packet));
        ddsm_build_torque(nan_packet, 1U, (float)NAN);
        TEST_EQ_INT(0, (int)(int16_t)(((uint16_t)nan_packet[2] << 8) | nan_packet[3]));
        TEST_EQ_U8(crc8_calculate(nan_packet, 9U), nan_packet[9]);
    }
}

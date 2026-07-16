#include "st3215.h"
#include "test_support.h"

#include <string.h>

static uint8_t st_checksum(const uint8_t *body, uint8_t size) {
    uint8_t sum = 0U;
    uint8_t i;
    for (i = 0U; i < size; ++i) sum = (uint8_t)(sum + body[i]);
    return (uint8_t)~sum;
}

static void make_state_frame(uint8_t frame[ST_STATE_FRAME_SIZE], uint8_t id) {
    memset(frame, 0, ST_STATE_FRAME_SIZE);
    frame[0] = 0xFFU; frame[1] = 0xFFU; frame[2] = id; frame[3] = 17U;
    frame[5] = 0x34U; frame[6] = 0x02U;
    frame[7] = 0x9CU; frame[8] = 0xFFU;
    frame[9] = 0x2CU; frame[10] = 0x05U;
    frame[11] = 120U; frame[12] = 45U;
    frame[18] = 0xF6U; frame[19] = 0xFFU;
    frame[20] = st_checksum(&frame[2], 18U);
}

static void feed(ST3215_Bus_t *bus, const uint8_t *bytes,
                 uint8_t count, uint32_t now_ms) {
    uint8_t i;
    for (i = 0U; i < count; ++i) {
        test_uart_supply_rx(&bytes[i], 1U);
        st3215_bus_on_rx_byte(bus, now_ms);
    }
}

void run_st3215_tests(void) {
    uint8_t packet[ST_MAX_PACKET_SIZE];
    uint8_t frame[ST_STATE_FRAME_SIZE];
    uint8_t ids[2] = {1U, 2U};
    int16_t positions[2] = {-1, 5000};
    uint16_t speeds[2] = {0x1234U, 0x5678U};
    uint8_t accels[2] = {10U, 20U};
    ST3215_State_t state;
    ST3215_Bus_t bus;
    UART_HandleTypeDef uart;
    int len;

    len = st3215_build_sync_write(packet, sizeof(packet), ids, 2U,
                                   positions, speeds, accels);
    TEST_EQ_INT(24, len);
    TEST_EQ_U8(0xFEU, packet[2]);
    TEST_EQ_U8(ST_INST_SYNC_WRITE, packet[4]);
    TEST_EQ_U8(0U, packet[9]); TEST_EQ_U8(0U, packet[10]);
    TEST_EQ_U8(0xFFU, packet[17]); TEST_EQ_U8(0x0FU, packet[18]);
    TEST_EQ_U8(st_checksum(&packet[2], 21U), packet[23]);

    TEST_EQ_INT(8, st3215_build_torque(packet, 3U, 1U));
    TEST_EQ_U8(3U, packet[2]); TEST_EQ_U8(ST_INST_WRITE, packet[4]);
    TEST_EQ_U8(ST_REG_TORQUE_ENABLE, packet[5]); TEST_EQ_U8(1U, packet[6]);
    TEST_EQ_U8(st_checksum(&packet[2], 5U), packet[7]);

    TEST_EQ_INT(8, st3215_build_torque(packet, ST3215_BROADCAST_ID, 0U));
    TEST_EQ_U8(ST3215_BROADCAST_ID, packet[2]);
    TEST_EQ_U8(0U, packet[6]);
    TEST_EQ_U8(st_checksum(&packet[2], 5U), packet[7]);

    memset(&state, 0, sizeof(state));
    state.id = 1U;
    device_health_init(&state.health);
    make_state_frame(frame, 1U);
    TEST_EQ_INT(0, st3215_parse_state_frame(frame, ST_STATE_FRAME_SIZE, &state));
    TEST_EQ_INT(0x234, state.position_tick);
    TEST_TRUE(state.velocity_rads < 0.0f);
    TEST_NEAR(-0.300f, state.load, 0.001f);
    TEST_NEAR(12.0f, state.voltage, 0.001f);
    TEST_NEAR(45.0f, state.temperature_c, 0.001f);
    TEST_TRUE(state.current_a < 0.0f);
    make_state_frame(frame, 2U);
    TEST_EQ_INT(-2, st3215_parse_state_frame(frame, ST_STATE_FRAME_SIZE, &state));
    make_state_frame(frame, 1U); frame[20] ^= 1U;
    TEST_EQ_INT(-3, st3215_parse_state_frame(frame, ST_STATE_FRAME_SIZE, &state));

    test_uart_reset();
    st3215_bus_init(&bus, &uart);
    TEST_EQ_INT(0, st3215_bus_queue_read(&bus, &state, 3U, 100U));
    TEST_EQ_INT(ST_BUS_TX_READ, bus.phase);
    st3215_bus_on_tx_complete(&bus);
    TEST_EQ_INT(ST_BUS_WAIT_REPLY, bus.phase);

    /* Known request echo is discarded. Noise and partial response are retained. */
    feed(&bus, bus.tx, bus.tx_len, 101U);
    { const uint8_t noise[] = {0x12U, 0xFFU, 0x44U}; feed(&bus, noise, 3U, 101U); }
    make_state_frame(frame, 1U);
    feed(&bus, frame, 7U, 101U);
    TEST_EQ_INT(ST_BUS_WAIT_REPLY, bus.phase);
    feed(&bus, &frame[7], (uint8_t)(ST_STATE_FRAME_SIZE - 7U), 102U);
    TEST_TRUE(st3215_bus_is_idle(&bus));
    TEST_TRUE(state.health.online);
    TEST_EQ_INT(102, (int)state.health.last_valid_ms);

    /* A complete stale frame for another ID is ignored; adjacent target frame wins. */
    TEST_EQ_INT(0, st3215_bus_queue_read(&bus, &state, 3U, 110U));
    st3215_bus_on_tx_complete(&bus);
    make_state_frame(frame, 2U);
    feed(&bus, frame, ST_STATE_FRAME_SIZE, 111U);
    TEST_EQ_INT(ST_BUS_WAIT_REPLY, bus.phase);
    make_state_frame(frame, 1U);
    feed(&bus, frame, ST_STATE_FRAME_SIZE, 112U);
    TEST_TRUE(st3215_bus_is_idle(&bus));

    TEST_EQ_INT(0, st3215_bus_queue_read(&bus, &state, 3U, 120U));
    st3215_bus_step(&bus, 122U);
    TEST_EQ_INT(ST_BUS_TX_READ, bus.phase);
    st3215_bus_step(&bus, 123U);
    TEST_TRUE(st3215_bus_is_idle(&bus));
    TEST_EQ_INT(1, (int)state.health.timeout_count);
}

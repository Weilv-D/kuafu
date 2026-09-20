#include "pi_transport.h"
#include "pi_link.h"
#include "crc8.h"
#include "test_support.h"

#include <string.h>

/* poll() is observed through its cursor/counter side effects and its parse
 * return value; pi_link_parse_packet runs underneath on real buffers, so
 * fill bytes that decode as noise (no A5 header) keep it harmless. */
void run_pi_transport_overrun_tests(void) {
    uint8_t ring_buf[8];
    PiTransport_t transport;
    memset(ring_buf, 0x11, sizeof(ring_buf));

    /* --- incremental production, no lap, no overrun --- */
    pi_transport_init(&transport, ring_buf, sizeof(ring_buf));
    TEST_EQ_INT(0, pi_transport_poll(&transport, 0U, 8U)); /* empty ring */
    TEST_EQ_INT(0, (int)pi_transport_overruns(&transport));
    TEST_EQ_INT(0, pi_transport_poll(&transport, 0U, 5U)); /* 3 new bytes */
    TEST_EQ_INT(3, (int)transport.producer_count);
    TEST_EQ_INT(3, (int)transport.consumed_count);
    TEST_EQ_INT(3, (int)transport.read_index);

    /* --- NDTR zero sample means "filled to the physical end" --- */
    TEST_EQ_INT(0, pi_transport_poll(&transport, 0U, 0U));
    TEST_EQ_INT(8, (int)transport.producer_count);
    TEST_EQ_INT(8, (int)transport.consumed_count);
    TEST_EQ_INT(0, (int)transport.read_index);
    TEST_EQ_INT(0, (int)pi_transport_overruns(&transport));

    /* --- one lap plus fresh bytes parses only the new span --- */
    pi_transport_note_lap(&transport);
    TEST_EQ_INT(0, pi_transport_poll(&transport, 1U, 6U));
    TEST_EQ_INT(10, (int)transport.producer_count);
    TEST_EQ_INT(10, (int)transport.consumed_count);
    TEST_EQ_INT(2, (int)transport.read_index);
    TEST_EQ_INT(0, (int)pi_transport_overruns(&transport));

    /* --- consumer laps the full ring: exact loss counted, cursor synced --- */
    pi_transport_note_lap(&transport);
    pi_transport_note_lap(&transport);
    TEST_EQ_INT(0, pi_transport_poll(&transport, 3U, 4U));
    TEST_EQ_INT(28, (int)transport.producer_count);
    /* 28 produced - 8 capacity = 20 readable; 10 were consumed, so exactly
     * 10 bytes were overwritten and billed.  The 8 still-valid bytes are
     * parsed in the SAME poll (consumed catches up to produced); a consumer
     * left one ring behind would never parse again. */
    TEST_EQ_INT(10, (int)pi_transport_overruns(&transport));
    TEST_EQ_INT(28, (int)transport.consumed_count);
    TEST_EQ_INT(4, (int)transport.read_index);

    /* --- stale lap sample undercounts and never fabricates a loss --- */
    pi_transport_note_lap(&transport);            /* completion lands "late" */
    TEST_EQ_INT(0, pi_transport_poll(&transport, 3U, 6U)); /* stale laps */
    TEST_EQ_INT(28, (int)transport.producer_count); /* monotonic max held */
    TEST_EQ_INT(10, (int)pi_transport_overruns(&transport));
    TEST_EQ_INT(0, pi_transport_poll(&transport, 4U, 6U)); /* healed */
    TEST_EQ_INT(34, (int)transport.producer_count);
    TEST_EQ_INT(10, (int)pi_transport_overruns(&transport));

    /* --- reset after a DMA re-arm skips the stale prefix --- */
    pi_transport_note_lap(&transport);
    memset(ring_buf, 0x22, sizeof(ring_buf));
    pi_transport_reset(&transport);
    TEST_EQ_INT(5U * 8, (int)transport.producer_count);
    TEST_EQ_INT(5U * 8, (int)transport.consumed_count);
    TEST_EQ_INT(0, (int)transport.read_index);
    ring_buf[0] = 0x33U; /* fresh byte at the ring base parses as noise */
    TEST_EQ_INT(0, pi_transport_poll(&transport, 5U, 7U));
    TEST_EQ_INT(5U * 8 + 1, (int)transport.producer_count);
    TEST_EQ_INT(5U * 8 + 1, (int)transport.consumed_count);
    TEST_EQ_INT(1, (int)transport.read_index);
    TEST_EQ_INT(10, (int)pi_transport_overruns(&transport));
}

/* Regression: after the consumer has been lapped by a full ring (the main-loop
 * stall case), fresh command frames must still parse.  The old span logic
 * derived the parse length from read_index vs write_index; after the overrun
 * drop the two coincide while a full ring is pending, so the transport billed
 * every later byte as a fresh overrun and parsed nothing — the Pi link stayed
 * deaf until reboot. */
static uint16_t build_heartbeat_frame(uint8_t *frame, uint16_t sequence) {
    uint8_t payload[7];
    payload[0] = 1U; /* mode_request = STAND */
    payload[1] = 0U; payload[2] = 0U;   /* vx = 0 */
    payload[3] = 0U; payload[4] = 0U;   /* wz = 0 */
    payload[5] = 0U; payload[6] = 58U;  /* d0 = 58 mm */
    frame[0] = PI_FRAME_HEADER;
    frame[1] = PI_PROTOCOL_VERSION;
    frame[2] = PI_CMD_HEARTBEAT;
    frame[3] = sizeof(payload);
    frame[4] = (uint8_t)(sequence >> 8);
    frame[5] = (uint8_t)sequence;
    frame[6] = 0U; frame[7] = 0U; frame[8] = 0U; frame[9] = 1U;
    memcpy(&frame[10], payload, sizeof(payload));
    frame[10 + sizeof(payload)] =
        crc8_calculate(&frame[1], (uint16_t)(9U + sizeof(payload)));
    frame[11 + sizeof(payload)] = PI_FRAME_FOOTER;
    return (uint16_t)(12U + sizeof(payload));
}

void run_pi_transport_overrun_recovery_tests(void) {
    uint8_t ring_buf[64];
    PiTransport_t transport;
    uint8_t frame[32];
    uint16_t frame_len;

    test_set_time_ms(1000U);
    pi_link_init(); /* clears the sequence-gate baseline for this test */
    memset(ring_buf, 0x11, sizeof(ring_buf)); /* noise: no A5 header */
    pi_transport_init(&transport, ring_buf, sizeof(ring_buf));

    /* Force a full-ring lap: the producer is three rings ahead of a consumer
     * that has read nothing (main-loop stall while the Pi streams).  Exactly
     * the overwritten prefix is billed, and the still-valid ring is parsed in
     * the same poll so the consumer catches up instead of stalling one ring
     * behind the producer. */
    TEST_EQ_INT(0, pi_transport_poll(&transport, 3U, 10U)); /* write_index 54 */
    TEST_EQ_INT(3U * 64U + 54U, (int)transport.producer_count);
    TEST_EQ_INT(3U * 64U + 54U - 64U, (int)pi_transport_overruns(&transport));
    TEST_EQ_INT(3U * 64U + 54U, (int)transport.consumed_count);
    TEST_EQ_INT(54, (int)transport.read_index);

    /* Fresh production across the wrap must parse normally: 19 bytes from
     * position 54 complete lap 4 and continue at position 9. */
    frame_len = build_heartbeat_frame(frame, 100U);
    memcpy(&ring_buf[54], frame, 10U);          /* 54..64 */
    memcpy(ring_buf, &frame[10], frame_len - 10U); /* 0..9 */
    TEST_EQ_INT(1, pi_transport_poll(&transport, 4U, 55U));
    TEST_EQ_INT(4U * 64U + 9U, (int)transport.producer_count);
    TEST_EQ_INT(4U * 64U + 9U, (int)transport.consumed_count);
    TEST_EQ_INT(9, (int)transport.read_index);
    TEST_EQ_INT(3U * 64U + 54U - 64U, (int)pi_transport_overruns(&transport));

    /* A second frame after the wrap keeps the link alive. */
    frame_len = build_heartbeat_frame(frame, 101U);
    memcpy(&ring_buf[9], frame, frame_len);
    TEST_EQ_INT(1, pi_transport_poll(&transport, 4U, 36U));
    TEST_EQ_INT(4U * 64U + 28U, (int)transport.producer_count);
    TEST_EQ_INT(4U * 64U + 28U, (int)transport.consumed_count);
    TEST_EQ_INT(28, (int)transport.read_index);
    TEST_EQ_INT(3U * 64U + 54U - 64U, (int)pi_transport_overruns(&transport));
}

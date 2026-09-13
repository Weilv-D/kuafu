#include "pi_transport.h"
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
     * 10 bytes were overwritten and billed. */
    TEST_EQ_INT(10, (int)pi_transport_overruns(&transport));
    TEST_EQ_INT(20, (int)transport.consumed_count);
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

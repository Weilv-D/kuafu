#include "dma_rx_ring.h"
#include "test_support.h"

#include <string.h>

#define TEST_RING_SIZE 8U
static uint8_t test_buffer[TEST_RING_SIZE];
static DmaRxRing_t ring;

/* Simulate the DMA writing one byte: land it at the next slot; the circular
 * stream reloads (transfer-complete) exactly when the position wraps to
 * zero, then the consumer observes (lap_count, position) in contract order. */
static uint16_t sim_pos;
static void produce_byte(uint8_t value) {
    ring.buffer[sim_pos] = value;
    sim_pos = (uint16_t)((sim_pos + 1U) % TEST_RING_SIZE);
    if (sim_pos == 0U) dma_rx_ring_note_lap(&ring);
    dma_rx_ring_update_producer(&ring, ring.lap_count, sim_pos);
}

static void test_init_and_empty(void) {
    dma_rx_ring_init(&ring, test_buffer, TEST_RING_SIZE); sim_pos = 0U;
    TEST_EQ_INT(0, (int)dma_rx_ring_available(&ring));
    TEST_EQ_INT(0, (int)dma_rx_ring_overruns(&ring));
    TEST_EQ_INT(0, (int)dma_rx_ring_produced(&ring));
    TEST_EQ_INT(0, (int)dma_rx_ring_consumed(&ring));
}

static void test_consume_preserves_order_across_wrap(void) {
    uint8_t out[TEST_RING_SIZE];
    uint16_t got;
    dma_rx_ring_init(&ring, test_buffer, TEST_RING_SIZE); sim_pos = 0U;
    for (uint8_t i = 0U; i < 5U; ++i) produce_byte((uint8_t)('a' + i));
    TEST_EQ_INT(5, (int)dma_rx_ring_available(&ring));
    got = dma_rx_ring_consume(&ring, out, sizeof(out));
    TEST_EQ_INT(5, (int)got);
    TEST_EQ_INT(0, memcmp(out, "abcde", 5U));
    TEST_EQ_INT(0, (int)dma_rx_ring_available(&ring));

    /* Wrap the physical ring and confirm FIFO order survives the boundary. */
    for (uint8_t i = 0U; i < 7U; ++i) produce_byte((uint8_t)('0' + i));
    got = dma_rx_ring_consume(&ring, out, sizeof(out));
    TEST_EQ_INT(7, (int)got);
    TEST_EQ_INT(0, memcmp(out, "0123456", 7U));
}

static void test_partial_consume_capacity_limited(void) {
    uint8_t out[3];
    dma_rx_ring_init(&ring, test_buffer, TEST_RING_SIZE); sim_pos = 0U;
    for (uint8_t i = 0U; i < 5U; ++i) produce_byte(i);
    TEST_EQ_INT(3, (int)dma_rx_ring_consume(&ring, out, sizeof(out)));
    TEST_EQ_INT(0, out[0]);
    TEST_EQ_INT(1, out[1]);
    TEST_EQ_INT(2, out[2]);
    TEST_EQ_INT(2, (int)dma_rx_ring_available(&ring));
}

static void test_overrun_keeps_newest_bytes_and_counts(void) {
    uint8_t out[TEST_RING_SIZE];
    dma_rx_ring_init(&ring, test_buffer, TEST_RING_SIZE); sim_pos = 0U;
    /* Produce size + 3 bytes with nothing consumed: the 3 oldest are lost,
     * overrun_count records exactly the loss, and a later consume returns the
     * newest size bytes in order. */
    for (uint16_t i = 0U; i < TEST_RING_SIZE + 3U; ++i) {
        produce_byte((uint8_t)(i & 0xFFU));
    }
    TEST_EQ_INT(3, (int)dma_rx_ring_overruns(&ring));
    TEST_EQ_INT((int)TEST_RING_SIZE, (int)dma_rx_ring_available(&ring));
    TEST_EQ_INT((int)TEST_RING_SIZE,
                (int)dma_rx_ring_consume(&ring, out, sizeof(out)));
    for (uint16_t i = 0U; i < TEST_RING_SIZE; ++i) {
        TEST_EQ_INT((int)((3U + i) & 0xFFU), (int)out[i]);
    }
    TEST_EQ_INT(3, (int)dma_rx_ring_overruns(&ring)); /* stable, not re-counted */
}

static void test_filled_to_end_without_transfer_complete(void) {
    dma_rx_ring_init(&ring, test_buffer, TEST_RING_SIZE); sim_pos = 0U;
    /* NDTR can read zero right before the reload: write index == size with
     * the lap counter not yet advanced.  That state is exactly one full ring
     * of production — at capacity, not an overrun — and the byte beyond it
     * (after the TC lands and the new lap begins) is the first overrun. */
    dma_rx_ring_update_producer(&ring, 0U, TEST_RING_SIZE);
    TEST_EQ_INT(0, (int)dma_rx_ring_overruns(&ring));
    TEST_EQ_INT((int)TEST_RING_SIZE, (int)dma_rx_ring_available(&ring));
    dma_rx_ring_note_lap(&ring);
    dma_rx_ring_update_producer(&ring, 1U, 1U);
    TEST_EQ_INT(1, (int)dma_rx_ring_overruns(&ring));
}

static void test_stale_lap_sample_undercounts_then_heals(void) {
    dma_rx_ring_init(&ring, test_buffer, TEST_RING_SIZE); sim_pos = 0U;
    /* Exact baseline: six bytes produced. */
    dma_rx_ring_update_producer(&ring, 0U, 6U);
    TEST_EQ_INT(6, (int)dma_rx_ring_produced(&ring));
    /* The transfer-complete lands between the lap read and the position
     * read: a stale lap count with a post-wrap position undercounts by one
     * lap; the monotonic max keeps the previous estimate, and no overrun is
     * fabricated. */
    dma_rx_ring_note_lap(&ring);
    dma_rx_ring_update_producer(&ring, 0U, 2U);
    TEST_EQ_INT(6, (int)dma_rx_ring_produced(&ring));
    TEST_EQ_INT(0, (int)dma_rx_ring_overruns(&ring));
    /* The next contract-ordered sample restores the truth exactly. */
    dma_rx_ring_update_producer(&ring, 1U, 2U);
    TEST_EQ_INT((int)(TEST_RING_SIZE + 2U), (int)dma_rx_ring_produced(&ring));
}

static void test_null_and_invalid_arguments(void) {
    uint8_t out[4];
    dma_rx_ring_init(&ring, test_buffer, TEST_RING_SIZE); sim_pos = 0U;
    dma_rx_ring_update_producer(&ring, 0U, (uint16_t)(TEST_RING_SIZE + 1U));
    TEST_EQ_INT(0, (int)dma_rx_ring_available(&ring));
    TEST_EQ_INT(0, (int)dma_rx_ring_consume(&ring, out, 0U));
    TEST_EQ_INT(0, (int)dma_rx_ring_consume(NULL, out, sizeof(out)));
    TEST_EQ_INT(0, (int)dma_rx_ring_available(NULL));
    TEST_EQ_INT(0, (int)dma_rx_ring_overruns(NULL));
}

void run_dma_rx_ring_tests(void) {
    test_init_and_empty();
    test_consume_preserves_order_across_wrap();
    test_partial_consume_capacity_limited();
    test_overrun_keeps_newest_bytes_and_counts();
    test_filled_to_end_without_transfer_complete();
    test_stale_lap_sample_undercounts_then_heals();
    test_null_and_invalid_arguments();
}

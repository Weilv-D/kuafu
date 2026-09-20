#ifndef PI_TRANSPORT_H
#define PI_TRANSPORT_H

#include <stdint.h>

/* Circular-DMA consumer for the Pi UART RX ring.
 *
 * The DMA stream is the sole producer.  The absolute produced count is
 * reconstructed as lap_count * size + write_index, where lap_count is fed by
 * the DMA transfer-complete callback (pi_transport_note_lap, once per ring
 * lap) and the write index comes from NDTR.  Ordering contract (same proof
 * as dma_rx_ring.h): the caller reads transport->lap_count BEFORE sampling
 * NDTR; every interrupt interleaving of that order is exact or a bounded
 * undercount, never an overcount, so a race can never fabricate an overrun.
 * poll() additionally derives the parse span from the absolute cursors
 * (producer - consumed, clamped to one ring) rather than from a
 * read_index-vs-write_index comparison, so an undercount sample can only
 * defer parsing by one poll and consumed_count can never overshoot
 * producer_count.
 *
 * When the consumer falls a full ring behind, the overwritten prefix is
 * dropped and counted in overrun_count and the still-valid remainder of the
 * ring is parsed in the same poll, so the consumer always catches back up;
 * the streaming decoder resynchronizes on the A5 header.  Frame-level
 * integrity is independently guaranteed by the CRC-8 and the monotonic
 * sequence check in pi_link, so dropped bytes can never replay a stale
 * command. */
typedef struct {
    uint8_t *buffer;
    uint16_t size;
    uint16_t read_index;        /* next byte to parse, in [0, size) */
    volatile uint32_t lap_count;/* DMA transfer-complete callbacks */
    uint32_t producer_count;    /* absolute bytes written (reconstructed) */
    uint32_t consumed_count;    /* absolute bytes handed to the parser */
    uint32_t overrun_count;     /* bytes overwritten before they were parsed */
} PiTransport_t;

void pi_transport_init(PiTransport_t *transport, uint8_t *buffer, uint16_t size);

/* Call from the DMA transfer-complete (UART RxCplt) callback for the Pi RX
 * stream — interrupt context, one invocation per completed ring lap. */
void pi_transport_note_lap(PiTransport_t *transport);

/* Feed the lap count (read BEFORE sampling NDTR) and the current NDTR:
 *     laps = transport->lap_count;
 *     remaining = NDTR;
 *     pi_transport_poll(transport, laps, remaining);
 * Returns the number of accepted command frames. */
int pi_transport_poll(PiTransport_t *transport,
                      uint32_t laps,
                      uint16_t dma_remaining);

/* After re-arming the DMA reception (abort + Receive_DMA restarts the stream
 * at the ring base), drop the pre-restart content instead of re-parsing it. */
void pi_transport_reset(PiTransport_t *transport);

uint32_t pi_transport_overruns(const PiTransport_t *transport);

#endif

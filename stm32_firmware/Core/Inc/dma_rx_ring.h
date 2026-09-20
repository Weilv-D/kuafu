#ifndef DMA_RX_RING_H
#define DMA_RX_RING_H

#include <stdint.h>
#include <stddef.h>

/* Producer/consumer accounting for a circular UART RX DMA stream.
 *
 * The DMA is the sole producer.  Its position register (NDTR) is modular,
 * so a consumer that samples less often than one full ring lap cannot tell
 * laps apart from position alone.  The lap truth source is the DMA
 * transfer-complete interrupt, which the circular stream raises exactly
 * once per lap; the consumer reconstructs the absolute produced count as
 *
 *     produced = lap_count * size + write_index        (monotonic max)
 *
 * Ordering contract: the caller MUST read ring->lap_count BEFORE sampling
 * NDTR.  Exhaustive over the interrupt interleavings of that order:
 *   - completion before both reads  -> exact;
 *   - completion between the two    -> stale lap count with post-wrap
 *     position: undercount by one lap, self-healed by the next update;
 *   - completion after both         -> exact at the sampling instant.
 * No interleaving overcounts, so a sampling race can never fabricate an
 * overrun and drop good bytes.  The monotonic max absorbs the undercount
 * transients (and the NDTR==0 "filled to the end" sample, which carries a
 * lap that the transfer-complete has not reported yet).
 *
 * When the consumer genuinely falls a full ring behind, the overwritten
 * prefix is billed to overrun_count and dropped; the streaming consumers
 * resynchronize on their frame headers and per-frame checksums. */
typedef struct {
    uint8_t *buffer;
    uint16_t size;
    volatile uint32_t lap_count;    /* DMA transfer-complete callbacks */
    volatile uint32_t produced;     /* absolute bytes written (reconstructed) */
    volatile uint32_t consumed;     /* absolute bytes consumed */
    volatile uint32_t overrun_count;/* bytes overwritten before consumption */
} DmaRxRing_t;

void dma_rx_ring_init(DmaRxRing_t *ring, uint8_t *buffer, uint16_t size);

/* Re-arm recovery for a restarted circular RX stream (AbortReceive +
 * Receive_DMA): rebases produced/consumed to the current lap boundary while
 * PRESERVING lap_count (every lap counted so far belongs to the old stream;
 * the restarted stream produces from lap_count * size).  overrun_count is a
 * cumulative diagnostic and is preserved.  Call only after the new
 * HAL_UART_Receive_DMA has succeeded. */
void dma_rx_ring_rebase(DmaRxRing_t *ring);

/* Call from the DMA transfer-complete (UART RxCplt) callback — interrupt
 * context, one invocation per completed ring lap. */
void dma_rx_ring_note_lap(DmaRxRing_t *ring);

/* Consumer-side producer update.  The write index is in [0, size]; NDTR
 * briefly reads zero at the reload and that sample means "filled to the
 * physical end".  Order the reads: lap_count first, then NDTR. */
void dma_rx_ring_update_producer(DmaRxRing_t *ring,
                                 uint32_t laps, uint16_t dma_write_index);

uint16_t dma_rx_ring_available(const DmaRxRing_t *ring);
uint16_t dma_rx_ring_consume(DmaRxRing_t *ring, uint8_t *out, uint16_t capacity);
uint32_t dma_rx_ring_overruns(const DmaRxRing_t *ring);
uint32_t dma_rx_ring_produced(const DmaRxRing_t *ring);
uint32_t dma_rx_ring_consumed(const DmaRxRing_t *ring);

#endif

#include "dma_rx_ring.h"

#include <string.h>

void dma_rx_ring_init(DmaRxRing_t *ring, uint8_t *buffer, uint16_t size) {
    if (ring == NULL) return;
    ring->buffer = buffer;
    ring->size = size;
    ring->lap_count = 0U;
    ring->produced = 0U;
    ring->consumed = 0U;
    ring->overrun_count = 0U;
}

void dma_rx_ring_note_lap(DmaRxRing_t *ring) {
    if (ring == NULL) return;
    ++ring->lap_count;
}

void dma_rx_ring_update_producer(DmaRxRing_t *ring,
                                 uint32_t laps, uint16_t dma_write_index) {
    uint32_t candidate;
    uint32_t pending;
    if (ring == NULL || ring->size == 0U || dma_write_index > ring->size) return;

    candidate = laps * (uint32_t)ring->size + dma_write_index;
    if (candidate > ring->produced) {
        ring->produced = candidate;
    }

    pending = ring->produced - ring->consumed;
    if (pending > (uint32_t)ring->size) {
        ring->overrun_count += pending - (uint32_t)ring->size;
        ring->consumed = ring->produced - (uint32_t)ring->size;
    }
}

uint16_t dma_rx_ring_available(const DmaRxRing_t *ring) {
    uint32_t pending;
    if (ring == NULL || ring->size == 0U) return 0U;
    pending = ring->produced - ring->consumed;
    return (uint16_t)(pending > (uint32_t)ring->size ? (uint32_t)ring->size : pending);
}

uint16_t dma_rx_ring_consume(DmaRxRing_t *ring, uint8_t *out, uint16_t capacity) {
    uint16_t count;
    uint16_t index;
    uint16_t first;
    if (ring == NULL || out == NULL || ring->buffer == NULL || capacity == 0U) return 0U;
    count = dma_rx_ring_available(ring);
    if (count > capacity) count = capacity;
    index = (uint16_t)(ring->consumed % (uint32_t)ring->size);
    first = (uint16_t)(ring->size - index);
    if (first > count) first = count;
    memcpy(out, &ring->buffer[index], first);
    if (count > first) memcpy(&out[first], ring->buffer, (size_t)(count - first));
    ring->consumed += count;
    return count;
}

uint32_t dma_rx_ring_overruns(const DmaRxRing_t *ring) {
    return ring == NULL ? 0U : ring->overrun_count;
}

uint32_t dma_rx_ring_produced(const DmaRxRing_t *ring) {
    return ring == NULL ? 0U : ring->produced;
}

uint32_t dma_rx_ring_consumed(const DmaRxRing_t *ring) {
    return ring == NULL ? 0U : ring->consumed;
}

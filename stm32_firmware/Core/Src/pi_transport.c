#include "pi_transport.h"
#include "pi_link.h"

#include <stddef.h>

/* Write index for a circular RX stream from its remaining-count register.
 * NDTR can briefly read zero before the reload; that sample means "filled
 * to the physical end of the ring". */
static uint16_t write_index_from_remaining(uint16_t size, uint16_t dma_remaining) {
    return dma_remaining == 0U ? size : (uint16_t)(size - dma_remaining);
}

void pi_transport_init(PiTransport_t *transport, uint8_t *buffer, uint16_t size) {
    if (transport == NULL) return;
    transport->buffer = buffer;
    transport->size = size;
    transport->read_index = 0U;
    transport->lap_count = 0U;
    transport->producer_count = 0U;
    transport->consumed_count = 0U;
    transport->overrun_count = 0U;
}

void pi_transport_note_lap(PiTransport_t *transport) {
    if (transport == NULL) return;
    ++transport->lap_count;
}

void pi_transport_reset(PiTransport_t *transport) {
    if (transport == NULL) return;
    /* The restarted DMA stream begins writing at the ring base.  Re-baseline
     * the reconstructed producer at the current lap boundary (lap_count keeps
     * counting across the restart, so the new stream's production resumes
     * from lap_count * size + 0) and bill the unread pre-restart prefix to
     * the consumer so the byte counters stay monotonic. */
    transport->producer_count = transport->lap_count * (uint32_t)transport->size;
    transport->consumed_count = transport->producer_count;
    transport->read_index = 0U;
}

int pi_transport_poll(PiTransport_t *transport,
                      uint32_t laps,
                      uint16_t dma_remaining) {
    uint16_t write_index;
    uint16_t span;
    uint32_t pending;
    int parsed = 0;
    if (transport == NULL || transport->buffer == NULL || transport->size == 0U ||
        dma_remaining > transport->size) return -1;

    write_index = write_index_from_remaining(transport->size, dma_remaining);
    {
        uint32_t candidate = laps * (uint32_t)transport->size + write_index;
        if (candidate > transport->producer_count) {
            transport->producer_count = candidate;
        }
    }

    /* The parse span is derived from the absolute cursors, never from a
     * read_index-vs-write_index comparison.  After an overrun the two indices
     * coincide while a FULL ring is still pending, and a cursor comparison
     * reports that as "nothing to do": the consumer would then sit exactly one
     * ring behind the producer forever, billing every freshly arrived byte as
     * another overrun and parsing nothing — permanent link deafness until
     * reboot.  Cursor-derived spans also make the laps/NDTR sampling race
     * harmless by construction: a stale sample can only under-advance the
     * producer for one poll (the monotonic max absorbs it next poll), and
     * consumed_count can never overshoot producer_count. */
    pending = transport->producer_count - transport->consumed_count;
    if (pending > (uint32_t)transport->size) {
        /* The DMA lapped the parser: the oldest pending bytes are already
         * overwritten.  Drop exactly the lost prefix and count it; the rest of
         * the ring is still valid and is parsed below, so the consumer catches
         * up in this same poll instead of stalling one ring behind. */
        uint16_t lost = (uint16_t)(pending - (uint32_t)transport->size);
        transport->overrun_count += lost;
        transport->consumed_count += lost;
        transport->read_index = (uint16_t)(transport->producer_count %
                                           (uint32_t)transport->size);
        pending = transport->producer_count - transport->consumed_count;
    }
    if (pending == 0U) return 0;
    span = (uint16_t)(pending > (uint32_t)transport->size
                          ? (uint32_t)transport->size
                          : pending);

    if ((uint32_t)transport->read_index + (uint32_t)span <=
        (uint32_t)transport->size) {
        parsed += pi_link_parse_packet(&transport->buffer[transport->read_index],
                                       span);
        transport->read_index =
            ((uint32_t)transport->read_index + span == (uint32_t)transport->size)
                ? 0U
                : (uint16_t)(transport->read_index + span);
    } else {
        uint16_t first = (uint16_t)(transport->size - transport->read_index);
        parsed += pi_link_parse_packet(&transport->buffer[transport->read_index],
                                       first);
        parsed += pi_link_parse_packet(transport->buffer,
                                       (uint16_t)(span - first));
        transport->read_index = (uint16_t)(span - first);
    }
    transport->consumed_count += span;
    return parsed;
}

uint32_t pi_transport_overruns(const PiTransport_t *transport) {
    return transport == NULL ? 0U : transport->overrun_count;
}

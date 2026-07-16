#include "pi_transport.h"
#include "pi_link.h"

#include <stddef.h>

void pi_transport_init(PiTransport_t *transport, uint8_t *buffer, uint16_t size) {
    if (transport == NULL) return;
    transport->buffer = buffer;
    transport->size = size;
    transport->read_index = 0U;
}

int pi_transport_poll(PiTransport_t *transport, uint16_t dma_remaining) {
    uint16_t write_index;
    int parsed = 0;
    if (transport == NULL || transport->buffer == NULL || transport->size == 0U ||
        dma_remaining > transport->size) return -1;

    /* NDTR can briefly reach zero before circular DMA reloads it. Treat that
     * sample as a completed span to the physical end of the ring. */
    write_index = dma_remaining == 0U
                      ? transport->size
                      : (uint16_t)(transport->size - dma_remaining);
    if (write_index == transport->read_index) return 0;

    if (write_index > transport->read_index) {
        parsed += pi_link_parse_packet(&transport->buffer[transport->read_index],
                                       (uint16_t)(write_index - transport->read_index));
    } else {
        parsed += pi_link_parse_packet(&transport->buffer[transport->read_index],
                                       (uint16_t)(transport->size - transport->read_index));
        if (write_index > 0U) {
            parsed += pi_link_parse_packet(transport->buffer, write_index);
        }
    }
    transport->read_index = write_index == transport->size ? 0U : write_index;
    return parsed;
}

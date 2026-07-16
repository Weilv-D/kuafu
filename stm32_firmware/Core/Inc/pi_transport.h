#ifndef PI_TRANSPORT_H
#define PI_TRANSPORT_H

#include <stdint.h>

typedef struct {
    uint8_t *buffer;
    uint16_t size;
    uint16_t read_index;
} PiTransport_t;

void pi_transport_init(PiTransport_t *transport, uint8_t *buffer, uint16_t size);
int pi_transport_poll(PiTransport_t *transport, uint16_t dma_remaining);

#endif

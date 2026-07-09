#include "crc8.h"

static uint8_t crc8_table[256];
static int table_initialized = 0;

void crc8_init(void) {
    if (table_initialized) {
        return;
    }
    for (int i = 0; i < 256; i++) {
        uint8_t crc = (uint8_t)i;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x01) {
                crc = (crc >> 1) ^ 0x8C; /* Reflected polynomial for 0x31 is 0x8C */
            } else {
                crc >>= 1;
            }
        }
        crc8_table[i] = crc;
    }
    table_initialized = 1;
}

uint8_t crc8_update(uint8_t crc, uint8_t data) {
    if (!table_initialized) {
        crc8_init();
    }
    return crc8_table[crc ^ data];
}

uint8_t crc8_calculate(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00; /* Initial value for CRC-8/MAXIM is 0x00 */
    if (!table_initialized) {
        crc8_init();
    }
    for (size_t i = 0; i < len; i++) {
        crc = crc8_table[crc ^ data[i]];
    }
    return crc;
}

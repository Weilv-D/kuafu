#ifndef CRC8_H
#define CRC8_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Initializes the CRC8 Maxim lookup table.
 */
void crc8_init(void);

/**
 * @brief Computes the CRC-8/MAXIM checksum for a block of data.
 * 
 * @param data Pointer to the data buffer.
 * @param len Length of the data in bytes.
 * @return uint8_t Computed CRC8 checksum.
 */
uint8_t crc8_calculate(const uint8_t *data, size_t len);

/**
 * @brief Updates an existing CRC8 value with a single byte.
 * 
 * @param crc Current CRC8 value.
 * @param data Byte to update the CRC with.
 * @return uint8_t Updated CRC8 value.
 */
uint8_t crc8_update(uint8_t crc, uint8_t data);

#endif /* CRC8_H */

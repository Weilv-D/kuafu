#include "test_support.h"
#include "stm32f4xx_hal.h"

int g_test_failures = 0;
static uint32_t g_fake_time_ms = 0;
static uint32_t g_i2c_operations = 0U;
static uint8_t g_accel_id = 0x1EU;
static uint8_t g_gyro_id = 0x0FU;

void test_set_time_ms(uint32_t now_ms) {
    g_fake_time_ms = now_ms;
}

uint32_t test_get_time_ms(void) {
    return g_fake_time_ms;
}

uint32_t HAL_GetTick(void) {
    return g_fake_time_ms;
}

void test_i2c_reset(void) {
    g_i2c_operations = 0U;
    g_accel_id = 0x1EU;
    g_gyro_id = 0x0FU;
}

void test_i2c_set_chip_ids(uint8_t accel_id, uint8_t gyro_id) {
    g_accel_id = accel_id;
    g_gyro_id = gyro_id;
}

uint32_t test_i2c_operation_count(void) {
    return g_i2c_operations;
}

HAL_StatusTypeDef HAL_I2C_Mem_Write(I2C_HandleTypeDef *hi2c,
                                    uint16_t dev_address,
                                    uint16_t mem_address,
                                    uint16_t mem_address_size,
                                    uint8_t *data,
                                    uint16_t size,
                                    uint32_t timeout) {
    (void)hi2c; (void)dev_address; (void)mem_address; (void)mem_address_size;
    (void)data; (void)size; (void)timeout;
    ++g_i2c_operations;
    return HAL_OK;
}

HAL_StatusTypeDef HAL_I2C_Mem_Read(I2C_HandleTypeDef *hi2c,
                                   uint16_t dev_address,
                                   uint16_t mem_address,
                                   uint16_t mem_address_size,
                                   uint8_t *data,
                                   uint16_t size,
                                   uint32_t timeout) {
    uint16_t i;
    (void)hi2c; (void)mem_address_size; (void)timeout;
    ++g_i2c_operations;
    for (i = 0U; i < size; ++i) {
        data[i] = 0U;
    }
    if (mem_address == 0U && size == 1U) {
        data[0] = dev_address == (uint16_t)(0x18U << 1) ? g_accel_id : g_gyro_id;
    }
    return HAL_OK;
}

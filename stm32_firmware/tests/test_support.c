#include "test_support.h"
#include "stm32f4xx_hal.h"

int g_test_failures = 0;
static uint32_t g_fake_time_ms = 0;
static uint32_t g_i2c_operations = 0U;
static uint8_t g_accel_id = 0x1EU;
static uint8_t g_gyro_id = 0x0FU;
static uint8_t g_uart_tx[256];
static uint16_t g_uart_tx_size = 0U;
static uint32_t g_uart_tx_count = 0U;
static uint8_t *g_uart_rx = 0;
static uint16_t g_uart_rx_size = 0U;
static uint32_t g_uart_abort_count = 0U;

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

void test_uart_reset(void) {
    g_uart_tx_size = 0U;
    g_uart_tx_count = 0U;
    g_uart_rx = 0;
    g_uart_rx_size = 0U;
    g_uart_abort_count = 0U;
}

uint32_t test_uart_tx_count(void) { return g_uart_tx_count; }
const uint8_t *test_uart_last_tx(void) { return g_uart_tx; }
uint16_t test_uart_last_tx_size(void) { return g_uart_tx_size; }
uint32_t test_uart_abort_count(void) { return g_uart_abort_count; }

void test_uart_supply_rx(const uint8_t *data, uint16_t size) {
    uint16_t i;
    if (g_uart_rx == 0 || size > g_uart_rx_size) return;
    for (i = 0U; i < size; ++i) g_uart_rx[i] = data[i];
}

HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *huart,
                                      uint8_t *data,
                                      uint16_t size) {
    uint16_t i;
    (void)huart;
    if (size > sizeof(g_uart_tx)) return HAL_ERROR;
    for (i = 0U; i < size; ++i) g_uart_tx[i] = data[i];
    g_uart_tx_size = size;
    ++g_uart_tx_count;
    return HAL_OK;
}

HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *huart,
                                     uint8_t *data,
                                     uint16_t size) {
    (void)huart;
    g_uart_rx = data;
    g_uart_rx_size = size;
    return HAL_OK;
}

HAL_StatusTypeDef HAL_UART_Abort(UART_HandleTypeDef *huart) {
    (void)huart;
    ++g_uart_abort_count;
    g_uart_rx = 0;
    g_uart_rx_size = 0U;
    return HAL_OK;
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

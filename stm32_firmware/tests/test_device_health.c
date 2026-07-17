#include "device_health.h"
#include "test_support.h"

#include "stm32f4xx_hal.h"

#include <stdint.h>

void run_device_health_tests(void) {
    DeviceHealth_t health;

    device_health_init(&health);
    TEST_TRUE(!health.online);
    TEST_TRUE(!device_health_is_fresh(&health, 0U, 10U));
    TEST_EQ_INT(0, health.overrun_count);
    TEST_EQ_INT(0, health.noise_count);
    TEST_EQ_INT(0, health.framing_count);

    device_health_mark_valid(&health, UINT32_MAX - 5U);
    TEST_TRUE(health.online);
    TEST_TRUE(device_health_is_fresh(&health, 2U, 8U));
    TEST_TRUE(!device_health_is_fresh(&health, 3U, 8U));

    device_health_mark_failure(&health, DEVICE_FAILURE_TIMEOUT, 3U);
    device_health_mark_failure(&health, DEVICE_FAILURE_CHECKSUM, 3U);
    TEST_TRUE(health.online);
    device_health_mark_failure(&health, DEVICE_FAILURE_PROTOCOL, 3U);
    TEST_TRUE(!health.online);
    TEST_EQ_INT(3, health.consecutive_failures);
    TEST_EQ_INT(1, health.timeout_count);
    TEST_EQ_INT(1, health.checksum_count);
    TEST_EQ_INT(1, health.protocol_count);

    device_health_mark_valid(&health, 100U);
    TEST_TRUE(health.online);
    TEST_EQ_INT(0, health.consecutive_failures);

    health.timeout_count = UINT16_MAX;
    health.consecutive_failures = UINT8_MAX;
    device_health_mark_failure(&health, DEVICE_FAILURE_TIMEOUT, 0U);
    TEST_EQ_INT(UINT16_MAX, health.timeout_count);
    TEST_EQ_INT(UINT8_MAX, health.consecutive_failures);

    device_health_init(&health);
    health.online = 1U;  /* keep online for uart-error accounting tests */
    device_health_mark_uart_error(&health, 0U);
    TEST_EQ_INT(0, health.overrun_count);
    TEST_EQ_INT(0, health.noise_count);
    TEST_EQ_INT(0, health.framing_count);
    TEST_EQ_INT(0, health.protocol_count);

    /* Overrun errors (ORE). */
    device_health_init(&health);
    health.online = 1U;
    device_health_mark_uart_error(&health, HAL_UART_ERROR_ORE);
    TEST_EQ_INT(1, health.overrun_count);
    TEST_EQ_INT(0, health.noise_count);
    TEST_EQ_INT(0, health.framing_count);
    TEST_EQ_INT(1, health.protocol_count);
    TEST_EQ_INT(1, health.consecutive_failures);

    /* Noise errors (NE). */
    device_health_init(&health);
    health.online = 1U;
    device_health_mark_uart_error(&health, HAL_UART_ERROR_NE);
    TEST_EQ_INT(0, health.overrun_count);
    TEST_EQ_INT(1, health.noise_count);
    TEST_EQ_INT(0, health.framing_count);
    TEST_EQ_INT(1, health.protocol_count);

    /* Framing errors (FE). */
    device_health_init(&health);
    health.online = 1U;
    device_health_mark_uart_error(&health, HAL_UART_ERROR_FE);
    TEST_EQ_INT(0, health.overrun_count);
    TEST_EQ_INT(0, health.noise_count);
    TEST_EQ_INT(1, health.framing_count);
    TEST_EQ_INT(1, health.protocol_count);

    /* Mixed error code increments all relevant counters. */
    device_health_init(&health);
    health.online = 1U;
    device_health_mark_uart_error(&health, HAL_UART_ERROR_ORE | HAL_UART_ERROR_FE);
    TEST_EQ_INT(1, health.overrun_count);
    TEST_EQ_INT(0, health.noise_count);
    TEST_EQ_INT(1, health.framing_count);
    TEST_EQ_INT(1, health.protocol_count);

    /* Saturation at UINT16_MAX. */
    health.overrun_count = UINT16_MAX;
    device_health_mark_uart_error(&health, HAL_UART_ERROR_ORE);
    TEST_EQ_INT(UINT16_MAX, health.overrun_count);
}

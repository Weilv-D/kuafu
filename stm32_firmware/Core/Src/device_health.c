#include "device_health.h"

#include <stddef.h>

/* When compiling host tests, the full HAL may not be in the include path.
 * These constants match stm32f4xx_hal_uart.h exactly. */
#ifndef HAL_UART_ERROR_ORE
#define HAL_UART_ERROR_ORE  0x00000008U
#define HAL_UART_ERROR_NE   0x00000002U
#define HAL_UART_ERROR_FE   0x00000004U
#endif

static void increment_u16(uint16_t *value) {
    if (*value != UINT16_MAX) {
        ++(*value);
    }
}

void device_health_init(DeviceHealth_t *health) {
    if (health == NULL) {
        return;
    }
    health->last_valid_ms = 0U;
    health->timeout_count = 0U;
    health->checksum_count = 0U;
    health->protocol_count = 0U;
    health->overrun_count = 0U;
    health->noise_count = 0U;
    health->framing_count = 0U;
    health->consecutive_failures = 0U;
    health->online = 0U;
}

void device_health_mark_valid(DeviceHealth_t *health, uint32_t now_ms) {
    if (health == NULL) {
        return;
    }
    health->last_valid_ms = now_ms;
    health->consecutive_failures = 0U;
    health->online = 1U;
}

void device_health_mark_failure(DeviceHealth_t *health,
                                DeviceFailure_t failure,
                                uint8_t offline_after) {
    if (health == NULL) {
        return;
    }

    if (failure == DEVICE_FAILURE_TIMEOUT) {
        increment_u16(&health->timeout_count);
    } else if (failure == DEVICE_FAILURE_CHECKSUM) {
        increment_u16(&health->checksum_count);
    } else {
        increment_u16(&health->protocol_count);
    }

    if (health->consecutive_failures != UINT8_MAX) {
        ++health->consecutive_failures;
    }
    if (offline_after != 0U && health->consecutive_failures >= offline_after) {
        health->online = 0U;
    }
}

void device_health_mark_uart_error(DeviceHealth_t *health, uint32_t hal_error_code) {
    if (health == NULL) {
        return;
    }
    if ((hal_error_code & HAL_UART_ERROR_ORE) != 0U) {
        increment_u16(&health->overrun_count);
    }
    if ((hal_error_code & HAL_UART_ERROR_NE) != 0U) {
        increment_u16(&health->noise_count);
    }
    if ((hal_error_code & HAL_UART_ERROR_FE) != 0U) {
        increment_u16(&health->framing_count);
    }
    if ((hal_error_code & (HAL_UART_ERROR_ORE | HAL_UART_ERROR_NE | HAL_UART_ERROR_FE)) != 0U) {
        /* ORE/NE/FE are all reported as protocol errors in the existing health frame too. */
        increment_u16(&health->protocol_count);
    }
    if (health->consecutive_failures != UINT8_MAX) {
        ++health->consecutive_failures;
    }
}

uint8_t device_health_is_fresh(const DeviceHealth_t *health,
                               uint32_t now_ms,
                               uint32_t max_age_ms) {
    if (health == NULL || !health->online) {
        return 0U;
    }
    return (uint8_t)((uint32_t)(now_ms - health->last_valid_ms) <= max_age_ms);
}

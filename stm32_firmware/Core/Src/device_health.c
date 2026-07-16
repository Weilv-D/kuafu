#include "device_health.h"

#include <stddef.h>

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

uint8_t device_health_is_fresh(const DeviceHealth_t *health,
                               uint32_t now_ms,
                               uint32_t max_age_ms) {
    if (health == NULL || !health->online) {
        return 0U;
    }
    return (uint8_t)((uint32_t)(now_ms - health->last_valid_ms) <= max_age_ms);
}

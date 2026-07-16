#ifndef DEVICE_HEALTH_H
#define DEVICE_HEALTH_H

#include <stdint.h>

typedef enum {
    DEVICE_FAILURE_TIMEOUT = 0,
    DEVICE_FAILURE_CHECKSUM = 1,
    DEVICE_FAILURE_PROTOCOL = 2
} DeviceFailure_t;

typedef struct {
    uint32_t last_valid_ms;
    uint16_t timeout_count;
    uint16_t checksum_count;
    uint16_t protocol_count;
    uint8_t consecutive_failures;
    uint8_t online;
} DeviceHealth_t;

void device_health_init(DeviceHealth_t *health);
void device_health_mark_valid(DeviceHealth_t *health, uint32_t now_ms);
void device_health_mark_failure(DeviceHealth_t *health,
                                DeviceFailure_t failure,
                                uint8_t offline_after);
uint8_t device_health_is_fresh(const DeviceHealth_t *health,
                               uint32_t now_ms,
                               uint32_t max_age_ms);

#endif

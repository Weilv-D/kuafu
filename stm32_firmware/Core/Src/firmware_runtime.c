#include "firmware_runtime.h"

#include <stddef.h>

#define CONTROL_PERIOD_MS 4U
#define SERVO_PERIOD_MS  20U

static void increment_u16(uint16_t *value) {
    if (*value != UINT16_MAX) ++(*value);
}

void firmware_runtime_init(FirmwareRuntime_t *runtime, uint32_t now_ms) {
    if (runtime == NULL) return;
    runtime->last_control_ms = now_ms;
    runtime->last_servo_ms = now_ms;
    runtime->wheel_busy_cycles = 0U;
    runtime->servo_busy_cycles = 0U;
}

FirmwareRuntimeOutputs_t firmware_runtime_step(FirmwareRuntime_t *runtime,
                                               const FirmwareRuntimeInputs_t *inputs) {
    FirmwareRuntimeOutputs_t outputs = {0U, 0U, 0U, 0U, 0U};
    uint8_t operational;
    if (runtime == NULL || inputs == NULL) return outputs;

    if ((uint32_t)(inputs->now_ms - runtime->last_control_ms) >= CONTROL_PERIOD_MS) {
        runtime->last_control_ms = inputs->now_ms;
        outputs.control_due = 1U;
        if (!inputs->wheel_bus_idle) increment_u16(&runtime->wheel_busy_cycles);
    }
    if ((uint32_t)(inputs->now_ms - runtime->last_servo_ms) >= SERVO_PERIOD_MS) {
        runtime->last_servo_ms = inputs->now_ms;
        outputs.servo_due = 1U;
        if (!inputs->servo_bus_idle) increment_u16(&runtime->servo_busy_cycles);
    }

    operational = (uint8_t)(inputs->mode != STATE_INIT && inputs->mode != STATE_FAULT);
    outputs.wheel_intent_allowed = (uint8_t)(operational && inputs->wheel_authorized &&
                                             inputs->wheel_bus_idle);
    outputs.servo_intent_allowed = (uint8_t)(operational && inputs->servo_bus_idle);
    outputs.residual_allowed = (uint8_t)(inputs->mode == STATE_ACTIVE &&
                                         inputs->link_compatible &&
                                         inputs->heartbeat_fresh &&
                                         inputs->action_fresh);
    return outputs;
}

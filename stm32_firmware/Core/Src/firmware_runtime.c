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
    FirmwareRuntimeOutputs_t outputs = {0U, 0U, 0U, 0U, 0U, 0U};
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
    /* wheel_intent_allowed gates the LQR balance computation itself, which must
     * run every control deadline regardless of whether the DDSM bus happens to
     * be mid-transaction at that instant. The dispatch layer already skips
     * sending when the bus is busy; gating the computation on bus idle starves
     * the controller and the robot cannot balance.
     * servo_intent_allowed is the mode-level mirror for the 50 Hz leg writer:
     * intent is a MODE verdict only.  Bus arbitration happens at queue time —
     * the scheduler retries a busy-refused write on its next pass instead of
     * dropping the whole 20 ms period (see the servo deadline block in
     * main.c) — so sampling servo_bus_idle here would re-introduce exactly
     * the drop-on-busy behavior the retry removed. */
    outputs.wheel_intent_allowed = (uint8_t)(operational && inputs->wheel_authorized);
    outputs.servo_intent_allowed = operational;
    outputs.residual_allowed = (uint8_t)(inputs->mode == STATE_ACTIVE &&
                                         inputs->link_compatible &&
                                         inputs->heartbeat_fresh &&
                                         inputs->action_fresh);
    /* NOTE: cached-Pi-motion clearing is owned by the safety state machine
     * (its clear_action / enter_hold decisions call pi_link_clear_action /
     * pi_link_enter_hold from the control section).  The runtime deliberately
     * exports no duplicate of that verdict: an unconsumed copy here was dead
     * logic that could drift from the real path. */
    outputs.velocity_command_active = (uint8_t)(inputs->mode == STATE_ACTIVE);
    return outputs;
}

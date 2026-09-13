#ifndef FIRMWARE_RUNTIME_H
#define FIRMWARE_RUNTIME_H

#include "safety_state.h"
#include <stdint.h>

typedef struct {
    uint32_t now_ms;
    RobotMode_t mode;
    uint8_t link_compatible;
    uint8_t heartbeat_fresh;
    uint8_t action_fresh;
    uint8_t wheel_authorized;
    uint8_t wheel_bus_idle;
    uint8_t servo_bus_idle;
} FirmwareRuntimeInputs_t;

typedef struct {
    uint8_t control_due;
    uint8_t servo_due;
    uint8_t wheel_intent_allowed;
    uint8_t servo_intent_allowed;
    uint8_t residual_allowed;
    /* Clear cached Pi motion on stale heartbeat/action; STAND remains valid
     * without a Pi and is intentionally not cleared by link absence alone. */
    uint8_t clear_motion;
    /* Base-layer velocity/yaw commands are an ACTIVE-mode contract: STAND is a
     * position hold and CLIMB is height-only.  The Pi heartbeat carries its
     * latest vx/wz regardless of requested mode, so the runtime tells the
     * controller when those fields are meaningful instead of trusting the
     * sender to zero them. */
    uint8_t velocity_command_active;
} FirmwareRuntimeOutputs_t;

typedef struct {
    uint32_t last_control_ms;
    uint32_t last_servo_ms;
    uint16_t wheel_busy_cycles;
    uint16_t servo_busy_cycles;
} FirmwareRuntime_t;

void firmware_runtime_init(FirmwareRuntime_t *runtime, uint32_t now_ms);
FirmwareRuntimeOutputs_t firmware_runtime_step(FirmwareRuntime_t *runtime,
                                               const FirmwareRuntimeInputs_t *inputs);

#endif

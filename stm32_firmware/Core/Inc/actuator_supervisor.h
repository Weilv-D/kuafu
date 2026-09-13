#ifndef ACTUATOR_SUPERVISOR_H
#define ACTUATOR_SUPERVISOR_H

#include "safety_state.h"
#include <stdint.h>

typedef enum {
    ACTUATOR_SUPERVISOR_STARTUP = 0,
    ACTUATOR_SUPERVISOR_HOLDING = 1,
    ACTUATOR_SUPERVISOR_READY = 2,
    ACTUATOR_SUPERVISOR_LATCHED = 3
} ActuatorSupervisorPhase_t;

typedef enum {
    ACTUATOR_OP_OK = 0,
    ACTUATOR_OP_BUSY = 1,
    ACTUATOR_OP_ERROR = -1
} ActuatorSupervisorOpResult_t;

typedef struct {
    uint32_t now_ms;
    RobotMode_t mode;
    FaultMask_t fault_mask;
    uint8_t startup_ready;
    uint8_t actuator_configured;
    uint8_t link_compatible;
    uint8_t heartbeat_fresh;
    uint8_t wheel_authorized;
    uint8_t wheel_bus_idle;
    uint8_t servo_bus_idle;
    /* Recovery is valid only after the leg target was transmitted and a new,
     * fresh position report confirms a safe posture. */
    uint8_t leg_target_tx_complete;
    uint8_t leg_feedback_fresh;
    uint8_t leg_posture_safe;
    uint8_t servo_enable_verified;
    uint8_t wheel_enable_verified;
    /* Compatibility aggregate for simple integrations/tests. */
    uint8_t leg_target_held;
    /* Set by the bus layer when the last queued operation failed. */
    uint8_t tx_failed;
} ActuatorSupervisorInputs_t;

typedef struct {
    int (*queue_leg_hold)(void *ctx);
    int (*queue_servo_enable)(void *ctx, uint8_t enable);
    int (*queue_wheel_enable)(void *ctx, uint8_t enable);
    int (*queue_wheel_zero)(void *ctx);
    void *ctx;
} ActuatorSupervisorOps_t;

typedef struct {
    uint8_t leg_motion_allowed;
    uint8_t wheel_output_allowed;
    uint8_t clear_motion;
    uint8_t request_hold;
    uint8_t request_wheel_disable;
    ActuatorSupervisorPhase_t phase;
} ActuatorSupervisorOutputs_t;

typedef struct {
    ActuatorSupervisorPhase_t phase;
    uint8_t wheel_enabled;
    uint8_t servos_enabled;
    uint8_t hold_requested;
    uint8_t zero_requested;
    /* Set on every entry to STARTUP/HOLDING: physical servo enable does not
     * survive a fault or restart, so recovery must re-issue it (and observe
     * the integration's verified flag on a LATER step) before any wheel
     * output, even when the verified input still reads true. */
    uint8_t enable_reissue_needed;
    /* ops-less integration support: records that the servo-enable verified
     * flag dropped since the re-issue became due.  Physical enables do not
     * survive a fault/restart, so the flag MUST drop; authorizing on a flag
     * that never dropped would trust a stale "verified". */
    uint8_t enable_verified_observed_low;
} ActuatorSupervisor_t;

void actuator_supervisor_init(ActuatorSupervisor_t *supervisor);
ActuatorSupervisorOutputs_t actuator_supervisor_step(
    ActuatorSupervisor_t *supervisor,
    const ActuatorSupervisorInputs_t *inputs,
    const ActuatorSupervisorOps_t *ops);

#endif

#include "actuator_supervisor.h"

#include <stddef.h>

void actuator_supervisor_init(ActuatorSupervisor_t *supervisor) {
    if (supervisor == NULL) return;
    supervisor->phase = ACTUATOR_SUPERVISOR_STARTUP;
    supervisor->wheel_enabled = 0U;
    supervisor->servos_enabled = 0U;
    supervisor->hold_requested = 0U;
    supervisor->zero_requested = 0U;
    supervisor->enable_reissue_needed = 1U;
    supervisor->enable_verified_observed_low = 0U;
}

ActuatorSupervisorOutputs_t actuator_supervisor_step(
    ActuatorSupervisor_t *supervisor,
    const ActuatorSupervisorInputs_t *inputs,
    const ActuatorSupervisorOps_t *ops) {
    ActuatorSupervisorOutputs_t out = {0U, 0U, 0U, 0U, 0U,
                                       ACTUATOR_SUPERVISOR_STARTUP};
    int result;
    uint8_t faulted;
    uint8_t severe;
    uint8_t legs_ready;
    uint8_t motion_link_bad;
    if (supervisor == NULL || inputs == NULL) return out;
    motion_link_bad = (uint8_t)(inputs->mode == STATE_ACTIVE &&
                                (!inputs->link_compatible || !inputs->heartbeat_fresh));

    faulted = (uint8_t)(inputs->fault_mask != FAULT_NONE ||
                        inputs->mode == STATE_FAULT);
    severe = (uint8_t)((inputs->fault_mask & SAFETY_SERIOUS_FAULTS) != 0U);
    legs_ready = (uint8_t)(inputs->leg_target_tx_complete &&
                           inputs->leg_feedback_fresh &&
                           inputs->leg_posture_safe &&
                           inputs->leg_target_held);
    /* Integrations that own the physical enable sequence report completion;
     * internal flags mirror verified state, never a merely queued command. */
    supervisor->servos_enabled = inputs->servo_enable_verified;
    supervisor->wheel_enabled = inputs->wheel_enable_verified;
    /* Record "verified observed low" on EVERY step, including the early-return
     * paths below.  The scheduler couples startup_ready and servo_enable_verified
     * to the same sequencer variable, so at boot the flag is low exactly while
     * the pre-READY HOLDING/STARTUP branches return early; without this latch
     * those observations are lost and enable_reissue_needed can never clear
     * (the gate would stay closed until some fault happens to reset the
     * sequencer). */
    if (!inputs->servo_enable_verified) {
        supervisor->enable_verified_observed_low = 1U;
    }
    if (severe || inputs->tx_failed) {
        supervisor->phase = ACTUATOR_SUPERVISOR_LATCHED;
    }

    if (supervisor->phase == ACTUATOR_SUPERVISOR_LATCHED) {
        out.clear_motion = 1U;
        out.request_wheel_disable = 1U;
        out.request_hold = 1U;
        if (!supervisor->zero_requested && ops != NULL && ops->queue_wheel_zero != NULL &&
            inputs->wheel_bus_idle) {
            result = ops->queue_wheel_zero(ops->ctx);
            if (result == ACTUATOR_OP_OK) supervisor->zero_requested = 1U;
        }
        if (supervisor->wheel_enabled && ops != NULL && ops->queue_wheel_enable != NULL &&
            inputs->wheel_bus_idle) {
            result = ops->queue_wheel_enable(ops->ctx, 0U);
            if (result == ACTUATOR_OP_OK) supervisor->wheel_enabled = 0U;
        }
        if (supervisor->servos_enabled && ops != NULL && ops->queue_servo_enable != NULL &&
            inputs->servo_bus_idle) {
            result = ops->queue_servo_enable(ops->ctx, 0U);
            if (result == ACTUATOR_OP_OK) supervisor->servos_enabled = 0U;
        }
        out.phase = supervisor->phase;
        return out;
    }

    /* Every fault, including a transient bus/feedback fault, revokes output
     * immediately. Recovery is permitted only after a fresh hold is observed;
     * a wheel command or enable must never survive the fault transition. */
    if (faulted || !inputs->wheel_authorized || motion_link_bad) {
        /* The drop-requirement is armed only when a REVOCATION follows an
         * authorized state (READY): that is the case where physical enables
         * may have been lost and a stale "verified" must not be trusted.
         * Boot-time entries (STARTUP/HOLDING before first authorization) have
         * nothing to revoke -- clearing there would discard the low-state
         * observations the boot coupling produced and deadlock the re-issue
         * stage (see the latch at the top of this function). */
        if (supervisor->phase == ACTUATOR_SUPERVISOR_READY) {
            supervisor->enable_verified_observed_low = 0U;
        }
        supervisor->phase = ACTUATOR_SUPERVISOR_HOLDING;
        supervisor->hold_requested = 0U;
        supervisor->zero_requested = 0U;
        supervisor->enable_reissue_needed = 1U;
        out.clear_motion = 1U;
        out.request_hold = 1U;
        out.request_wheel_disable = 1U;
        if (ops != NULL && ops->queue_leg_hold != NULL && inputs->servo_bus_idle) {
            result = ops->queue_leg_hold(ops->ctx);
            if (result == ACTUATOR_OP_OK) supervisor->hold_requested = 1U;
        }
        if (!supervisor->zero_requested && ops != NULL && ops->queue_wheel_zero != NULL &&
            inputs->wheel_bus_idle) {
            result = ops->queue_wheel_zero(ops->ctx);
            if (result == ACTUATOR_OP_OK) supervisor->zero_requested = 1U;
        }
        if (supervisor->wheel_enabled && ops != NULL && ops->queue_wheel_enable != NULL &&
            inputs->wheel_bus_idle) {
            result = ops->queue_wheel_enable(ops->ctx, 0U);
            if (result == ACTUATOR_OP_OK) supervisor->wheel_enabled = 0U;
        }
        out.phase = supervisor->phase;
        return out;
    }

    if (!inputs->startup_ready || !inputs->actuator_configured) {
        if (supervisor->phase == ACTUATOR_SUPERVISOR_READY) {
            /* Demotion from READY re-arms the drop-requirement (a real
             * post-authorization revocation); boot-time entries do not. */
            supervisor->enable_verified_observed_low = 0U;
        }
        supervisor->phase = ACTUATOR_SUPERVISOR_STARTUP;
        supervisor->enable_reissue_needed = 1U;
        out.clear_motion = 1U;
        out.request_hold = 1U;
        out.phase = supervisor->phase;
        return out;
    }

    if (supervisor->phase == ACTUATOR_SUPERVISOR_STARTUP ||
        supervisor->phase == ACTUATOR_SUPERVISOR_HOLDING) {
        /* First transmit the target/hold posture.  Servo enable is deliberately
         * not gated on already reaching that target: disabled servos cannot
         * produce the feedback needed to reach it. */
        if (!inputs->leg_target_tx_complete) {
            out.request_hold = 1U;
            out.clear_motion = 1U;
            if (ops != NULL && ops->queue_leg_hold != NULL && inputs->servo_bus_idle) {
                result = ops->queue_leg_hold(ops->ctx);
                if (result == ACTUATOR_OP_ERROR) supervisor->phase = ACTUATOR_SUPERVISOR_LATCHED;
            }
            out.phase = supervisor->phase;
            return out;
        }
        /* Physical servo torque does not survive a fault/restart.  Re-issue
         * the enable once per recovery — via the ops executor when wired,
         * otherwise by observing the integration's own sequencer — and
         * REQUIRE a later step to observe the verified flag: a stale
         * "verified" input must never shortcut this stage.  Queued is not
         * completed. */
        if (supervisor->enable_reissue_needed || !inputs->servo_enable_verified) {
            out.request_hold = 1U;
            out.clear_motion = 1U;
            if (ops != NULL && ops->queue_servo_enable != NULL && inputs->servo_bus_idle) {
                result = ops->queue_servo_enable(ops->ctx, 1U);
                if (result == ACTUATOR_OP_ERROR) {
                    supervisor->phase = ACTUATOR_SUPERVISOR_LATCHED;
                    out.phase = supervisor->phase;
                    return out;
                }
                if (result == ACTUATOR_OP_OK) supervisor->enable_reissue_needed = 0U;
            } else {
                /* ops-less integration: the external sequencer owns the
                 * enable frames.  Because physical enables do not survive a
                 * fault/restart, the verified flag must first DROP and then
                 * return on a later step before authorization proceeds. */
                if (!inputs->servo_enable_verified) {
                    supervisor->enable_verified_observed_low = 1U;
                } else if (supervisor->enable_verified_observed_low) {
                    supervisor->enable_reissue_needed = 0U;
                }
            }
            out.phase = supervisor->phase;
            return out;
        }
        supervisor->servos_enabled = 1U;
        if (!legs_ready) {
            out.request_hold = 1U;
            out.clear_motion = 1U;
            out.phase = supervisor->phase;
            return out;
        }
        supervisor->phase = ACTUATOR_SUPERVISOR_READY;
    }

    if (supervisor->phase == ACTUATOR_SUPERVISOR_READY) {
        if (!legs_ready || !inputs->leg_feedback_fresh || !inputs->leg_posture_safe ||
            !inputs->servo_enable_verified) {
            /* A transient readiness gap (e.g. one missed leg-hold window) is
             * NOT an enable-lifecycle restart: the physical torque enables
             * never dropped, so the verified flag is still trustworthy and the
             * re-issue stage must not wait for a drop that will never come.
             * Persistent gaps escalate to a latched FAULT in the safety layer,
             * which is the path that re-arms the drop-requirement. */
            supervisor->phase = ACTUATOR_SUPERVISOR_HOLDING;
            supervisor->enable_reissue_needed = 1U;
            out.request_hold = 1U;
            out.clear_motion = 1U;
            out.phase = supervisor->phase;
            return out;
        }
        if (!inputs->wheel_enable_verified && inputs->wheel_bus_idle &&
            ops != NULL && ops->queue_wheel_enable != NULL) {
            result = ops->queue_wheel_enable(ops->ctx, 1U);
            if (result == ACTUATOR_OP_ERROR) {
                supervisor->phase = ACTUATOR_SUPERVISOR_LATCHED;
                out.clear_motion = 1U;
                out.phase = supervisor->phase;
                return out;
            }
            /* Do not set wheel_enabled until the bus reports verified enable. */
        }
        /* Critical ordering: target TX, servo enable completion, fresh
         * feedback, safe posture, then verified wheel enable. */
        out.leg_motion_allowed = (uint8_t)(supervisor->servos_enabled &&
                                            inputs->servo_enable_verified);
        out.wheel_output_allowed = (uint8_t)(supervisor->wheel_enabled &&
                                              inputs->wheel_enable_verified &&
                                              supervisor->servos_enabled &&
                                              inputs->servo_enable_verified &&
                                              legs_ready);
    }
    out.phase = supervisor->phase;
    return out;
}

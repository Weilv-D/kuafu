#include "startup_manager.h"

#include <stddef.h>

static uint8_t deadline_reached(uint32_t now_ms, uint32_t deadline_ms) {
    return (uint8_t)((int32_t)(now_ms - deadline_ms) >= 0);
}

static void enter_phase(StartupManager_t *manager, StartupPhase_t phase, uint32_t now_ms) {
    manager->phase = phase;
    manager->phase_started_ms = now_ms;
    manager->next_action_ms = now_ms;
}

static StartupOutputs_t failed_outputs(StartupManager_t *manager,
                                       StartupFailureReason_t reason) {
    StartupOutputs_t outputs = {0U, 0U, 0U, 1U, reason};
    manager->phase = STARTUP_FAILED;
    manager->failure_reason = reason;
    return outputs;
}

void startup_manager_init(StartupManager_t *manager, uint32_t now_ms) {
    if (manager == NULL) return;
    manager->phase = STARTUP_WAIT_POWER;
    manager->failure_reason = STARTUP_FAILURE_NONE;
    manager->phase_started_ms = now_ms;
    manager->next_action_ms = now_ms + STARTUP_POWER_WAIT_MS;
    manager->startup_started_ms = now_ms;
}

StartupOutputs_t startup_manager_step(StartupManager_t *manager,
                                      const StartupInputs_t *inputs) {
    StartupOutputs_t outputs = {0U, 0U, 0U, 0U, STARTUP_FAILURE_NONE};
    if (manager == NULL || inputs == NULL) {
        outputs.fault_requested = 1U;
        outputs.failure_reason = STARTUP_FAILURE_INVALID_INPUT;
        return outputs;
    }
    if (manager->phase == STARTUP_FAILED) {
        outputs.fault_requested = 1U;
        outputs.failure_reason = manager->failure_reason;
        return outputs;
    }
    if (manager->phase != STARTUP_READY &&
        (uint32_t)(inputs->now_ms - manager->startup_started_ms) >= STARTUP_TOTAL_TIMEOUT_MS) {
        if (manager->phase == STARTUP_WAIT_POWER || manager->phase == STARTUP_IMU_DISCOVERY) {
            return failed_outputs(manager, STARTUP_FAILURE_IMU_TIMEOUT);
        }
        if (manager->phase == STARTUP_ACTUATOR_DISCOVERY) {
            return failed_outputs(manager, STARTUP_FAILURE_ACTUATOR_TIMEOUT);
        }
    }
    if (manager->phase == STARTUP_WAIT_POWER) {
        if (!deadline_reached(inputs->now_ms, manager->next_action_ms)) return outputs;
        enter_phase(manager, STARTUP_IMU_DISCOVERY, inputs->now_ms);
    }
    if (manager->phase == STARTUP_IMU_DISCOVERY) {
        /* Calibration is intentionally not a startup gate: zero gyro bias is a
         * valid autonomous STAND starting point and may be refined later. */
        if (inputs->imu_initialized && inputs->accel_valid && inputs->gyro_valid) {
            enter_phase(manager, STARTUP_ACTUATOR_DISCOVERY, inputs->now_ms);
            return outputs;
        }
        if (deadline_reached(inputs->now_ms, manager->next_action_ms)) {
            outputs.request_imu_init = 1U;
            manager->next_action_ms = inputs->now_ms + STARTUP_RETRY_MS;
        }
        return outputs;
    }
    if (manager->phase == STARTUP_ACTUATOR_DISCOVERY) {
        if (inputs->actuator_configured && inputs->wheel_l_online &&
            inputs->wheel_r_online && inputs->servos_online) {
            enter_phase(manager, STARTUP_READY, inputs->now_ms);
            outputs.enable_actuators = 1U;
            return outputs;
        }
        if (deadline_reached(inputs->now_ms, manager->next_action_ms)) {
            outputs.request_actuator_discovery = 1U;
            manager->next_action_ms = inputs->now_ms + STARTUP_RETRY_MS;
        }
        return outputs;
    }
    if (manager->phase == STARTUP_READY) {
        outputs.enable_actuators = 1U;
        return outputs;
    }
    return failed_outputs(manager, STARTUP_FAILURE_INVALID_INPUT);
}

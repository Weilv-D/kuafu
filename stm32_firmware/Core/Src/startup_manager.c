#include "startup_manager.h"

#include <stddef.h>

#define STARTUP_POWER_WAIT_MS          500U
#define STARTUP_RETRY_MS               100U
#define STARTUP_IMU_TIMEOUT_MS        5000U
#define STARTUP_GYRO_TIMEOUT_MS       5000U
#define STARTUP_ACTUATOR_TIMEOUT_MS  10000U

static uint8_t deadline_reached(uint32_t now_ms, uint32_t deadline_ms) {
    return (uint8_t)((int32_t)(now_ms - deadline_ms) >= 0);
}

static void enter_phase(StartupManager_t *manager,
                        StartupPhase_t phase,
                        uint32_t now_ms) {
    manager->phase = phase;
    manager->phase_started_ms = now_ms;
    manager->next_action_ms = now_ms;
}

void startup_manager_init(StartupManager_t *manager, uint32_t now_ms) {
    if (manager == NULL) {
        return;
    }
    manager->phase = STARTUP_WAIT_POWER;
    manager->phase_started_ms = now_ms;
    manager->next_action_ms = now_ms + STARTUP_POWER_WAIT_MS;
}

StartupOutputs_t startup_manager_step(StartupManager_t *manager,
                                      const StartupInputs_t *inputs) {
    StartupOutputs_t outputs = {0U, 0U, 0U, 0U};
    uint32_t elapsed;

    if (manager == NULL || inputs == NULL) {
        outputs.fault_requested = 1U;
        return outputs;
    }

    if (manager->phase == STARTUP_WAIT_POWER) {
        if (deadline_reached(inputs->now_ms, manager->next_action_ms)) {
            enter_phase(manager, STARTUP_IMU_DISCOVERY, inputs->now_ms);
        } else {
            return outputs;
        }
    }

    elapsed = (uint32_t)(inputs->now_ms - manager->phase_started_ms);
    if (manager->phase == STARTUP_IMU_DISCOVERY) {
        if (inputs->imu_initialized) {
            enter_phase(manager, STARTUP_GYRO_CALIBRATION, inputs->now_ms);
            return outputs;
        }
        if (elapsed > STARTUP_IMU_TIMEOUT_MS) {
            enter_phase(manager, STARTUP_FAILED, inputs->now_ms);
            outputs.fault_requested = 1U;
            return outputs;
        }
        if (deadline_reached(inputs->now_ms, manager->next_action_ms)) {
            outputs.request_imu_init = 1U;
            manager->next_action_ms = inputs->now_ms + STARTUP_RETRY_MS;
        }
        return outputs;
    }

    if (manager->phase == STARTUP_GYRO_CALIBRATION) {
        if (inputs->gyro_calibrated) {
            enter_phase(manager, STARTUP_ACTUATOR_DISCOVERY, inputs->now_ms);
        } else if (elapsed > STARTUP_GYRO_TIMEOUT_MS) {
            enter_phase(manager, STARTUP_FAILED, inputs->now_ms);
            outputs.fault_requested = 1U;
            return outputs;
        } else {
            return outputs;
        }
    }

    elapsed = (uint32_t)(inputs->now_ms - manager->phase_started_ms);
    if (manager->phase == STARTUP_ACTUATOR_DISCOVERY) {
        if (inputs->actuator_configured && inputs->wheel_l_online &&
            inputs->wheel_r_online && inputs->servos_online) {
            enter_phase(manager, STARTUP_READY, inputs->now_ms);
            outputs.enable_actuators = 1U;
            return outputs;
        }
        if (elapsed > STARTUP_ACTUATOR_TIMEOUT_MS) {
            enter_phase(manager, STARTUP_FAILED, inputs->now_ms);
            outputs.fault_requested = 1U;
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
    } else if (manager->phase == STARTUP_FAILED) {
        outputs.fault_requested = 1U;
    } else {
        enter_phase(manager, STARTUP_FAILED, inputs->now_ms);
        outputs.fault_requested = 1U;
    }
    return outputs;
}

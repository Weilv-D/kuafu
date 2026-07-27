#include "startup_manager.h"

#include <stddef.h>

#define STARTUP_POWER_WAIT_MS          500U
#define STARTUP_RETRY_MS               100U

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

    if (manager->phase == STARTUP_IMU_DISCOVERY) {
        if (inputs->imu_initialized) {
            enter_phase(manager, STARTUP_GYRO_CALIBRATION, inputs->now_ms);
            return outputs;
        }
        /* No hard failure: a transient (e.g. I2C busy at power-on) must not
         * permanently disable the robot.  Keep re-requesting discovery until
         * the IMU answers. */
        if (deadline_reached(inputs->now_ms, manager->next_action_ms)) {
            outputs.request_imu_init = 1U;
            manager->next_action_ms = inputs->now_ms + STARTUP_RETRY_MS;
        }
        return outputs;
    }

    if (manager->phase == STARTUP_GYRO_CALIBRATION) {
        /* Calibration only completes once the robot is still long enough to
         * accumulate GYRO_CALIB_SAMPLES quiet samples.  If the operator is
         * still placing / settling the robot, just keep waiting -- a moving
         * platform must never latch a fatal fault.  Proceed as soon as done. */
        if (inputs->gyro_calibrated) {
            enter_phase(manager, STARTUP_ACTUATOR_DISCOVERY, inputs->now_ms);
        }
        /* Fall through to the ACTUATOR_DISCOVERY handler so the first discovery
         * request is issued on the same tick. */
    }

    if (manager->phase == STARTUP_ACTUATOR_DISCOVERY) {
        if (inputs->actuator_configured && inputs->wheel_l_online &&
            inputs->wheel_r_online && inputs->servos_online) {
            enter_phase(manager, STARTUP_READY, inputs->now_ms);
            outputs.enable_actuators = 1U;
            return outputs;
        }
        /* Likewise, actuators coming online after a cold power-up (servos/wheels
         * need a moment to answer) must not latch a permanent FAULT.  Keep
         * re-requesting discovery until everything is present. */
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

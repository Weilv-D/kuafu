#ifndef STARTUP_MANAGER_H
#define STARTUP_MANAGER_H

#include <stdint.h>

typedef enum {
    STARTUP_WAIT_POWER = 0,
    STARTUP_IMU_DISCOVERY = 1,
    /* Value 2 is retired (the old gyro-calibration gate; calibration now
     * accumulates in the background and never gates startup).  The numeric
     * gap is kept deliberately so ACTUATOR_DISCOVERY/READY/FAILED retain the
     * values the balance trace's startup_state encoding and the SWD dump
     * tools were built against. */
    STARTUP_ACTUATOR_DISCOVERY = 3,
    STARTUP_READY = 4,
    STARTUP_FAILED = 5
} StartupPhase_t;

typedef enum {
    STARTUP_FAILURE_NONE = 0,
    STARTUP_FAILURE_INVALID_INPUT = 1,
    STARTUP_FAILURE_IMU_TIMEOUT = 2,
    STARTUP_FAILURE_ACTUATOR_TIMEOUT = 3
} StartupFailureReason_t;

typedef struct {
    uint32_t now_ms;
    uint8_t imu_initialized;
    uint8_t accel_valid;
    uint8_t gyro_valid;
    /* Gyro bias calibration is deliberately absent: it accumulates in the
     * background (safety_state_gyro_calib_update) and never gates startup. */
    uint8_t wheel_l_online;
    uint8_t wheel_r_online;
    uint8_t servos_online;
    uint8_t actuator_configured;
} StartupInputs_t;

typedef struct {
    uint8_t request_imu_init;
    uint8_t request_actuator_discovery;
    uint8_t enable_actuators;
    uint8_t fault_requested;
    StartupFailureReason_t failure_reason;
} StartupOutputs_t;

typedef struct {
    StartupPhase_t phase;
    StartupFailureReason_t failure_reason;
    uint32_t phase_started_ms;
    uint32_t next_action_ms;
    uint32_t startup_started_ms;
} StartupManager_t;

#define STARTUP_POWER_WAIT_MS 500U
#define STARTUP_RETRY_MS 100U
#define STARTUP_TOTAL_TIMEOUT_MS 15000U

void startup_manager_init(StartupManager_t *manager, uint32_t now_ms);
StartupOutputs_t startup_manager_step(StartupManager_t *manager,
                                      const StartupInputs_t *inputs);

#endif

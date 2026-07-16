#ifndef STARTUP_MANAGER_H
#define STARTUP_MANAGER_H

#include <stdint.h>

typedef enum {
    STARTUP_WAIT_POWER = 0,
    STARTUP_IMU_DISCOVERY = 1,
    STARTUP_GYRO_CALIBRATION = 2,
    STARTUP_ACTUATOR_DISCOVERY = 3,
    STARTUP_READY = 4,
    STARTUP_FAILED = 5
} StartupPhase_t;

typedef struct {
    uint32_t now_ms;
    uint8_t imu_initialized;
    uint8_t gyro_calibrated;
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
} StartupOutputs_t;

typedef struct {
    StartupPhase_t phase;
    uint32_t phase_started_ms;
    uint32_t next_action_ms;
} StartupManager_t;

void startup_manager_init(StartupManager_t *manager, uint32_t now_ms);
StartupOutputs_t startup_manager_step(StartupManager_t *manager,
                                      const StartupInputs_t *inputs);

#endif

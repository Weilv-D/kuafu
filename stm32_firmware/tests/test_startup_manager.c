#include "startup_manager.h"
#include "test_support.h"

#include <string.h>

void run_startup_manager_tests(void) {
    StartupManager_t manager;
    StartupInputs_t inputs;
    StartupOutputs_t outputs;

    memset(&inputs, 0, sizeof(inputs));
    startup_manager_init(&manager, 100U);
    inputs.now_ms = 599U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_WAIT_POWER, manager.phase);
    TEST_TRUE(!outputs.request_imu_init);

    inputs.now_ms = 600U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_IMU_DISCOVERY, manager.phase);
    TEST_TRUE(outputs.request_imu_init);

    inputs.now_ms = 650U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_TRUE(!outputs.request_imu_init);
    inputs.now_ms = 700U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_TRUE(outputs.request_imu_init);

    inputs.imu_initialized = 1U;
    inputs.accel_valid = 1U;
    inputs.gyro_valid = 1U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_ACTUATOR_DISCOVERY, manager.phase);

    inputs.now_ms = 900U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_ACTUATOR_DISCOVERY, manager.phase);
    TEST_TRUE(outputs.request_actuator_discovery);

    inputs.wheel_l_online = 1U;
    inputs.wheel_r_online = 1U;
    inputs.servos_online = 1U;
    inputs.actuator_configured = 1U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_READY, manager.phase);
    TEST_TRUE(outputs.enable_actuators);

    startup_manager_init(&manager, 0U);
    inputs = (StartupInputs_t){0};
    inputs.now_ms = 5600U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_IMU_DISCOVERY, manager.phase);
    TEST_TRUE(outputs.request_imu_init);
    /* A transient at power-on (IMU slow to answer) must NOT latch FAULT: the
     * manager keeps re-requesting discovery instead of failing. */
    inputs.now_ms = 10601U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_IMU_DISCOVERY, manager.phase);
    TEST_TRUE(!outputs.fault_requested);

    /* Persistent IMU failure is diagnosed and remains latched. */
    startup_manager_init(&manager, 0U);
    inputs = (StartupInputs_t){0};
    inputs.now_ms = 15500U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_EQ_INT(STARTUP_FAILED, manager.phase);
    TEST_TRUE(outputs.fault_requested);
    TEST_EQ_INT(STARTUP_FAILURE_IMU_TIMEOUT, outputs.failure_reason);
    inputs.now_ms = 20000U;
    outputs = startup_manager_step(&manager, &inputs);
    TEST_TRUE(outputs.fault_requested);
    TEST_EQ_INT(STARTUP_FAILURE_IMU_TIMEOUT, outputs.failure_reason);
}

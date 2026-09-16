#include "firmware_runtime.h"
#include "test_support.h"

#include <string.h>

void run_firmware_runtime_tests(void) {
    FirmwareRuntime_t runtime;
    FirmwareRuntimeInputs_t inputs;
    FirmwareRuntimeOutputs_t outputs;

    memset(&inputs, 0, sizeof(inputs));
    inputs.mode = STATE_ACTIVE;
    inputs.link_compatible = 1U;
    inputs.heartbeat_fresh = 1U;
    inputs.action_fresh = 1U;
    inputs.wheel_authorized = 1U;
    inputs.wheel_bus_idle = 1U;
    inputs.servo_bus_idle = 1U;
    firmware_runtime_init(&runtime, 0U);

    inputs.now_ms = 3U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.control_due);
    inputs.now_ms = 4U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(outputs.control_due);
    TEST_TRUE(outputs.wheel_intent_allowed);
    TEST_TRUE(outputs.residual_allowed);

    /* A skipped deadline produces one update, never a catch-up burst. */
    inputs.now_ms = 17U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(outputs.control_due);
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.control_due);

    inputs.now_ms = 20U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(outputs.servo_due);

    inputs.action_fresh = 0U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.residual_allowed);
    /* Stale-action motion clearing is owned by the safety state machine
     * (clear_action/enter_hold); the runtime exports no duplicate verdict. */
    TEST_TRUE(outputs.wheel_intent_allowed);

    inputs.wheel_authorized = 0U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.wheel_intent_allowed);
    inputs.wheel_authorized = 1U;

    inputs.wheel_bus_idle = 0U;
    inputs.servo_bus_idle = 0U;
    inputs.now_ms = 40U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(outputs.control_due);
    TEST_TRUE(outputs.servo_due);
    /* Neither intent gates on bus idle: the LQR must compute every deadline
     * and the leg writer must keep its deadline pending so a busy bus defers
     * (not drops) the 20 ms write — bus arbitration belongs to the queue
     * layer's retry, not to the mode-level intent. */
    TEST_TRUE(outputs.wheel_intent_allowed);
    TEST_TRUE(outputs.servo_intent_allowed);
    TEST_EQ_INT(1, runtime.wheel_busy_cycles);
    TEST_EQ_INT(1, runtime.servo_busy_cycles);

    inputs.mode = STATE_FAULT;
    inputs.wheel_bus_idle = 1U;
    inputs.servo_bus_idle = 1U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.wheel_intent_allowed);
    TEST_TRUE(!outputs.servo_intent_allowed);
    TEST_TRUE(!outputs.residual_allowed);
    inputs.mode = STATE_INIT;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.wheel_intent_allowed);
    TEST_TRUE(!outputs.servo_intent_allowed);

    /* Base velocity/yaw references are an ACTIVE-mode contract: STAND holds
     * position, CLIMB drives height only, and the heartbeat's latest vx/wz
     * are ignored outside ACTIVE regardless of what the sender carries. */
    inputs.mode = STATE_STAND;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.velocity_command_active);
    inputs.mode = STATE_CLIMB;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(!outputs.velocity_command_active);
    inputs.mode = STATE_ACTIVE;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(outputs.velocity_command_active);

    firmware_runtime_init(&runtime, UINT32_MAX - 2U);
    inputs.mode = STATE_STAND;
    inputs.now_ms = 2U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(outputs.control_due);
    TEST_TRUE(!outputs.velocity_command_active);
}

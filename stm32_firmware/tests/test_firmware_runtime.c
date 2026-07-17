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
    /* wheel_intent_allowed no longer gates on bus idle (the LQR must compute
     * every deadline; the dispatch layer skips sending when busy). */
    TEST_TRUE(outputs.wheel_intent_allowed);
    TEST_TRUE(!outputs.servo_intent_allowed);
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

    firmware_runtime_init(&runtime, UINT32_MAX - 2U);
    inputs.mode = STATE_STAND;
    inputs.now_ms = 2U;
    outputs = firmware_runtime_step(&runtime, &inputs);
    TEST_TRUE(outputs.control_due);
}

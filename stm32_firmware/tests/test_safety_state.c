#include "pin_config.h"
#include "safety_state.h"
#include "test_support.h"

#include <string.h>

static SafetyInputs_t healthy_inputs(void) {
    SafetyInputs_t inputs;
    memset(&inputs, 0, sizeof(inputs));
    inputs.now_ms = 100U;
    inputs.gyro_calibrated = 1U;
    inputs.startup_ready = 1U;
    inputs.imu_fresh = 1U;
    inputs.wheel_l_fresh = 1U;
    inputs.wheel_r_fresh = 1U;
    inputs.servos_fresh = 1U;
    inputs.link_compatible = 1U;
    inputs.heartbeat_fresh = 1U;
    inputs.action_fresh = 1U;
    inputs.requested_mode = STATE_STAND;
    return inputs;
}

static void enter_stand(SafetyInputs_t *inputs) {
    safety_state_init();
    (void)safety_state_update(inputs);
    TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);
}

static void expect_immediate_fault(SafetyInputs_t inputs, FaultMask_t fault) {
    SafetyInputs_t healthy = healthy_inputs();
    SafetyDecision_t decision;
    enter_stand(&healthy);
    decision = safety_state_update(&inputs);
    (void)decision;
    TEST_EQ_INT(STATE_FAULT, g_safety_state.current_mode);
    TEST_TRUE((g_safety_state.fault_mask & fault) != 0U);
}

static void expect_freshness_fault_after_debounce(SafetyInputs_t inputs,
                                                  FaultMask_t fault) {
    SafetyInputs_t healthy = healthy_inputs();
    enter_stand(&healthy);

    /* Move past the mode-transition grace window so the debounce logic is
     * actually exercised. */
    inputs.now_ms = g_safety_state.mode_grace_until_ms + 1U;

    /* One stale tick does NOT latch. */
    (void)safety_state_update(&inputs);
    TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);
    TEST_TRUE((g_safety_state.fault_mask & fault) == 0U);

    /* Repeat until just before the debounce threshold. */
    for (uint8_t i = 1U; i < SAFETY_FRESHNESS_DEBOUNCE_TICKS - 1U; ++i) {
        inputs.now_ms += 4U;
        (void)safety_state_update(&inputs);
        TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);
    }

    /* One more stale tick at the threshold latches. */
    inputs.now_ms += 4U;
    (void)safety_state_update(&inputs);
    TEST_EQ_INT(STATE_FAULT, g_safety_state.current_mode);
    TEST_TRUE((g_safety_state.fault_mask & fault) != 0U);
}

void run_safety_state_tests(void) {
    SafetyInputs_t inputs = healthy_inputs();
    SafetyDecision_t decision;

    enter_stand(&inputs);
    inputs.requested_mode = STATE_ACTIVE;
    decision = safety_state_update(&inputs);
    TEST_EQ_INT(STATE_ACTIVE, g_safety_state.current_mode);
    TEST_TRUE(!decision.enter_hold);

    inputs.action_fresh = 0U;
    decision = safety_state_update(&inputs);
    TEST_TRUE(decision.clear_action);
    TEST_EQ_INT(STATE_ACTIVE, g_safety_state.current_mode);

    inputs.heartbeat_fresh = 0U;
    decision = safety_state_update(&inputs);
    TEST_TRUE(decision.enter_hold);
    TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);

    inputs = healthy_inputs(); inputs.pitch_rad = SAFETY_MAX_PITCH_RAD + 0.01f;
    expect_immediate_fault(inputs, FAULT_TILT);
    inputs = healthy_inputs(); inputs.pitch_rate_rads = SAFETY_MAX_PITCH_RATE_RAD_S + 0.01f;
    expect_immediate_fault(inputs, FAULT_PITCH_RATE);
    TEST_TRUE((safety_state_legacy_fault_mask() & 0x80U) != 0U);
    inputs = healthy_inputs();
    enter_stand(&inputs);
    inputs.max_temp_c = SAFETY_MAX_TEMP_C + 1.0f;
    inputs.now_ms = 200U;
    (void)safety_state_update(&inputs);
    TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);
    inputs.now_ms = 299U;
    (void)safety_state_update(&inputs);
    TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);
    inputs.now_ms = 300U;
    (void)safety_state_update(&inputs);
    TEST_EQ_INT(STATE_FAULT, g_safety_state.current_mode);
    TEST_TRUE((g_safety_state.fault_mask & FAULT_OVERTEMP) != 0U);
    inputs = healthy_inputs(); inputs.imu_fresh = 0U;
    expect_freshness_fault_after_debounce(inputs, FAULT_IMU);
    inputs = healthy_inputs(); inputs.wheel_l_fresh = 0U;
    expect_freshness_fault_after_debounce(inputs, FAULT_WHEEL_LEFT);
    inputs = healthy_inputs(); inputs.wheel_r_fresh = 0U;
    expect_freshness_fault_after_debounce(inputs, FAULT_WHEEL_RIGHT);
    inputs = healthy_inputs(); inputs.servos_fresh = 0U;
    expect_freshness_fault_after_debounce(inputs, FAULT_SERVO);

    safety_state_init();
    inputs = healthy_inputs();
    inputs.requested_mode = STATE_FAULT;
    (void)safety_state_update(&inputs);
    TEST_TRUE((g_safety_state.fault_mask & FAULT_EMERGENCY) != 0U);

    safety_state_init();
    g_safety_state.current_mode = (RobotMode_t)99;
    inputs = healthy_inputs();
    (void)safety_state_update(&inputs);
    TEST_TRUE((g_safety_state.fault_mask & FAULT_INTERNAL) != 0U);

    safety_state_init();
    for (int i = 0; i < 1000; ++i) {
        safety_state_gyro_calib_update(0.1f, -0.2f, 0.3f, (uint32_t)i);
    }
    TEST_TRUE(g_safety_state.is_gyro_calibrated);
    TEST_NEAR(0.1f, g_safety_state.gyro_calib_offset[0], 1.0e-5f);
    TEST_NEAR(-0.2f, g_safety_state.gyro_calib_offset[1], 1.0e-5f);
    TEST_NEAR(0.3f, g_safety_state.gyro_calib_offset[2], 1.0e-5f);
    TEST_EQ_INT(STATE_INIT, g_safety_state.current_mode);

    /* --- Grace window on mode transition --- */
    safety_state_init();
    inputs = healthy_inputs();
    (void)safety_state_update(&inputs);           /* INIT -> STAND */
    TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);
    TEST_TRUE(g_safety_state.mode_grace_until_ms > 0U);

    /* Within grace window, a single stale IMU tick does not latch. */
    inputs.now_ms = 101U;
    inputs.imu_fresh = 0U;
    (void)safety_state_update(&inputs);
    TEST_EQ_INT(STATE_STAND, g_safety_state.current_mode);
    TEST_TRUE((g_safety_state.fault_mask & FAULT_IMU) == 0U);

    /* After grace expires, debounce still applies. */
    inputs.now_ms = g_safety_state.mode_grace_until_ms + 1U;
    for (uint8_t i = 0; i < SAFETY_FRESHNESS_DEBOUNCE_TICKS; ++i) {
        (void)safety_state_update(&inputs);
    }
    TEST_EQ_INT(STATE_FAULT, g_safety_state.current_mode);
    TEST_TRUE((g_safety_state.fault_mask & FAULT_IMU) != 0U);

    /* Hard faults still latch even during the grace window. */
    safety_state_init();
    inputs = healthy_inputs();
    (void)safety_state_update(&inputs);           /* INIT -> STAND, grace active */
    inputs.now_ms = 101U;
    inputs.pitch_rad = SAFETY_MAX_PITCH_RAD + 0.1f;
    (void)safety_state_update(&inputs);
    TEST_EQ_INT(STATE_FAULT, g_safety_state.current_mode);
    TEST_TRUE((g_safety_state.fault_mask & FAULT_TILT) != 0U);
}

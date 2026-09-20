#include "control_section.h"
#include "test_support.h"

#include <math.h>
#include <string.h>

/* Integration tests of the 250 Hz control section: the exact composition
 * (safety -> supervisor verdict -> gate -> LQR -> trace) that authorises
 * wheel torque.  These reproduce the scheduler's interleaving — the gate
 * verdict computed at one deadline is what the dispatch layer must obey
 * until the next deadline — which module-level suites cannot see. */

static ControlSection_t cs;
static ControlSectionInputs_t base_inputs;

static void reset_world(void) {
    memset(&base_inputs, 0, sizeof(base_inputs));
    base_inputs.now_ms = 1000U;
    base_inputs.runtime.wheel_intent_allowed = 1U;
    base_inputs.runtime.velocity_command_active = 1U;
    base_inputs.imu_control_valid = 1U;
    base_inputs.max_temp_c = 30.0f;
    base_inputs.imu_fresh = 1U;
    base_inputs.wheel_l_fresh = 1U;
    base_inputs.wheel_r_fresh = 1U;
    base_inputs.servos_fresh = 1U;
    base_inputs.startup_ready = 1U;
    base_inputs.actuator_configured = 1U;
    base_inputs.wheel_authorized = 1U;
    base_inputs.wheel_bus_idle = 1U;
    base_inputs.servo_bus_idle = 1U;
    base_inputs.link_compatible = 1U;
    base_inputs.heartbeat_fresh = 1U;
    base_inputs.action_fresh = 1U;
    base_inputs.leg_hold_tx_recent = 1U;
    base_inputs.leg_feedback_fresh = 1U;
    base_inputs.leg_posture_safe = 1U;
    base_inputs.servo_enable_verified = 0U;
    base_inputs.wheel_enable_verified = 1U;
    base_inputs.wheel_mode_complete = 1U;
    base_inputs.wheel_sent_age_ms = 8U;
    base_inputs.heartbeat.target_leg_d0 = 0.058f;

    test_set_time_ms(1000U);
    pi_link_init();
    safety_state_init();
    balance_trace_init();
    control_section_init(&cs, 1000U);
}

/* Simulate the scheduler's physical enable sequencer completing: the
 * supervisor must first observe the verified flag low (boot), then high. */
static void sequencer_enables_servos(void) {
    base_inputs.servo_enable_verified = 1U;
}

static ControlSectionOutputs_t run_deadline(uint32_t now_ms, float pitch) {
    base_inputs.now_ms = now_ms;
    base_inputs.pitch_rad = pitch;
    base_inputs.pitch_rate_rad_s = 0.0f;
    test_set_time_ms(now_ms);
    return control_section_step(&cs, &base_inputs);
}

/* Drive the safety machine from INIT to STAND via fresh inputs. */
static void bring_to_stand(void) {
    int i;
    run_deadline(1000U, 0.0f);                 /* pre-enable deadline */
    sequencer_enables_servos();
    for (i = 0; i < 4; ++i) {
        run_deadline(1004U + 4U * (uint32_t)i, 0.0f);
    }
    TEST_EQ_INT((uint32_t)STATE_STAND, (uint32_t)g_safety_state.current_mode);
}

static void test_gate_closed_until_all_verdicts_open(void) {
    ControlSectionOutputs_t out;
    reset_world();
    /* Pre-enable: the gate stays closed even with every other verdict open. */
    out = run_deadline(1004U, 0.0f);
    TEST_TRUE(!out.wheel_output_gate);
    out = run_deadline(1008U, 0.0f);
    TEST_TRUE(!out.wheel_output_gate);
    sequencer_enables_servos();
    out = run_deadline(1012U, 0.0f);           /* verified returns: re-issue done */
    TEST_TRUE(!out.wheel_output_gate);         /* ...but READY only on a later step */
    out = run_deadline(1016U, 0.0f);
    TEST_TRUE(out.wheel_output_gate);
    TEST_NEAR(0.0f, out.torque_left_nm, 1e-6f); /* balanced, zero torque */

    base_inputs.wheel_mode_complete = 0U;       /* current-loop not re-asserted */
    out = run_deadline(1008U, 0.0f);
    TEST_TRUE(!out.wheel_output_gate);
    TEST_NEAR(0.0f, out.torque_left_nm, 1e-6f);
    base_inputs.wheel_mode_complete = 1U;

    base_inputs.wheel_authorized = 0U;          /* startup not READY */
    out = run_deadline(1012U, 0.0f);
    TEST_TRUE(!out.wheel_output_gate);
    base_inputs.wheel_authorized = 1U;
}

static void test_supervisor_denial_blocks_torque_every_deadline(void) {
    /* A1 regression: leg feedback goes stale while the wheels stay fresh.
     * The supervisor denies output; the gate must stay closed and the
     * torque zero at EVERY deadline — never a nonzero frame between the
     * denial and the next verdict, regardless of interleaving. */
    ControlSectionOutputs_t out;
    int i;
    reset_world();
    run_deadline(1004U, 0.0f);
    sequencer_enables_servos();
    run_deadline(1008U, 0.0f);
    base_inputs.leg_feedback_fresh = 0U;
    for (i = 0; i < 10; ++i) {
        out = run_deadline(1008U + 4U * (uint32_t)i, 0.05f /* tilted */);
        TEST_TRUE(!out.wheel_output_gate);
        TEST_NEAR(0.0f, out.torque_left_nm, 1e-6f);
        TEST_NEAR(0.0f, out.torque_right_nm, 1e-6f);
    }
    /* Feedback restored: the supervisor re-runs its readiness sequence and
     * the gate reopens only after it reports READY again. */
    base_inputs.leg_feedback_fresh = 1U;
    out = run_deadline(1052U, 0.0f);
    TEST_TRUE(out.wheel_output_gate);
    (void)out;
}

static void test_stand_ignores_velocity_commands(void) {
    /* A2 regression: a heartbeat carrying vx=0.4 m/s must drive the
     * reference in ACTIVE only.  In STAND the same heartbeat holds
     * position (v_ref stays zero). */
    ControlSectionOutputs_t out;
    int i;
    reset_world();
    bring_to_stand();
    base_inputs.runtime.velocity_command_active = 0U; /* STAND semantics */
    base_inputs.heartbeat.target_velocity = 0.4f;
    for (i = 0; i < 25; ++i) { /* 100 ms of deadlines */
        out = run_deadline(1100U + 4U * (uint32_t)i, 0.0f);
        (void)out;
    }
    TEST_TRUE(fabsf(cs.lqr.v_ref) < 1e-4f);
    TEST_NEAR(0.0f, cs.lqr.x_ref, 1e-3f);

    /* Same heartbeat, ACTIVE semantics: the reference ramps toward 0.4
     * (2 m/s^2 accel, 8 m/s^3 jerk limits — 240 ms clears both). */
    base_inputs.runtime.velocity_command_active = 1U;
    for (i = 0; i < 60; ++i) {
        out = run_deadline(1200U + 4U * (uint32_t)i, 0.0f);
        (void)out;
    }
    TEST_TRUE(cs.lqr.v_ref > 0.1f);
    TEST_TRUE(cs.lqr.v_ref < 0.45f);
    TEST_TRUE(cs.lqr.x_ref > 0.005f);
}

static void test_fault_entry_pulses_once_and_latches(void) {
    ControlSectionOutputs_t out;
    reset_world();
    bring_to_stand();
    /* Tilt past SAFETY_MAX_PITCH_RAD latches FAULT and pulses the edge for
     * exactly one deadline; torque is zero from that deadline on. */
    out = run_deadline(1200U, 0.9f);
    TEST_TRUE(out.fault_entry_edge);
    TEST_TRUE(!out.wheel_output_gate);
    TEST_NEAR(0.0f, out.torque_left_nm, 1e-6f);
    out = run_deadline(1204U, 0.0f);
    TEST_TRUE(!out.fault_entry_edge);           /* one pulse only */
    TEST_EQ_INT((uint32_t)STATE_FAULT, (uint32_t)g_safety_state.current_mode);
    TEST_TRUE(g_safety_state.fault_mask & FAULT_TILT);
    /* TILT is serious: stays latched even when the pitch recovers. */
    out = run_deadline(1208U, 0.0f);
    TEST_EQ_INT((uint32_t)STATE_FAULT, (uint32_t)g_safety_state.current_mode);
    TEST_TRUE(!out.wheel_output_gate);
}

static void test_imu_invalid_revokes_torque_but_keeps_tracing(void) {
    ControlSectionOutputs_t out;
    reset_world();
    bring_to_stand();
    {
        uint16_t count_before = g_bt_count;
        base_inputs.imu_control_valid = 0U;
        out = run_deadline(1200U, 0.05f);
        TEST_NEAR(0.0f, out.torque_left_nm, 1e-6f); /* no torque without it */
        /* The trace still recorded this deadline, including the invalid-IMU
         * validity bit — wall-clock tracing survives an estimation loss. */
        TEST_EQ_INT((int)(count_before + 1U), (int)g_bt_count);
    }
    TEST_TRUE((g_bt_buf[(g_bt_head + BALANCE_TRACE_LEN - 1) % BALANCE_TRACE_LEN]
               .validity & 0x1U) == 0U); /* IMU validity bit clear */
}

static void test_pitch_fade_limits_fallen_robot_torque(void) {
    ControlSectionOutputs_t out;
    reset_world();
    bring_to_stand();
    out = run_deadline(1200U, 0.40f); /* between FADE_START and FADE_END */
    TEST_TRUE(out.wheel_output_gate);
    TEST_TRUE(fabsf(out.torque_left_nm) < 1.1f);
    out = run_deadline(1204U, 0.9f);  /* beyond FADE_END: zero authority */
    TEST_NEAR(0.0f, out.torque_left_nm, 1e-6f);
}

static void test_boot_reaches_gate_with_coupled_enable_flags(void) {
    /* main.c feeds the section with startup_ready and servo_enable_verified
     * driven by the SAME sequencer variable (servos_enabled): they flip
     * 0->1 TOGETHER at the deadline the servo enable sequence completes, so
     * servo_enable_verified is never observed low while startup_ready is 1.
     * The supervisor must still clear its enable-reissue stage from the
     * low observations made during the pre-READY phase, or the gate stays
     * closed for the whole power-on session. */
    ControlSectionOutputs_t out;
    int i;
    reset_world();
    base_inputs.startup_ready = 0U;
    base_inputs.servo_enable_verified = 0U;
    base_inputs.leg_hold_tx_recent = 0U;
    base_inputs.leg_feedback_fresh = 0U;
    /* Pre-READY deadlines: supervisor parked in the revocation/STARTUP
     * branches while the sequencer is still enabling the servos. */
    run_deadline(1000U, 0.0f);
    run_deadline(1004U, 0.0f);
    /* servos_enabled flips: safety INIT->STAND, both coupled inputs 1. */
    base_inputs.startup_ready = 1U;
    base_inputs.servo_enable_verified = 1U;
    out = run_deadline(1008U, 0.0f);
    TEST_TRUE(!out.wheel_output_gate); /* no leg hold on the wire yet */
    base_inputs.leg_hold_tx_recent = 1U;
    base_inputs.leg_feedback_fresh = 1U;
    for (i = 0; i < 6; ++i) {          /* bounded: must not need a fault */
        out = run_deadline(1012U + 4U * (uint32_t)i, 0.0f);
    }
    TEST_TRUE(out.wheel_output_gate);
}

static void test_stale_imu_freshness_revokes_torque(void) {
    /* DRDY-line death freezes g_imu_control_valid at its last value while
     * the published attitude is frozen with it; the section must also
     * require pair freshness before computing torque.  A nonzero pitch
     * makes the stale computation produce nonzero torque, so the assertion
     * has discriminating power against the old filter-verdict-only gate. */
    ControlSectionOutputs_t out;
    float stale_torque = 0.0f;
    reset_world();
    bring_to_stand();
    base_inputs.imu_fresh = 1U;        /* control case: fresh pair */
    out = run_deadline(1200U, 0.05f);
    stale_torque = out.torque_left_nm;
    TEST_TRUE(fabsf(stale_torque) > 1e-3f); /* leaning body -> real torque */
    base_inputs.imu_fresh = 0U;        /* now the pair goes stale */
    out = run_deadline(1204U, 0.05f);
    TEST_TRUE(out.wheel_output_gate);  /* gate open: devices fine */
    TEST_NEAR(0.0f, out.torque_left_nm, 1e-6f); /* but no stale-authority torque */
    TEST_NEAR(0.0f, out.torque_right_nm, 1e-6f);
}

static void test_transient_fault_recovery_reruns_enable_sequencer(void) {
    /* Scheduler contract for transient faults (wheel/servo freshness):
     * FAULT auto-recovers, and the servo enable sequencer in main.c is gated
     * by mode != FAULT, so the coupled servo_enable_verified/startup_ready
     * inputs DROP for the whole FAULT sojourn and rise again only after
     * recovery has physically re-run the enable frames.  The supervisor must
     * clear its re-issue stage from that genuine drop-and-return, and the
     * gate must reopen -- never on a stale verified flag that stayed high
     * while the FAULT one-shot was disabling the physical torque. */
    ControlSectionOutputs_t out;
    int i;
    reset_world();
    bring_to_stand();
    out = run_deadline(1200U, 0.0f);
    TEST_TRUE(out.wheel_output_gate);

    /* Wheel-feedback glitch: the 8-tick freshness debounce (32 ms) lets the
     * gate ride through the first stale deadlines by design -- wheel
     * staleness is a safety-machine input, not a supervisor input -- and
     * then latches a transient FAULT_WHEEL_LEFT. */
    base_inputs.wheel_l_fresh = 0U;
    for (i = 0; i < 10; ++i) {
        out = run_deadline(1204U + 4U * (uint32_t)i, 0.0f);
    }
    TEST_EQ_INT((uint32_t)STATE_FAULT, (uint32_t)g_safety_state.current_mode);
    TEST_TRUE(!out.wheel_output_gate);

    /* Feedback returns: the transient fault clears and the machine recovers
     * to STAND while the (gated) sequencer still reports verified = 0 --
     * startup_ready and leg_hold_tx_recent are coupled to the same state. */
    base_inputs.wheel_l_fresh = 1U;
    base_inputs.servo_enable_verified = 0U;
    base_inputs.startup_ready = 0U;
    base_inputs.leg_hold_tx_recent = 0U;
    for (i = 0; i < 6; ++i) {
        out = run_deadline(1300U + 4U * (uint32_t)i, 0.0f);
        TEST_TRUE(!out.wheel_output_gate);
    }
    TEST_EQ_INT((uint32_t)STATE_STAND, (uint32_t)g_safety_state.current_mode);

    /* Recovery re-runs the sequencer: enables complete, hold writes return,
     * and the gate reopens through the supervisor's re-issue stage within a
     * bounded number of deadlines. */
    base_inputs.servo_enable_verified = 1U;
    base_inputs.startup_ready = 1U;
    base_inputs.leg_hold_tx_recent = 1U;
    for (i = 0; i < 8; ++i) {
        out = run_deadline(1400U + 4U * (uint32_t)i, 0.0f);
    }
    TEST_TRUE(out.wheel_output_gate);
    TEST_TRUE(fabsf(out.torque_left_nm) < 1e-6f); /* balanced: no torque */
}

void run_control_section_tests(void) {
    test_gate_closed_until_all_verdicts_open();
    test_supervisor_denial_blocks_torque_every_deadline();
    test_stand_ignores_velocity_commands();
    test_fault_entry_pulses_once_and_latches();
    test_imu_invalid_revokes_torque_but_keeps_tracing();
    test_pitch_fade_limits_fallen_robot_torque();
    test_boot_reaches_gate_with_coupled_enable_flags();
    test_stale_imu_freshness_revokes_torque();
    test_transient_fault_recovery_reruns_enable_sequencer();
}

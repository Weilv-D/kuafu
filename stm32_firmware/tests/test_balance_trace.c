#include "balance_trace.h"
#include "test_support.h"

#include <string.h>

static BalanceTraceInput_t sample_input(uint32_t now_ms, float target_l) {
    BalanceTraceInput_t in;
    memset(&in, 0, sizeof(in));
    in.timestamp_ms = now_ms;
    in.mode = 2U;
    in.fault_mask = 0U;
    in.startup_state = 4U;
    in.actuator_state = 7U;
    in.validity = 0x3FU;
    in.imu_age_ms = 2U;
    in.wheel_left_age_ms = 3U;
    in.wheel_right_age_ms = 4U;
    in.pitch_rad = 0.1f;
    in.pitch_rate_rad_s = -0.2f;
    in.wheel_left_rad_s = 0.5f;
    in.wheel_right_rad_s = 0.6f;
    in.target_torque_left_nm = target_l;
    in.target_torque_right_nm = target_l + 0.1f;
    in.sent_torque_left_nm = target_l + 0.2f;
    in.sent_torque_right_nm = target_l + 0.3f;
    in.feedback_torque_left_nm = target_l + 0.4f;
    in.feedback_torque_right_nm = target_l + 0.5f;
    return in;
}

static void test_empty_and_partial_snapshot(void) {
    BalanceTraceSnapshot_t snap;
    balance_trace_init();
    balance_trace_set_metadata("test-fw", "build-1");
    TEST_TRUE(balance_trace_snapshot(&snap));
    TEST_EQ_INT(0, (int)snap.count);
    TEST_EQ_INT(0, (int)snap.total_records);
    TEST_TRUE(!snap.is_full);

    BalanceTraceInput_t in = sample_input(1234U, 0.3f);
    balance_trace_record(&in);
    TEST_TRUE(balance_trace_snapshot(&snap));
    TEST_EQ_INT(1, (int)snap.count);
    TEST_EQ_INT(1, (int)snap.total_records);
    TEST_EQ_INT(0, (int)snap.samples[0].seq);
    TEST_EQ_INT(1234, (int)snap.samples[0].timestamp_ms);
    TEST_EQ_INT(2, (int)snap.samples[0].mode);
    TEST_NEAR(0.3f, snap.samples[0].target_torque_left_nm, 1.0e-6f);
    TEST_NEAR(0.5f, snap.samples[0].sent_torque_left_nm, 1.0e-6f);
    TEST_NEAR(0.7f, snap.samples[0].feedback_torque_left_nm, 1.0e-6f);
}

static void test_ring_wraps_and_keeps_chronological_window(void) {
    BalanceTraceSnapshot_t snap;
    balance_trace_init();
    for (uint32_t i = 0U; i < BALANCE_TRACE_LEN + 5U; ++i) {
        BalanceTraceInput_t in = sample_input(i * 4U, (float)i);
        balance_trace_record(&in);
    }
    TEST_TRUE(balance_trace_snapshot(&snap));
    TEST_EQ_INT(BALANCE_TRACE_LEN, (int)snap.count);
    TEST_EQ_INT(BALANCE_TRACE_LEN + 5U, (int)snap.total_records);
    TEST_TRUE(snap.is_full);
    TEST_EQ_INT(5, (int)snap.samples[0].seq);
    TEST_EQ_INT(BALANCE_TRACE_LEN + 4U, (int)snap.samples[BALANCE_TRACE_LEN - 1U].seq);
    TEST_NEAR(5.0f, snap.samples[0].target_torque_left_nm, 1.0e-6f);
}

static void test_init_and_fault_event_freeze_are_bounded(void) {
    BalanceTraceSnapshot_t snap;
    balance_trace_init();
    BalanceTraceInput_t in = sample_input(10U, 1.0f);
    balance_trace_note_event(BALANCE_TRACE_EVENT_INIT, 10U);
    balance_trace_record(&in);
    in.timestamp_ms = 20U;
    in.fault_mask = 0x80000004UL;
    in.mode = 4U;
    in.actuator_state = 9U;
    balance_trace_note_event(BALANCE_TRACE_EVENT_FAULT, 20U);
    balance_trace_record(&in);
    TEST_TRUE(balance_trace_snapshot(&snap));
    TEST_EQ_INT(BALANCE_TRACE_EVENT_INIT, (int)snap.samples[0].event);
    TEST_EQ_INT(BALANCE_TRACE_EVENT_FAULT, (int)snap.samples[1].event);
    TEST_EQ_INT(4, (int)snap.samples[1].mode);
    TEST_EQ_INT(0x80000004UL, (int)snap.samples[1].fault_mask);

    for (uint32_t i = 0U; i < BALANCE_TRACE_POST_FAULT_SAMPLES + 10U; ++i) {
        in.timestamp_ms += 4U;
        balance_trace_note_event(BALANCE_TRACE_EVENT_FAULT, in.timestamp_ms);
        balance_trace_record(&in);
    }
    TEST_TRUE(balance_trace_snapshot(&snap));
    TEST_EQ_INT(2U + BALANCE_TRACE_POST_FAULT_SAMPLES, (int)snap.total_records);
    TEST_TRUE(snap.frozen);
    TEST_EQ_INT(2U + BALANCE_TRACE_POST_FAULT_SAMPLES - 1U,
                (int)snap.samples[snap.count - 1U].seq);
}

static void test_metadata_and_size_budget(void) {
    BalanceTraceSnapshot_t snap;
    balance_trace_init();
    balance_trace_set_metadata("fw-1.2.3", "build-abc");
    TEST_TRUE(balance_trace_snapshot(&snap));
    TEST_TRUE(strcmp(snap.firmware_version, "fw-1.2.3") == 0);
    TEST_TRUE(strcmp(snap.build_id, "build-abc") == 0);
    TEST_EQ_INT(80, (int)sizeof(BalanceTraceSample_t));
    TEST_EQ_INT(20480, (int)sizeof(g_bt_buf));
}

void run_balance_trace_tests(void) {
    test_empty_and_partial_snapshot();
    test_ring_wraps_and_keeps_chronological_window();
    test_init_and_fault_event_freeze_are_bounded();
    test_metadata_and_size_budget();
}

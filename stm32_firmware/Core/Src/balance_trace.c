#include "balance_trace.h"

#include <stddef.h>

#if defined(_MSC_VER)
#include <intrin.h>
#define BALANCE_TRACE_BARRIER() _ReadWriteBarrier()
#elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 6000000)
/* armcc v5: compiler-ordering intrinsic. The Cortex-M4 is an in-order core
 * that does not reorder normal SRM writes, which is what the SWD seqlock
 * reader observes; a compiler barrier is sufficient here. */
#define BALANCE_TRACE_BARRIER() __schedule_barrier()
#else
#define BALANCE_TRACE_BARRIER() __asm volatile("dmb" ::: "memory")
#endif

BalanceTraceSample_t g_bt_buf[BALANCE_TRACE_LEN];
volatile uint16_t g_bt_head;
volatile uint16_t g_bt_count;
volatile uint32_t g_bt_seq;
volatile uint32_t g_bt_snapshot_seq;
volatile uint8_t g_bt_frozen;
volatile uint8_t g_bt_post_fault_remaining;
char g_bt_firmware_version[BALANCE_TRACE_TEXT_LEN];
char g_bt_build_id[BALANCE_TRACE_TEXT_LEN];
const uint32_t g_bt_trace_version = BALANCE_TRACE_VERSION;

static uint8_t g_bt_pending_event;

static void copy_text(char *dst, const char *src) {
    size_t i;
    if (src == NULL) {
        src = "unknown";
    }
    for (i = 0U; i + 1U < BALANCE_TRACE_TEXT_LEN && src[i] != '\0'; ++i) {
        dst[i] = src[i];
    }
    dst[i] = '\0';
    for (++i; i < BALANCE_TRACE_TEXT_LEN; ++i) {
        dst[i] = '\0';
    }
}

void balance_trace_init(void) {
    g_bt_snapshot_seq = 0U;
    g_bt_head = 0U;
    g_bt_count = 0U;
    g_bt_seq = 0U;
    g_bt_frozen = 0U;
    g_bt_post_fault_remaining = 0U;
    g_bt_pending_event = BALANCE_TRACE_EVENT_NONE;
    copy_text(g_bt_firmware_version, "unknown");
    copy_text(g_bt_build_id, "unknown");
    BALANCE_TRACE_BARRIER();
}

void balance_trace_set_metadata(const char *firmware_version, const char *build_id) {
    g_bt_snapshot_seq++;
    BALANCE_TRACE_BARRIER();
    copy_text(g_bt_firmware_version, firmware_version);
    copy_text(g_bt_build_id, build_id);
    BALANCE_TRACE_BARRIER();
    g_bt_snapshot_seq++;
}

void balance_trace_note_event(BalanceTraceEvent_t event, uint32_t timestamp_ms) {
    (void)timestamp_ms;
    if (event == BALANCE_TRACE_EVENT_INIT || event == BALANCE_TRACE_EVENT_FAULT) {
        g_bt_pending_event = (uint8_t)event;
    }
}

void balance_trace_record(const BalanceTraceInput_t *input) {
    BalanceTraceSample_t *sample;
    uint8_t event;

    if (input == NULL || g_bt_frozen != 0U) {
        return;
    }

    /* Publish odd before writes and even only after the complete sample. */
    g_bt_snapshot_seq++;
    BALANCE_TRACE_BARRIER();
    sample = &g_bt_buf[g_bt_head];
    sample->timestamp_ms = input->timestamp_ms;
    sample->seq = g_bt_seq;
    sample->imu_age_ms = input->imu_age_ms;
    sample->wheel_left_age_ms = input->wheel_left_age_ms;
    sample->wheel_right_age_ms = input->wheel_right_age_ms;
    sample->mode = input->mode;
    sample->fault_mask = input->fault_mask;
    sample->startup_state = input->startup_state;
    sample->actuator_state = input->actuator_state;
    sample->validity = input->validity;
    event = g_bt_pending_event;
    sample->event = event;
    sample->pitch_rad = input->pitch_rad;
    sample->pitch_rate_rad_s = input->pitch_rate_rad_s;
    sample->wheel_left_rad_s = input->wheel_left_rad_s;
    sample->wheel_right_rad_s = input->wheel_right_rad_s;
    sample->target_torque_left_nm = input->target_torque_left_nm;
    sample->target_torque_right_nm = input->target_torque_right_nm;
    sample->sent_torque_left_nm = input->sent_torque_left_nm;
    sample->sent_torque_right_nm = input->sent_torque_right_nm;
    sample->feedback_torque_left_nm = input->feedback_torque_left_nm;
    sample->feedback_torque_right_nm = input->feedback_torque_right_nm;
    BALANCE_TRACE_BARRIER();

    g_bt_seq++;
    g_bt_head = (uint16_t)((g_bt_head + 1U) % BALANCE_TRACE_LEN);
    if (g_bt_count < BALANCE_TRACE_LEN) {
        g_bt_count++;
    }

    /* A fault starts one bounded tail; repeated FAULT samples do not extend it. */
    if ((event == BALANCE_TRACE_EVENT_FAULT || input->fault_mask != 0U) &&
        g_bt_post_fault_remaining == 0U && g_bt_frozen == 0U) {
        g_bt_post_fault_remaining = BALANCE_TRACE_POST_FAULT_SAMPLES;
    } else if (g_bt_post_fault_remaining != 0U) {
        g_bt_post_fault_remaining--;
        if (g_bt_post_fault_remaining == 0U) {
            g_bt_frozen = 1U;
        }
    }
    g_bt_pending_event = BALANCE_TRACE_EVENT_NONE;
    BALANCE_TRACE_BARRIER();
    g_bt_snapshot_seq++;
}

int balance_trace_snapshot(BalanceTraceSnapshot_t *out) {
    uint32_t begin;
    uint32_t end;
    uint16_t head;
    uint16_t count;
    uint16_t i;

    if (out == NULL) {
        return 0;
    }
    for (uint32_t attempt = 0U; attempt < 3U; ++attempt) {
        begin = g_bt_snapshot_seq;
        if ((begin & 1U) != 0U) {
            continue;
        }
        head = g_bt_head;
        count = g_bt_count;
        out->version = g_bt_trace_version;
        out->head = head;
        out->count = count;
        out->total_records = g_bt_seq;
        out->snapshot_seq = begin;
        out->is_full = (count == BALANCE_TRACE_LEN) ? 1U : 0U;
        out->frozen = g_bt_frozen;
        copy_text(out->firmware_version, g_bt_firmware_version);
        copy_text(out->build_id, g_bt_build_id);
        /* For an unfilled ring, valid records occupy slots [0, count). */
        for (i = 0U; i < count; ++i) {
            uint16_t index = (count == BALANCE_TRACE_LEN) ?
                (uint16_t)((head + i) % BALANCE_TRACE_LEN) : i;
            out->samples[i] = g_bt_buf[index];
        }
        BALANCE_TRACE_BARRIER();
        end = g_bt_snapshot_seq;
        if (begin == end && (end & 1U) == 0U) {
            return 1;
        }
    }
    return 0;
}

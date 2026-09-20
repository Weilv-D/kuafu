#ifndef BALANCE_TRACE_H
#define BALANCE_TRACE_H

#include <stdint.h>

/* 256 samples at 250 Hz = 1.024 s; post-fault tail is 64 samples = 256 ms.
 * A fault arms the bounded tail and freezes the ring so the post-fault window
 * survives for the SWD dump; the freeze is released when a new fault arrives
 * after a fault-free period, or once the system has been fault-free for a
 * full tail, so an auto-recovered transient fault cannot blind the rest of
 * the session.  A latched (serious) fault keeps the ring frozen by design:
 * the robot is locked down and there is nothing further to record. */
#define BALANCE_TRACE_LEN 256U
#define BALANCE_TRACE_POST_FAULT_SAMPLES 64U
#define BALANCE_TRACE_VERSION 2U
#define BALANCE_TRACE_TEXT_LEN 32U

typedef enum {
    BALANCE_TRACE_EVENT_NONE = 0U,
    BALANCE_TRACE_EVENT_INIT = 1U,
    BALANCE_TRACE_EVENT_FAULT = 2U
} BalanceTraceEvent_t;

typedef struct {
    uint32_t timestamp_ms;
    uint32_t seq;
    uint16_t imu_age_ms;
    uint16_t wheel_left_age_ms;
    uint16_t wheel_right_age_ms;
    uint32_t mode;
    uint32_t fault_mask;
    uint32_t startup_state;
    uint32_t actuator_state;
    uint32_t validity;
    uint32_t event;
    float pitch_rad;
    float pitch_rate_rad_s;
    float wheel_left_rad_s;
    float wheel_right_rad_s;
    float target_torque_left_nm;
    float target_torque_right_nm;
    float sent_torque_left_nm;
    float sent_torque_right_nm;
    float feedback_torque_left_nm;
    float feedback_torque_right_nm;
} BalanceTraceSample_t;

typedef struct {
    uint32_t timestamp_ms;
    uint16_t imu_age_ms;
    uint16_t wheel_left_age_ms;
    uint16_t wheel_right_age_ms;
    uint32_t mode;
    uint32_t fault_mask;
    uint32_t startup_state;
    uint32_t actuator_state;
    uint32_t validity;
    float pitch_rad;
    float pitch_rate_rad_s;
    float wheel_left_rad_s;
    float wheel_right_rad_s;
    float target_torque_left_nm;
    float target_torque_right_nm;
    float sent_torque_left_nm;
    float sent_torque_right_nm;
    float feedback_torque_left_nm;
    float feedback_torque_right_nm;
} BalanceTraceInput_t;

typedef struct {
    uint32_t version;
    uint32_t head;
    uint32_t count;
    uint32_t total_records;
    uint32_t snapshot_seq;
    uint8_t is_full;
    uint8_t frozen;
    uint8_t reserved[2];
    char firmware_version[BALANCE_TRACE_TEXT_LEN];
    char build_id[BALANCE_TRACE_TEXT_LEN];
    BalanceTraceSample_t samples[BALANCE_TRACE_LEN];
} BalanceTraceSnapshot_t;

/* ABI/RAM contract: 80 bytes/sample, 20,480 bytes for the ring. */
typedef char balance_trace_sample_size_must_be_80[(sizeof(BalanceTraceSample_t) == 80U) ? 1 : -1];
typedef char balance_trace_ring_size_must_be_20k[(sizeof(BalanceTraceSample_t) * BALANCE_TRACE_LEN == 20480U) ? 1 : -1];

/* SWD-visible symbols. g_bt_snapshot_seq is an even/odd writer seqlock. */
extern BalanceTraceSample_t g_bt_buf[BALANCE_TRACE_LEN];
extern volatile uint16_t g_bt_head;
extern volatile uint16_t g_bt_count;
extern volatile uint32_t g_bt_seq;
extern volatile uint32_t g_bt_snapshot_seq;
extern volatile uint8_t g_bt_frozen;
extern volatile uint8_t g_bt_post_fault_remaining;
extern char g_bt_firmware_version[BALANCE_TRACE_TEXT_LEN];
extern char g_bt_build_id[BALANCE_TRACE_TEXT_LEN];
extern const uint32_t g_bt_trace_version;

void balance_trace_init(void);
void balance_trace_set_metadata(const char *firmware_version, const char *build_id);
void balance_trace_note_event(BalanceTraceEvent_t event, uint32_t timestamp_ms);
void balance_trace_record(const BalanceTraceInput_t *input);
int balance_trace_snapshot(BalanceTraceSnapshot_t *out);

#endif

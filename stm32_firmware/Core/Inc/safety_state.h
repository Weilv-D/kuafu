#ifndef SAFETY_STATE_H
#define SAFETY_STATE_H

#include <stdint.h>

typedef enum {
    STATE_INIT = 0,
    STATE_STAND = 1,
    STATE_ACTIVE = 2,
    STATE_CLIMB = 3,
    STATE_FAULT = 4
} RobotMode_t;

typedef uint32_t FaultMask_t;

#define FAULT_NONE          ((FaultMask_t)0U)
#define FAULT_TILT          ((FaultMask_t)1U << 0)
#define FAULT_HEARTBEAT     ((FaultMask_t)1U << 1) /* legacy bit; stale heartbeat is recoverable */
#define FAULT_OVERTEMP      ((FaultMask_t)1U << 2)
#define FAULT_EMERGENCY     ((FaultMask_t)1U << 3)
#define FAULT_SERVO         ((FaultMask_t)1U << 4)
#define FAULT_IMU           ((FaultMask_t)1U << 5)
#define FAULT_WHEEL_LEFT    ((FaultMask_t)1U << 6)
#define FAULT_WHEEL_RIGHT   ((FaultMask_t)1U << 7)
#define FAULT_PITCH_RATE    ((FaultMask_t)1U << 8)
#define FAULT_INIT          ((FaultMask_t)1U << 9)
#define FAULT_INTERNAL      ((FaultMask_t)1U << 10)

typedef struct {
    RobotMode_t current_mode;
    uint32_t mode_timer_ms;
    FaultMask_t fault_mask;
    float gyro_calib_offset[3];
    uint8_t is_gyro_calibrated;
} SafetyState_t;

typedef struct {
    uint32_t now_ms;
    float pitch_rad;
    float pitch_rate_rads;
    float max_temp_c;
    uint8_t gyro_calibrated;
    uint8_t startup_ready;
    uint8_t imu_fresh;
    uint8_t wheel_l_fresh;
    uint8_t wheel_r_fresh;
    uint8_t servos_fresh;
    uint8_t link_compatible;
    uint8_t heartbeat_fresh;
    uint8_t action_fresh;
    uint8_t requested_mode;
} SafetyInputs_t;

typedef struct {
    uint8_t enter_hold;
    uint8_t clear_action;
} SafetyDecision_t;

extern SafetyState_t g_safety_state;

void safety_state_init(void);
void safety_state_trigger_fault(FaultMask_t fault);
void safety_state_gyro_calib_update(float gx, float gy, float gz, uint32_t now_ms);
SafetyDecision_t safety_state_update(const SafetyInputs_t *inputs);
uint8_t safety_state_legacy_fault_mask(void);

#endif

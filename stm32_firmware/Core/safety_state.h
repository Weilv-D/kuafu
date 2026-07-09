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

typedef enum {
    FAULT_NONE = 0x00,
    FAULT_TILT = 0x01,          /* Tilt / Pitch too high (> 45 degrees) */
    FAULT_HEARTBEAT = 0x02,     /* Pi link heartbeat lost */
    FAULT_OVERTEMP = 0x04,      /* Servo/sensor overtemperature */
    FAULT_EMERGENCY = 0x08      /* Emergency stop request from Pi */
} FaultCode_t;

typedef struct {
    RobotMode_t current_mode;
    FaultCode_t active_fault;
    uint32_t mode_timer_ms;     /* Time spent in current mode */
    uint8_t error_mask;         /* Bitmask of all triggered faults */
    float gyro_calib_offset[3]; /* Calibrated gyroscope offsets */
    int is_gyro_calibrated;
} SafetyState_t;

extern SafetyState_t g_safety_state;

/**
 * @brief Initializes the safety state machine.
 */
void safety_state_init(void);

/**
 * @brief Updates the robot state machine and executes transitions based on telemetry.
 * 
 * @param pitch_rad Current pitch angle of the body (from fusion).
 * @param gyro_y_rads Current pitch rate.
 * @param max_temp_c Maximum detected system temperature.
 * @param dt Loop time step (seconds).
 */
void safety_state_update(float pitch_rad, float gyro_y_rads, float max_temp_c, float dt);

/**
 * @brief Checks if gyro calibration is in progress and processes raw samples.
 * 
 * @param gx Gyro x reading.
 * @param gy Gyro y reading.
 * @param gz Gyro z reading.
 */
void safety_state_gyro_calib_update(float gx, float gy, float gz);

/**
 * @brief Force transitions the state machine to FAULT state.
 * 
 * @param fault Target fault code to trigger.
 */
void safety_state_trigger_fault(FaultCode_t fault);

#endif /* SAFETY_STATE_H */

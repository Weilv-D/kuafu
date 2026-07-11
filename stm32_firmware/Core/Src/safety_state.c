#include "safety_state.h"
#include "pi_link.h"
#include "pin_config.h"
#include <string.h>
#include <math.h>

SafetyState_t g_safety_state;

static uint32_t calibration_samples_count = 0;
static float calibration_sum[3] = {0.0f, 0.0f, 0.0f};

void safety_state_init(void) {
    memset(&g_safety_state, 0, sizeof(g_safety_state));
    g_safety_state.current_mode = STATE_INIT;
    g_safety_state.active_fault = FAULT_NONE;
    g_safety_state.is_gyro_calibrated = 0;
    calibration_samples_count = 0;
    calibration_sum[0] = 0.0f;
    calibration_sum[1] = 0.0f;
    calibration_sum[2] = 0.0f;
}

void safety_state_trigger_fault(FaultCode_t fault) {
    g_safety_state.active_fault |= fault;
    g_safety_state.error_mask |= fault;
    g_safety_state.current_mode = STATE_FAULT;
}

void safety_state_gyro_calib_update(float gx, float gy, float gz) {
    if (g_safety_state.is_gyro_calibrated) {
        return;
    }

    calibration_sum[0] += gx;
    calibration_sum[1] += gy;
    calibration_sum[2] += gz;
    calibration_samples_count++;

    if (calibration_samples_count >= 1000) { /* 1000 samples = 1.0 second at 1kHz */
        g_safety_state.gyro_calib_offset[0] = calibration_sum[0] / 1000.0f;
        g_safety_state.gyro_calib_offset[1] = calibration_sum[1] / 1000.0f;
        g_safety_state.gyro_calib_offset[2] = calibration_sum[2] / 1000.0f;
        g_safety_state.is_gyro_calibrated = 1;
        
        /* Auto transition from INIT to STAND */
        g_safety_state.current_mode = STATE_STAND;
        g_safety_state.mode_timer_ms = HAL_GetTick();
    }
}

void safety_state_update(float pitch_rad, float gyro_y_rads, float max_temp_c, float dt) {
    uint32_t current_time = HAL_GetTick();

    /* --- 1. Global Fault Detection --- */
    if (g_safety_state.current_mode != STATE_INIT) {
        /* Emergency stop requested by the Pi */
        if (g_pi_cmd_heartbeat.mode_request == STATE_FAULT) {
            safety_state_trigger_fault(FAULT_EMERGENCY);
        }

        /* Check Tilt (Pitch > 45 degrees) */
        if (fabsf(pitch_rad) > SAFETY_MAX_PITCH_RAD) {
            safety_state_trigger_fault(FAULT_TILT);
        }

        /* Check Overtemperature */
        if (max_temp_c > SAFETY_MAX_TEMP_C) {
            safety_state_trigger_fault(FAULT_OVERTEMP);
        }

        /* Check Heartbeat Timeout (Only enforce after Pi link starts sending heartbeats or when in ACTIVE/CLIMB) */
        if (g_pi_cmd_heartbeat.last_heartbeat_ms > 0) {
            if ((current_time - g_pi_cmd_heartbeat.last_heartbeat_ms) > SAFETY_HEARTBEAT_MS) {
                safety_state_trigger_fault(FAULT_HEARTBEAT);
            }
        }
    }

    /* --- 2. State Transition Management --- */
    switch (g_safety_state.current_mode) {
        case STATE_INIT:
            /* Gyro calibration handles transition to STAND */
            break;

        case STATE_STAND:
            /* If Pi requests ACTIVE mode and heartbeat is healthy, transition */
            if (g_pi_cmd_heartbeat.mode_request == STATE_ACTIVE && 
                g_pi_cmd_heartbeat.last_heartbeat_ms > 0 &&
                (current_time - g_pi_cmd_heartbeat.last_heartbeat_ms) <= SAFETY_HEARTBEAT_MS) {
                g_safety_state.current_mode = STATE_ACTIVE;
                g_safety_state.mode_timer_ms = current_time;
            }
            break;

        case STATE_ACTIVE:
            /* Transition back to STAND if Pi requests it */
            if (g_pi_cmd_heartbeat.mode_request == STATE_STAND) {
                g_safety_state.current_mode = STATE_STAND;
                g_safety_state.mode_timer_ms = current_time;
            }
            /* Transition to CLIMB if requested */
            else if (g_pi_cmd_heartbeat.mode_request == STATE_CLIMB) {
                g_safety_state.current_mode = STATE_CLIMB;
                g_safety_state.mode_timer_ms = current_time;
            }
            break;

        case STATE_CLIMB:
            /* Transition back to ACTIVE or STAND if requested */
            if (g_pi_cmd_heartbeat.mode_request == STATE_ACTIVE) {
                g_safety_state.current_mode = STATE_ACTIVE;
                g_safety_state.mode_timer_ms = current_time;
            } else if (g_pi_cmd_heartbeat.mode_request == STATE_STAND) {
                g_safety_state.current_mode = STATE_STAND;
                g_safety_state.mode_timer_ms = current_time;
            }
            break;

        case STATE_FAULT:
            /* System is locked down in FAULT mode. Requires system reset to recover. */
            break;

        default:
            g_safety_state.current_mode = STATE_FAULT;
            break;
    }
    
    (void)gyro_y_rads; /* Reserved for future rate-based fault detection */
    (void)dt;          /* Unused */
}

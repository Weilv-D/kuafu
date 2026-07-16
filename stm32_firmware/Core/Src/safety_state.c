#include "safety_state.h"
#include "pin_config.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

SafetyState_t g_safety_state;

static uint32_t calibration_samples_count = 0U;
static float calibration_sum[3] = {0.0f, 0.0f, 0.0f};
static uint32_t overtemp_started_ms = 0U;
static uint8_t overtemp_active = 0U;

void safety_state_init(void) {
    memset(&g_safety_state, 0, sizeof(g_safety_state));
    g_safety_state.current_mode = STATE_INIT;
    calibration_samples_count = 0U;
    calibration_sum[0] = 0.0f;
    calibration_sum[1] = 0.0f;
    calibration_sum[2] = 0.0f;
    overtemp_started_ms = 0U;
    overtemp_active = 0U;
}

void safety_state_trigger_fault(FaultMask_t fault) {
    g_safety_state.fault_mask |= fault;
    g_safety_state.current_mode = STATE_FAULT;
}

void safety_state_gyro_calib_update(float gx, float gy, float gz, uint32_t now_ms) {
    (void)now_ms;
    if (g_safety_state.is_gyro_calibrated) {
        return;
    }

    calibration_sum[0] += gx;
    calibration_sum[1] += gy;
    calibration_sum[2] += gz;
    ++calibration_samples_count;

    if (calibration_samples_count >= 1000U) {
        g_safety_state.gyro_calib_offset[0] = calibration_sum[0] / 1000.0f;
        g_safety_state.gyro_calib_offset[1] = calibration_sum[1] / 1000.0f;
        g_safety_state.gyro_calib_offset[2] = calibration_sum[2] / 1000.0f;
        g_safety_state.is_gyro_calibrated = 1U;
    }
}

static FaultMask_t runtime_faults(const SafetyInputs_t *inputs) {
    FaultMask_t faults = FAULT_NONE;
    if (fabsf(inputs->pitch_rad) > SAFETY_MAX_PITCH_RAD) {
        faults |= FAULT_TILT;
    }
    if (fabsf(inputs->pitch_rate_rads) > SAFETY_MAX_PITCH_RATE_RAD_S) {
        faults |= FAULT_PITCH_RATE;
    }
    if (inputs->max_temp_c > SAFETY_MAX_TEMP_C) {
        if (!overtemp_active) {
            overtemp_active = 1U;
            overtemp_started_ms = inputs->now_ms;
        } else if ((uint32_t)(inputs->now_ms - overtemp_started_ms) >=
                   SAFETY_OVERTEMP_DEBOUNCE_MS) {
            faults |= FAULT_OVERTEMP;
        }
    } else {
        overtemp_active = 0U;
    }
    if (!inputs->imu_fresh) {
        faults |= FAULT_IMU;
    }
    if (!inputs->wheel_l_fresh) {
        faults |= FAULT_WHEEL_LEFT;
    }
    if (!inputs->wheel_r_fresh) {
        faults |= FAULT_WHEEL_RIGHT;
    }
    if (!inputs->servos_fresh) {
        faults |= FAULT_SERVO;
    }
    return faults;
}

SafetyDecision_t safety_state_update(const SafetyInputs_t *inputs) {
    SafetyDecision_t decision = {0U, 0U};
    FaultMask_t faults;

    if (inputs == NULL) {
        safety_state_trigger_fault(FAULT_INTERNAL);
        return decision;
    }
    if ((uint8_t)g_safety_state.current_mode > (uint8_t)STATE_FAULT ||
        inputs->requested_mode > (uint8_t)STATE_FAULT) {
        safety_state_trigger_fault(FAULT_INTERNAL);
        return decision;
    }
    if (inputs->requested_mode == (uint8_t)STATE_FAULT) {
        safety_state_trigger_fault(FAULT_EMERGENCY);
        return decision;
    }
    if (g_safety_state.current_mode == STATE_FAULT) {
        return decision;
    }

    faults = runtime_faults(inputs);
    if (g_safety_state.current_mode == STATE_INIT) {
        if ((faults & (FAULT_TILT | FAULT_PITCH_RATE | FAULT_OVERTEMP)) != 0U) {
            safety_state_trigger_fault(faults & (FAULT_TILT | FAULT_PITCH_RATE | FAULT_OVERTEMP));
            return decision;
        }
        if (inputs->startup_ready && inputs->gyro_calibrated && inputs->imu_fresh &&
            inputs->wheel_l_fresh && inputs->wheel_r_fresh && inputs->servos_fresh) {
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        }
        return decision;
    }

    if (faults != FAULT_NONE) {
        safety_state_trigger_fault(faults);
        return decision;
    }

    if (g_safety_state.current_mode == STATE_STAND) {
        if (inputs->requested_mode == (uint8_t)STATE_ACTIVE &&
            inputs->link_compatible && inputs->heartbeat_fresh) {
            g_safety_state.current_mode = STATE_ACTIVE;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        }
    } else if (g_safety_state.current_mode == STATE_ACTIVE) {
        if (!inputs->link_compatible || !inputs->heartbeat_fresh) {
            decision.enter_hold = 1U;
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        } else if (inputs->requested_mode == (uint8_t)STATE_STAND) {
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        } else if (inputs->requested_mode == (uint8_t)STATE_CLIMB) {
            g_safety_state.current_mode = STATE_CLIMB;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        }
        if (!inputs->action_fresh) {
            decision.clear_action = 1U;
        }
    } else if (g_safety_state.current_mode == STATE_CLIMB) {
        if (!inputs->link_compatible || !inputs->heartbeat_fresh) {
            decision.enter_hold = 1U;
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        } else if (inputs->requested_mode == (uint8_t)STATE_ACTIVE) {
            g_safety_state.current_mode = STATE_ACTIVE;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        } else if (inputs->requested_mode == (uint8_t)STATE_STAND) {
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
        }
    } else {
        safety_state_trigger_fault(FAULT_INTERNAL);
    }

    return decision;
}

uint8_t safety_state_legacy_fault_mask(void) {
    uint8_t legacy = (uint8_t)(g_safety_state.fault_mask & 0xFFU);
    if ((g_safety_state.fault_mask & ~((FaultMask_t)0xFFU)) != 0U) {
        legacy |= 0x80U;
    }
    return legacy;
}

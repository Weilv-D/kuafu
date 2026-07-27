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

/* Consecutive stale ticks before a freshness fault latches. */
static uint8_t imu_stale_ticks = 0U;
static uint8_t servo_stale_ticks = 0U;
static uint8_t wheel_l_stale_ticks = 0U;
static uint8_t wheel_r_stale_ticks = 0U;

void safety_state_init(void) {
    memset(&g_safety_state, 0, sizeof(g_safety_state));
    g_safety_state.current_mode = STATE_INIT;
    calibration_samples_count = 0U;
    calibration_sum[0] = 0.0f;
    calibration_sum[1] = 0.0f;
    calibration_sum[2] = 0.0f;
    overtemp_started_ms = 0U;
    overtemp_active = 0U;
    imu_stale_ticks = 0U;
    servo_stale_ticks = 0U;
    wheel_l_stale_ticks = 0U;
    wheel_r_stale_ticks = 0U;
}

void safety_state_trigger_fault(FaultMask_t fault) {
    g_safety_state.fault_mask |= fault;
    g_safety_state.current_mode = STATE_FAULT;
}

/* Stillness gate for gyro bias calibration: a stationary BMI088 reads well
 * below 0.08 rad/s (~4.6 deg/s) on all axes.  Moving samples are skipped (not
 * reset) so intermittent wobble does not poison the estimate.  Startup now
 * self-recovers instead of latching FAULT, so this only needs to be long
 * enough for a good bias mean: 1000 still samples @250 Hz ~= 4 s untouched. */
#define GYRO_CALIB_STILL_RAD_S 0.08f
#define GYRO_CALIB_SAMPLES 1000U

void safety_state_gyro_calib_update(float gx, float gy, float gz, uint32_t now_ms) {
    (void)now_ms;
    if (g_safety_state.is_gyro_calibrated) {
        return;
    }

    if (fabsf(gx) > GYRO_CALIB_STILL_RAD_S ||
        fabsf(gy) > GYRO_CALIB_STILL_RAD_S ||
        fabsf(gz) > GYRO_CALIB_STILL_RAD_S) {
        /* Skip moving samples WITHOUT resetting the accumulation: gyro bias
         * is orientation-independent, so still samples collected around
         * intermittent wobble (e.g. bench stand vibration) remain valid.
         * Resetting here would make calibration impossible on any surface
         * that is not perfectly still. */
        return;
    }

    calibration_sum[0] += gx;
    calibration_sum[1] += gy;
    calibration_sum[2] += gz;
    ++calibration_samples_count;

    if (calibration_samples_count >= GYRO_CALIB_SAMPLES) {
        g_safety_state.gyro_calib_offset[0] = calibration_sum[0] / (float)GYRO_CALIB_SAMPLES;
        g_safety_state.gyro_calib_offset[1] = calibration_sum[1] / (float)GYRO_CALIB_SAMPLES;
        g_safety_state.gyro_calib_offset[2] = calibration_sum[2] / (float)GYRO_CALIB_SAMPLES;
        g_safety_state.is_gyro_calibrated = 1U;
    }
}

static uint8_t freshness_under_grace(uint32_t now_ms) {
    return (g_safety_state.mode_grace_until_ms != 0U &&
            (uint32_t)(now_ms - g_safety_state.mode_grace_until_ms) > (uint32_t)(1UL << 31));
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

    if (inputs->imu_fresh) {
        imu_stale_ticks = 0U;
    } else if (imu_stale_ticks < UINT8_MAX) {
        ++imu_stale_ticks;
    }
    if (inputs->wheel_l_fresh) {
        wheel_l_stale_ticks = 0U;
    } else if (wheel_l_stale_ticks < UINT8_MAX) {
        ++wheel_l_stale_ticks;
    }
    if (inputs->wheel_r_fresh) {
        wheel_r_stale_ticks = 0U;
    } else if (wheel_r_stale_ticks < UINT8_MAX) {
        ++wheel_r_stale_ticks;
    }
    if (inputs->servos_fresh) {
        servo_stale_ticks = 0U;
    } else if (servo_stale_ticks < UINT8_MAX) {
        ++servo_stale_ticks;
    }

    /* Freshness faults require debounce unless we are in a mode-transition grace window. */
    if (!freshness_under_grace(inputs->now_ms) &&
        imu_stale_ticks >= SAFETY_FRESHNESS_DEBOUNCE_TICKS) {
        faults |= FAULT_IMU;
    }
    if (!freshness_under_grace(inputs->now_ms) &&
        wheel_l_stale_ticks >= SAFETY_FRESHNESS_DEBOUNCE_TICKS) {
        faults |= FAULT_WHEEL_LEFT;
    }
    if (!freshness_under_grace(inputs->now_ms) &&
        wheel_r_stale_ticks >= SAFETY_FRESHNESS_DEBOUNCE_TICKS) {
        faults |= FAULT_WHEEL_RIGHT;
    }
    if (!freshness_under_grace(inputs->now_ms) &&
        servo_stale_ticks >= SAFETY_FRESHNESS_DEBOUNCE_TICKS) {
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
    faults = runtime_faults(inputs);

    if (g_safety_state.current_mode == STATE_FAULT) {
        /* Transient freshness faults (wheel/servo telemetry loss) self-clear
         * once the device resumes replying, so a momentary bus blip at power-on
         * or under load does not permanently disable balance.  Serious faults
         * (tilt, pitch-rate, overtemp, IMU, emergency, init, internal) stay
         * latched until reset as before. */
        const FaultMask_t transient =
            (FaultMask_t)(FAULT_WHEEL_LEFT | FAULT_WHEEL_RIGHT | FAULT_SERVO);
        if ((faults & FAULT_WHEEL_LEFT) == 0U) {
            g_safety_state.fault_mask &= ~((FaultMask_t)FAULT_WHEEL_LEFT);
        }
        if ((faults & FAULT_WHEEL_RIGHT) == 0U) {
            g_safety_state.fault_mask &= ~((FaultMask_t)FAULT_WHEEL_RIGHT);
        }
        if ((faults & FAULT_SERVO) == 0U) {
            g_safety_state.fault_mask &= ~((FaultMask_t)FAULT_SERVO);
        }
        if ((g_safety_state.fault_mask & ~transient) != 0U) {
            return decision;   /* still seriously faulted: stay in FAULT */
        }
        if (g_safety_state.fault_mask == 0U) {
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        }
        return decision;
    }

    if (g_safety_state.current_mode == STATE_INIT) {
        if ((faults & (FAULT_TILT | FAULT_PITCH_RATE | FAULT_OVERTEMP)) != 0U) {
            safety_state_trigger_fault(faults & (FAULT_TILT | FAULT_PITCH_RATE | FAULT_OVERTEMP));
            return decision;
        }
        if (inputs->startup_ready && inputs->gyro_calibrated && inputs->imu_fresh &&
            inputs->wheel_l_fresh && inputs->wheel_r_fresh && inputs->servos_fresh) {
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
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
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        }
    } else if (g_safety_state.current_mode == STATE_ACTIVE) {
        if (!inputs->link_compatible || !inputs->heartbeat_fresh) {
            decision.enter_hold = 1U;
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        } else if (inputs->requested_mode == (uint8_t)STATE_STAND) {
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        } else if (inputs->requested_mode == (uint8_t)STATE_CLIMB) {
            g_safety_state.current_mode = STATE_CLIMB;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        }
        if (!inputs->action_fresh) {
            decision.clear_action = 1U;
        }
    } else if (g_safety_state.current_mode == STATE_CLIMB) {
        if (!inputs->link_compatible || !inputs->heartbeat_fresh) {
            decision.enter_hold = 1U;
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        } else if (inputs->requested_mode == (uint8_t)STATE_ACTIVE) {
            g_safety_state.current_mode = STATE_ACTIVE;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        } else if (inputs->requested_mode == (uint8_t)STATE_STAND) {
            g_safety_state.current_mode = STATE_STAND;
            g_safety_state.mode_timer_ms = inputs->now_ms;
            g_safety_state.mode_grace_until_ms =
                inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
            imu_stale_ticks = 0U;
            wheel_l_stale_ticks = 0U;
            wheel_r_stale_ticks = 0U;
            servo_stale_ticks = 0U;
        }
    } else {
        safety_state_trigger_fault(FAULT_INTERNAL);
    }

    return decision;
}

/* The legacy 8-bit mask folds every fault at bit 8 and above (PITCH_RATE,
 * INIT, INTERNAL) into 0x80, which deliberately collides with the low-byte
 * FAULT_WHEEL_RIGHT bit.  Consumers that need the exact cause must read the
 * full 32-bit fault_mask from the health telemetry frame instead. */
uint8_t safety_state_legacy_fault_mask(void) {
    uint8_t legacy = (uint8_t)(g_safety_state.fault_mask & 0xFFU);
    if ((g_safety_state.fault_mask & ~((FaultMask_t)0xFFU)) != 0U) {
        legacy |= 0x80U;
    }
    return legacy;
}

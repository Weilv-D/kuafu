#include "safety_state.h"
#include "pin_config.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

SafetyState_t g_safety_state;

static uint32_t calibration_samples_count = 0U;
static float calibration_sum[3] = {0.0f, 0.0f, 0.0f};
static float calibration_sum_sq[3] = {0.0f, 0.0f, 0.0f};
static uint32_t calibration_last_ms = 0U;
static uint8_t calibration_have_timestamp = 0U;
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
    calibration_sum_sq[0] = 0.0f;
    calibration_sum_sq[1] = 0.0f;
    calibration_sum_sq[2] = 0.0f;
    calibration_last_ms = 0U;
    calibration_have_timestamp = 0U;
    overtemp_started_ms = 0U;
    overtemp_active = 0U;
    imu_stale_ticks = 0U;
    servo_stale_ticks = 0U;
    wheel_l_stale_ticks = 0U;
    wheel_r_stale_ticks = 0U;
}

void safety_state_trigger_fault(FaultMask_t fault) {
    if (fault == FAULT_NONE) return;
    g_safety_state.fault_mask |= fault;
    g_safety_state.current_mode = STATE_FAULT;
}

/* Stillness gate for gyro bias calibration: a stationary BMI088 reads well
 * below 0.08 rad/s (~4.6 deg/s) on all axes.  Moving samples are SKIPPED, not
 * reset: bench-stand wobble must not make calibration impossible on any real
 * surface, and an unstable robot balances (uncalibrated) while the window
 * accumulates in the background.  Two guards bound the window instead:
 * a >50 ms sample gap restarts it (the IMU path runs at ~1 kHz), and the
 * completed window must pass a per-axis variance gate. */
#define GYRO_CALIB_STILL_RAD_S 0.08f
#define GYRO_CALIB_SAMPLES 1000U
#define GYRO_CALIB_MAX_GAP_MS 50U
#define GYRO_CALIB_MAX_VARIANCE 0.0025f

void safety_state_gyro_calib_update(float gx, float gy, float gz, uint32_t now_ms) {
    uint32_t n;
    float mean;
    float variance;
    if (g_safety_state.is_gyro_calibrated) return;

    /* Never allow NaN/Inf to poison the running sums.  This is deliberately a
     * local guard: balance remains available while a bad sample is discarded. */
    if (!isfinite(gx) || !isfinite(gy) || !isfinite(gz)) return;
    if (calibration_have_timestamp && now_ms == calibration_last_ms) return;
    if (calibration_have_timestamp &&
        (uint32_t)(now_ms - calibration_last_ms) > GYRO_CALIB_MAX_GAP_MS) {
        calibration_samples_count = 0U;
        calibration_sum[0] = calibration_sum[1] = calibration_sum[2] = 0.0f;
        calibration_sum_sq[0] = calibration_sum_sq[1] = calibration_sum_sq[2] = 0.0f;
    }
    calibration_last_ms = now_ms;
    calibration_have_timestamp = 1U;

    if (fabsf(gx) > GYRO_CALIB_STILL_RAD_S ||
        fabsf(gy) > GYRO_CALIB_STILL_RAD_S ||
        fabsf(gz) > GYRO_CALIB_STILL_RAD_S) {
        return;
    }

    calibration_sum[0] += gx;
    calibration_sum[1] += gy;
    calibration_sum[2] += gz;
    calibration_sum_sq[0] += gx * gx;
    calibration_sum_sq[1] += gy * gy;
    calibration_sum_sq[2] += gz * gz;
    ++calibration_samples_count;

    if (calibration_samples_count < GYRO_CALIB_SAMPLES) return;
    n = calibration_samples_count;
    mean = calibration_sum[0] / (float)n;
    variance = calibration_sum_sq[0] / (float)n - mean * mean;
    /* A constant signal can round to a tiny negative variance; that is float
     * error, not inconsistency.  Only genuinely excessive spread restarts. */
    if (variance < 0.0f) variance = 0.0f;
    if (variance > GYRO_CALIB_MAX_VARIANCE) goto reset_window;
    mean = calibration_sum[1] / (float)n;
    variance = calibration_sum_sq[1] / (float)n - mean * mean;
    if (variance < 0.0f) variance = 0.0f;
    if (variance > GYRO_CALIB_MAX_VARIANCE) goto reset_window;
    mean = calibration_sum[2] / (float)n;
    variance = calibration_sum_sq[2] / (float)n - mean * mean;
    if (variance < 0.0f) variance = 0.0f;
    if (variance > GYRO_CALIB_MAX_VARIANCE) goto reset_window;

    g_safety_state.gyro_calib_offset[0] = calibration_sum[0] / (float)n;
    g_safety_state.gyro_calib_offset[1] = calibration_sum[1] / (float)n;
    g_safety_state.gyro_calib_offset[2] = calibration_sum[2] / (float)n;
    g_safety_state.is_gyro_calibrated = 1U;
    return;

reset_window:
    calibration_samples_count = 0U;
    calibration_sum[0] = calibration_sum[1] = calibration_sum[2] = 0.0f;
    calibration_sum_sq[0] = calibration_sum_sq[1] = calibration_sum_sq[2] = 0.0f;
}

static uint8_t freshness_under_grace(uint32_t now_ms) {
    return (g_safety_state.mode_grace_until_ms != 0U &&
            (uint32_t)(now_ms - g_safety_state.mode_grace_until_ms) > (uint32_t)(1UL << 31));
}

/* Device-driven transitions (INIT->STAND, FAULT->STAND recovery) genuinely
 * change what the buses and the mode machine are asking of the devices, so
 * they earn the freshness-fault grace window and a stale-counter reset.
 * Command-driven transitions (Pi mode requests, link-loss demotion to STAND)
 * change no device state: granting them the same reset would let a Pi that
 * flaps mode_request at >10 Hz keep the four stale counters cleared and the
 * grace window permanently open, suppressing freshness faults (wheel loss,
 * servo loss) indefinitely while the wheels stay enabled.  Those transitions
 * therefore only record the time. */
static void enter_mode_with_grace(RobotMode_t mode, const SafetyInputs_t *inputs) {
    g_safety_state.current_mode = mode;
    g_safety_state.mode_timer_ms = inputs->now_ms;
    g_safety_state.mode_grace_until_ms =
        inputs->now_ms + SAFETY_MODE_TRANSITION_GRACE_MS;
    imu_stale_ticks = 0U;
    wheel_l_stale_ticks = 0U;
    wheel_r_stale_ticks = 0U;
    servo_stale_ticks = 0U;
}

static void enter_mode_command_driven(RobotMode_t mode, const SafetyInputs_t *inputs) {
    g_safety_state.current_mode = mode;
    g_safety_state.mode_timer_ms = inputs->now_ms;
}

static FaultMask_t runtime_faults(const SafetyInputs_t *inputs) {
    FaultMask_t faults = FAULT_NONE;
    if (!isfinite(inputs->pitch_rad)) faults |= FAULT_TILT;
    else if (fabsf(inputs->pitch_rad) > SAFETY_MAX_PITCH_RAD) faults |= FAULT_TILT;
    if (!isfinite(inputs->pitch_rate_rads)) faults |= FAULT_PITCH_RATE;
    else if (fabsf(inputs->pitch_rate_rads) > SAFETY_MAX_PITCH_RATE_RAD_S) {
        faults |= FAULT_PITCH_RATE;
    }
    if (!isfinite(inputs->max_temp_c)) faults |= FAULT_OVERTEMP;
    else if (inputs->max_temp_c > SAFETY_MAX_TEMP_C) {
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
    SafetyDecision_t decision = {0U, 0U, 0U, 0U};
    FaultMask_t faults;

    if (inputs == NULL) {
        safety_state_trigger_fault(FAULT_INTERNAL);
        decision.fault_active = 1U;
        decision.fault_latched = 1U;
        return decision;
    }
    if ((uint8_t)g_safety_state.current_mode > (uint8_t)STATE_FAULT ||
        inputs->requested_mode > (uint8_t)STATE_FAULT) {
        safety_state_trigger_fault(FAULT_INTERNAL);
        decision.fault_active = 1U;
        decision.fault_latched = 1U;
        return decision;
    }
    if (inputs->requested_mode == (uint8_t)STATE_FAULT) {
        safety_state_trigger_fault(FAULT_EMERGENCY);
        decision.fault_active = 1U;
        decision.fault_latched = 1U;
        return decision;
    }
    faults = runtime_faults(inputs);
    if (faults != FAULT_NONE) decision.fault_active = 1U;

    if (g_safety_state.current_mode == STATE_FAULT) {
        /* Re-evaluate and combine faults before considering recovery.  The old
         * code only examined the old mask, so a new tilt/NaN/over-temperature
         * fault arriving during transient recovery could be lost. */
        /* Preserve every newly observed cause, including a second transient
         * device fault while another transient fault is recovering. */
        g_safety_state.fault_mask |= faults;
        if ((g_safety_state.fault_mask & SAFETY_SERIOUS_FAULTS) != 0U) {
            decision.fault_latched = 1U;
            return decision;
        }
        if ((faults & FAULT_WHEEL_LEFT) == 0U) {
            g_safety_state.fault_mask &= ~((FaultMask_t)FAULT_WHEEL_LEFT);
        }
        if ((faults & FAULT_WHEEL_RIGHT) == 0U) {
            g_safety_state.fault_mask &= ~((FaultMask_t)FAULT_WHEEL_RIGHT);
        }
        if ((faults & FAULT_SERVO) == 0U) {
            g_safety_state.fault_mask &= ~((FaultMask_t)FAULT_SERVO);
        }
        if (g_safety_state.fault_mask == 0U) {
            enter_mode_with_grace(STATE_STAND, inputs);
        }
        return decision;
    }

    if (g_safety_state.current_mode == STATE_INIT) {
        if ((faults & SAFETY_SERIOUS_FAULTS) != 0U) {
            safety_state_trigger_fault(faults & SAFETY_SERIOUS_FAULTS);
            decision.fault_active = 1U;
            decision.fault_latched = 1U;
            return decision;
        }
        if (inputs->startup_ready && inputs->imu_fresh &&
            inputs->wheel_l_fresh && inputs->wheel_r_fresh && inputs->servos_fresh) {
            enter_mode_with_grace(STATE_STAND, inputs);
        }
        return decision;
    }

    if (faults != FAULT_NONE) {
        safety_state_trigger_fault(faults);
        decision.fault_active = 1U;
        decision.fault_latched = (uint8_t)((faults & SAFETY_SERIOUS_FAULTS) != 0U);
        return decision;
    }

    if (g_safety_state.current_mode == STATE_STAND) {
        /* CLIMB is deliberately reachable only through ACTIVE (the arming
         * step), see docs/architecture/system.md operating-modes table; an
         * INIT request has no inbound transition at all and is ignored. */
        if (inputs->requested_mode == (uint8_t)STATE_ACTIVE &&
            inputs->link_compatible && inputs->heartbeat_fresh) {
            enter_mode_command_driven(STATE_ACTIVE, inputs);
        }
    } else if (g_safety_state.current_mode == STATE_ACTIVE) {
        if (!inputs->link_compatible || !inputs->heartbeat_fresh) {
            decision.enter_hold = 1U;
            enter_mode_command_driven(STATE_STAND, inputs);
        } else if (inputs->requested_mode == (uint8_t)STATE_STAND ||
                   inputs->requested_mode == (uint8_t)STATE_INIT) {
            /* STAND and "leave the motion mode": INIT is a boot state with no
             * inbound transition, so a sender that asks for it while the robot
             * walks is asking to stop.  Honouring it as STAND keeps the mode
             * consistent with the request; ignoring it would leave the robot
             * driving while the sender believes it was obeyed. */
            enter_mode_command_driven(STATE_STAND, inputs);
        } else if (inputs->requested_mode == (uint8_t)STATE_CLIMB) {
            enter_mode_command_driven(STATE_CLIMB, inputs);
        }
        if (!inputs->action_fresh) {
            decision.clear_action = 1U;
        }
    } else if (g_safety_state.current_mode == STATE_CLIMB) {
        if (!inputs->link_compatible || !inputs->heartbeat_fresh) {
            decision.enter_hold = 1U;
            enter_mode_command_driven(STATE_STAND, inputs);
        } else if (inputs->requested_mode == (uint8_t)STATE_ACTIVE) {
            enter_mode_command_driven(STATE_ACTIVE, inputs);
        } else if (inputs->requested_mode == (uint8_t)STATE_STAND ||
                   inputs->requested_mode == (uint8_t)STATE_INIT) {
            enter_mode_command_driven(STATE_STAND, inputs);
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

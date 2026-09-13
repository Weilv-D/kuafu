#include "lqr_controller.h"
#include "pin_config.h"
#include <math.h>

static float clamp_float(float value, float lower, float upper) {
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

static float limit_wheel_torque(float torque, float omega) {
    /* Torque-speed envelope: linearly derate available torque from stall at
     * zero speed to zero at no-load speed.  This is the manufacturer's
     * speed-torque curve and is symmetric for both wheels. */
    float available = DDSM_MAX_TORQUE_NM * (1.0f - fabsf(omega) / KUAFU_OMEGA_NOLOAD);
    available = clamp_float(available, 0.0f, DDSM_MAX_TORQUE_NM);
    return clamp_float(torque, -available, available);
}

static float wrap_angle(float value) {
    return atan2f(sinf(value), cosf(value));
}

static float jerk_limited_ref(float target, float *value, float *accel,
                              float max_accel, float max_jerk, float dt) {
    float target_accel = clamp_float((target - *value) / dt, -max_accel, max_accel);
    *accel += clamp_float(target_accel - *accel, -max_jerk * dt, max_jerk * dt);
    *value += *accel * dt;
    return *value;
}

void lqr_init(LQRController_t *controller) {
    controller->K[0] = LQR_K0;
    controller->K[1] = LQR_K1;
    controller->K[2] = LQR_K2;
    controller->K[3] = LQR_K3;
    controller->Ki = LQI_KI;
    lqr_reset(controller, 0.0f, 0.0f);
}

void lqr_reset(LQRController_t *controller, float x_est, float yaw_rad) {
    controller->x_est = x_est;
    controller->x_ref = x_est;
    controller->x_int = 0.0f;
    controller->v_ref = 0.0f;
    controller->v_accel = 0.0f;
    controller->yaw_ref = yaw_rad;
    controller->w_ref = 0.0f;
    controller->w_accel = 0.0f;
    controller->air_sustain_ticks = 0U;
}

float lqr_update_elapsed_dt(LQRController_t *controller,
                            float elapsed_dt_s,
                            float pitch_rad,
                            float pitch_rate_rads,
                            float wheel_vel_l_rads,
                            float wheel_vel_r_rads,
                            float yaw_rad,
                            float yaw_rate_rads,
                            float vx_cmd,
                            float wz_cmd,
                            float delta_tau_common,
                            float delta_tau_yaw,
                            float *out_tau_l,
                            float *out_tau_r) {
    float dt = clamp_float(elapsed_dt_s, 0.0f, LQR_ELAPSED_DT_MAX);
    if (dt <= 0.0f) {
        *out_tau_l = 0.0f;
        *out_tau_r = 0.0f;
        return 0.0f;
    }
    float wheel_vel_avg_rads = 0.5f * (wheel_vel_l_rads + wheel_vel_r_rads);
    float vx = wheel_vel_avg_rads * WHEEL_RADIUS_M;
    controller->x_est += vx * dt;
    float v_ref = jerk_limited_ref(vx_cmd, &controller->v_ref, &controller->v_accel, 2.0f, 8.0f, dt);
    float w_ref = jerk_limited_ref(wz_cmd, &controller->w_ref, &controller->w_accel, 4.0f, 16.0f, dt);
    controller->x_ref += v_ref * dt;
    if (wz_cmd == 0.0f) {
        /* No turn command: track the measured heading so the yaw loop
         * degenerates to pure rate damping.  Mahony yaw has no magnetometer
         * correction and drifts with residual gyro bias; a fixed yaw_ref turns
         * that drift into a growing YAW_KP * yaw_error differential torque
         * that spins the robot in place (observed: constant tau_yaw ~0.07 Nm
         * with both wheels counter-rotating at +-3.3 rad/s). */
        controller->yaw_ref = yaw_rad;
    } else {
        controller->yaw_ref = wrap_angle(controller->yaw_ref + w_ref * dt);
    }

    float x_error = controller->x_est - controller->x_ref;
    /* Airborne re-anchor: when the wheels are unloaded (robot lifted or
     * hopping) they spin near no-load speed with the body staying level —
     * a state real driving never produces (the VX_SAFE_LIMIT brake below
     * engages first).  Integrating that free spin into x_est would leave a
     * multi-meter x_error, and on touchdown K0 would slam maximum torque —
     * the observed "落地冲撞" (landing charge).  While airborne, drag x_ref
     * along with x_est and freeze the integral so the position loop starts
     * from ~zero the instant the wheels grip again.
     *
     * The signature must ACCUMULATE before re-anchoring: a ground rush
     * crosses |vx|>0.8 with |pitch|<0.2 as well (the stand-entry transient
     * toward the lean-offset equilibrium spikes past the brake), and an
     * instantaneous trigger drags the anchor along with the runaway robot,
     * killing the position loop and the LQI integral mid-rush — the observed
     * "疯狂后退" (frantic reverse).  The counter is a leaky bucket (+1 on
     * signature, -1 otherwise) rather than reset-to-zero: hand-held air
     * jitter crosses the speed threshold in bursts, and a hard reset lets
     * those bursts integrate meters of phantom x_est that slam the wheels on
     * touchdown ("一落地就爆冲"). */
    #define AIR_VX_MIN 0.8f
    #define AIR_PITCH_MAX 0.2f
    #define AIR_SUSTAIN_TICKS 62U  /* 62 * 4 ms ≈ 250 ms of accumulated signature */
    if (fabsf(vx) > AIR_VX_MIN && fabsf(pitch_rad) < AIR_PITCH_MAX) {
        if (controller->air_sustain_ticks < (uint16_t)(2U * AIR_SUSTAIN_TICKS)) {
            ++controller->air_sustain_ticks;
        }
    } else if (controller->air_sustain_ticks > 0U) {
        --controller->air_sustain_ticks;
    }
    if (controller->air_sustain_ticks >= AIR_SUSTAIN_TICKS) {
        controller->x_ref = controller->x_est;
        controller->x_int = 0.0f;
        x_error = 0.0f;
    } else if (fabsf(pitch_rad) > PITCH_FADE_END_RAD) {
        /* Anti-windup: if the body is tipped past the recoverable window, the
         * position integral is meaningless (the robot is falling, not drifting)
         * and only makes the wheels accelerate uncontrollably. */
        controller->x_int = 0.0f;
    } else {
        /* Tight integral clamp: the LQI integral gain (Ki) is large enough that
         * a 0.25 m·s accumulation produces multi-newton force overshoot.  Limit
         * to 0.05 so the integral never contributes more than ~10% of rated
         * wheel torque — enough to cancel steady-state drift without sprinting. */
        controller->x_int = clamp_float(controller->x_int + x_error * dt,
                                             -LQI_INTEGRAL_CLAMP, LQI_INTEGRAL_CLAMP);
    }
    float force = -(controller->K[0] * x_error
                    + controller->K[1] * pitch_rad
                    + controller->K[2] * (vx - v_ref)
                    + controller->K[3] * pitch_rate_rads)
                  - controller->Ki * controller->x_int;
    /* Speed safety limit: if the body is moving faster than a safe cruise
     * speed, add a strong opposing force to brake.  This prevents the
     * "爆冲" (runaway sprint) that happens when the pitch integral or a
     * large tilt drives both wheels to maximum speed. */
    const float VX_SAFE_LIMIT = 0.5f;  /* m/s, ~half of no-load wheel speed */
    if (vx > VX_SAFE_LIMIT) {
        force -= 50.0f * (vx - VX_SAFE_LIMIT);
    } else if (vx < -VX_SAFE_LIMIT) {
        force -= 50.0f * (vx + VX_SAFE_LIMIT);
    }
    float tau_pitch = force * WHEEL_RADIUS_M * 0.5f;

    float yaw_error = wrap_angle(controller->yaw_ref - yaw_rad);
    /* Clamp yaw torque to a fraction of rated: the yaw PD gains act on gyro
     * noise and motor asymmetry.  Without a cap the differential torque can
     * exceed the pitch torque and spin the robot in place ("旋转"). */
    float tau_yaw = clamp_float(
        YAW_KP * yaw_error + YAW_KD * (w_ref - yaw_rate_rads),
        -0.1f, 0.1f);
    float tau_common_residual = clamp_float(delta_tau_common, -1.0f, 1.0f) * TAU_WHEEL_RATED;
    float tau_yaw_residual = clamp_float(delta_tau_yaw, -1.0f, 1.0f) * TAU_WHEEL_RATED;
    /* DDSM315 +torque produces backward body-frame motion on this robot, so
     * the LQR output is negated.  Right motor is mirror-mounted
     * (WHEEL_DIR_R = -1 at dispatch).  Both wheels get the same pitch torque;
     * yaw is applied as a differential. */
    *out_tau_l = limit_wheel_torque(
        -tau_pitch + tau_yaw - tau_common_residual + tau_yaw_residual,
        wheel_vel_l_rads);
    *out_tau_r = limit_wheel_torque(
        -tau_pitch - tau_yaw - tau_common_residual - tau_yaw_residual,
        wheel_vel_r_rads);
    return dt;
}

void lqr_update(LQRController_t *controller,
                float pitch_rad,
                float pitch_rate_rads,
                float wheel_vel_l_rads,
                float wheel_vel_r_rads,
                float yaw_rad,
                float yaw_rate_rads,
                float vx_cmd,
                float wz_cmd,
                float delta_tau_common,
                float delta_tau_yaw,
                float *out_tau_l,
                float *out_tau_r) {
    (void)lqr_update_elapsed_dt(controller, BASE_DT, pitch_rad, pitch_rate_rads,
                                wheel_vel_l_rads, wheel_vel_r_rads, yaw_rad,
                                yaw_rate_rads, vx_cmd, wz_cmd, delta_tau_common,
                                delta_tau_yaw, out_tau_l, out_tau_r);
}

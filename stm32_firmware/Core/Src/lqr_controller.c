#include "lqr_controller.h"
#include "pin_config.h"
#include <math.h>

static float clamp_float(float value, float lower, float upper) {
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

static float limit_wheel_torque(float torque, float omega) {
    float available = DDSM_MAX_TORQUE_NM * (1.0f - fabsf(omega) / KUAFU_OMEGA_NOLOAD);
    available = clamp_float(available, 0.0f, DDSM_MAX_TORQUE_NM);
    return clamp_float(torque, -available, available);
}

static float wrap_angle(float value) {
    return atan2f(sinf(value), cosf(value));
}

static float jerk_limited_ref(float target, float *value, float *accel,
                              float max_accel, float max_jerk) {
    float target_accel = clamp_float((target - *value) / BASE_DT, -max_accel, max_accel);
    *accel += clamp_float(target_accel - *accel, -max_jerk * BASE_DT, max_jerk * BASE_DT);
    *value += *accel * BASE_DT;
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
    float wheel_vel_avg_rads = 0.5f * (wheel_vel_l_rads + wheel_vel_r_rads);
    float vx = wheel_vel_avg_rads * WHEEL_RADIUS_M;
    controller->x_est += vx * BASE_DT;
    float v_ref = jerk_limited_ref(vx_cmd, &controller->v_ref, &controller->v_accel, 2.0f, 8.0f);
    float w_ref = jerk_limited_ref(wz_cmd, &controller->w_ref, &controller->w_accel, 4.0f, 16.0f);
    controller->x_ref += v_ref * BASE_DT;
    controller->yaw_ref = wrap_angle(controller->yaw_ref + w_ref * BASE_DT);

    float x_error = controller->x_est - controller->x_ref;
    /* Anti-windup: if the body is tipped past the recoverable window, the
     * position integral is meaningless (the robot is falling, not drifting)
     * and only makes the wheels accelerate uncontrollably. */
    if (fabsf(pitch_rad) > PITCH_FADE_END_RAD) {
        controller->x_int = 0.0f;
    } else {
        controller->x_int = clamp_float(controller->x_int + x_error * BASE_DT, -0.25f, 0.25f);
    }
    float force = -(controller->K[0] * x_error
                    + controller->K[1] * pitch_rad
                    + controller->K[2] * (vx - v_ref)
                    + controller->K[3] * pitch_rate_rads)
                  - controller->Ki * controller->x_int;
    float tau_pitch = force * WHEEL_RADIUS_M * 0.5f;

    float yaw_error = wrap_angle(controller->yaw_ref - yaw_rad);
    float tau_yaw = YAW_KP * yaw_error + YAW_KD * (w_ref - yaw_rate_rads);
    float tau_common_residual = clamp_float(delta_tau_common, -1.0f, 1.0f) * TAU_WHEEL_RATED;
    float tau_yaw_residual = clamp_float(delta_tau_yaw, -1.0f, 1.0f) * TAU_WHEEL_RATED;
    /* DDSM315 +torque produces backward body-frame motion on this robot, so
     * the LQR output is negated.  The right motor is mounted mirrored
     * (WHEEL_DIR_R = -1 at dispatch).
     *
     *   τ_l_body = -(τ_pitch − τ_yaw) − τ_common + τ_yaw_residual
     *   τ_r_body = -(τ_pitch + τ_yaw) − τ_common − τ_yaw_residual
     *
     * After WHEEL_DIR_L(+1), WHEEL_DIR_R(−1):
     *   motor_l = −τ_pitch + τ_yaw − τ_common + τ_yaw_residual
     *   motor_r = +τ_pitch + τ_yaw + τ_common + τ_yaw_residual
     *
     * On the reversed right motor +motor → robot backward.  For a right turn
     * (τ_yaw > 0) left motor becomes less backward (more forward), right motor
     * becomes more forward (more backward on robot) — correct. */
    *out_tau_l = limit_wheel_torque(
        -tau_pitch + tau_yaw - tau_common_residual + tau_yaw_residual,
        wheel_vel_l_rads);
    *out_tau_r = limit_wheel_torque(
        -tau_pitch - tau_yaw - tau_common_residual - tau_yaw_residual,
        wheel_vel_r_rads);
}

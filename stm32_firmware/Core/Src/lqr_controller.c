#include "lqr_controller.h"
#include "pin_config.h"
#include <math.h>

/* Helper function for clipping float values */
static float clamp_float(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

void lqr_init(LQRController_t *controller) {
    /* Initialize LQR gains from config (mirrors kuafu_physics.LQR_K) */
    controller->K[0] = LQR_K0; /* Position gain  (unused, x ≡ 0) */
    controller->K[1] = LQR_K1; /* Pitch gain     (N/rad)          */
    controller->K[2] = LQR_K2; /* Velocity gain  (N/(m/s))        */
    controller->K[3] = LQR_K3; /* Pitch rate gain (N/(rad/s))     */
}

void lqr_update(LQRController_t *controller,
                float pitch_rad,
                float pitch_rate_rads,
                float wheel_vel_avg_rads,
                float yaw_rate_rads,
                float delta_tau_l,
                float delta_tau_r,
                float *out_tau_l,
                float *out_tau_r) {

    /* 1. Forward linear velocity from average wheel speed (ẋ = ω_avg × R).
     *    Mirrors kuafu_mjx_env: xdot = lin_vel_local[0] (body-frame x). */
    float xdot = wheel_vel_avg_rads * WHEEL_RADIUS_M;

    /* 2. LQR state feedback: state = [x=0, θ, ẋ, θ̇], F = -(K · state).
     *    x is held at 0 (same as sim), so K[0] is dropped. */
    float F = -(controller->K[1] * pitch_rad +
                controller->K[2] * xdot +
                controller->K[3] * pitch_rate_rads);

    /* 3. Convert ground force F to per-wheel torque: τ_pitch = F × R / 2. */
    float tau_pitch = F * WHEEL_RADIUS_M / 2.0f;

    /* 4. Conditional yaw damping: only when |ωz| < threshold.
     *    τ_diff = clip(-YAW_KD · ωz, ±τ_rated); otherwise 0. */
    float tau_diff = 0.0f;
    if (fabsf(yaw_rate_rads) < YAW_DAMP_THRESH) {
        tau_diff = clamp_float(-YAW_KD * yaw_rate_rads,
                               -TAU_WHEEL_RATED, TAU_WHEEL_RATED);
    }

    /* 5. RL residual overlay (mirrors kuafu_mjx_env.step):
     *    τ_L = τ_pitch + τ_diff + Δτ_L × τ_rated
     *    τ_R = τ_pitch - τ_diff + Δτ_R × τ_rated */
    float tau_l = tau_pitch + tau_diff + delta_tau_l * TAU_WHEEL_RATED;
    float tau_r = tau_pitch - tau_diff + delta_tau_r * TAU_WHEEL_RATED;

    /* 6. Clamp to DDSM315 stall torque limit (±1.1 N-m). */
    tau_l = clamp_float(tau_l, -DDSM_MAX_TORQUE_NM, DDSM_MAX_TORQUE_NM);
    tau_r = clamp_float(tau_r, -DDSM_MAX_TORQUE_NM, DDSM_MAX_TORQUE_NM);

    *out_tau_l = tau_l;
    *out_tau_r = tau_r;
}

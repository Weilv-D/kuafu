#ifndef LQR_CONTROLLER_H
#define LQR_CONTROLLER_H

#include <stdint.h>

/**
 * Base-layer LQR-lite controller (mirrors rl/env/kuafu_mjx_env._lqr_balance).
 *
 *   pitch: LQR state [x, θ, ẋ, θ̇] -> ground force F, split equally to both
 *          wheels as τ_pitch = F * R / 2. The position state x is held at 0
 *          (same as the RL sim), so only [θ, ẋ, θ̇] participate in feedback.
 *   yaw:   conditional damping only. When |ωz| < YAW_DAMP_THRESH a small
 *          resistive differential torque τ_diff = clip(-YAW_KD·ωz, ±τ_rated)
 *          is applied; otherwise it is disabled to avoid pitch coupling.
 *
 * RL residual overlay (mirrors kuafu_mjx_env.step):
 *   τ_L = τ_pitch + τ_diff + Δτ_L × τ_rated
 *   τ_R = τ_pitch - τ_diff + Δτ_R × τ_rated
 * then clamped to ±DDSM_MAX_TORQUE_NM (stall torque).
 */
typedef struct {
    float K[4];             /* LQR gains: [K_x, K_θ, K_ẋ, K_θ̇] (from pin_config.h) */
} LQRController_t;

/**
 * @brief Initializes the LQR controller structure with default gains.
 */
void lqr_init(LQRController_t *controller);

/**
 * @brief Computes the wheel torque commands from the LQR base layer plus RL
 *        residual deltas. Identical control law to kuafu_mjx_env._lqr_balance
 *        followed by the residual overlay in step().
 *
 * @param controller        Pointer to the LQR controller structure.
 * @param pitch_rad         Present body pitch angle (rad, from Mahony fusion).
 * @param pitch_rate_rads   Present body pitch rate (rad/s, gyro Y).
 * @param wheel_vel_avg_rads Average wheel angular velocity (rad/s) = (ω_L+ω_R)/2.
 *                          Used to derive forward speed ẋ = ω_avg × R.
 * @param yaw_rate_rads     Present yaw rate (rad/s, gyro Z) for conditional damping.
 * @param delta_tau_l       RL residual for left wheel, normalized [-1, 1].
 * @param delta_tau_r       RL residual for right wheel, normalized [-1, 1].
 * @param out_tau_l         Output: target torque for left wheel (N-m).
 * @param out_tau_r         Output: target torque for right wheel (N-m).
 */
void lqr_update(LQRController_t *controller,
                float pitch_rad,
                float pitch_rate_rads,
                float wheel_vel_avg_rads,
                float yaw_rate_rads,
                float delta_tau_l,
                float delta_tau_r,
                float *out_tau_l,
                float *out_tau_r);

#endif /* LQR_CONTROLLER_H */

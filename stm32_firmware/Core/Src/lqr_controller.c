#include "lqr_controller.h"
#include "pin_config.h"
#include <math.h>

#define WHEEL_RADIUS_M  0.03908f /* Wheel radius in meters (39.08mm) */

void lqr_init(LQRController_t *controller) {
    /* Initialize LQR gains from config */
    controller->K[0] = LQR_K0; /* Position gain (N/m) */
    controller->K[1] = LQR_K1; /* Pitch gain (N/rad) */
    controller->K[2] = LQR_K2; /* Velocity gain (N/(m/s)) */
    controller->K[3] = LQR_K3; /* Pitch rate gain (N/(rad/s)) */
    
    lqr_reset(controller);
}

void lqr_reset(LQRController_t *controller) {
    controller->pos_error = 0.0f;
    controller->target_velocity = 0.0f;
    controller->target_yaw_rate = 0.0f;
}

void lqr_update(LQRController_t *controller,
                float pitch_rad,
                float pitch_rate_rads,
                float left_vel_rads,
                float right_vel_rads,
                float delta_tau_l,
                float delta_tau_r,
                float dt,
                float *out_tau_l,
                float *out_tau_r) {
    
    /* 1. Calculate linear wheel velocities (v = omega * R) */
    float v_left = left_vel_rads * WHEEL_RADIUS_M;
    float v_right = right_vel_rads * WHEEL_RADIUS_M;

    /* 2. Calculate average forward linear velocity of the robot */
    float v_forward = (v_left + v_right) / 2.0f;

    /* 3. Integrate position error */
    float velocity_error = v_forward - controller->target_velocity;
    controller->pos_error += velocity_error * dt;

    /* Anti-windup / Saturation on integrated position error to prevent instability */
    if (controller->pos_error > 1.5f)  controller->pos_error = 1.5f;
    if (controller->pos_error < -1.5f) controller->pos_error = -1.5f;

    /* 4. LQR State Feedback Equation (computes balancing force F in Newtons)
     * States: x = [pos_error, pitch, velocity_error, pitch_rate]
     * Force: F = -K * x */
    float force_lqr = -(controller->K[0] * controller->pos_error +
                        controller->K[1] * pitch_rad +
                        controller->K[2] * velocity_error +
                        controller->K[3] * pitch_rate_rads);

    /* 5. Convert linear force F to total balancing torque (tau = F * R) */
    float total_torque_lqr = force_lqr * WHEEL_RADIUS_M;

    /* 6. Distribute torque to left and right wheels and add steering component if active */
    /* Pi yaw rate control is usually added as a differential torque */
    float steer_torque = 0.0f;
    if (fabsf(controller->target_yaw_rate) > 0.01f) {
        /* Simple P controller for yaw tracking: steer_torque = K_yaw * (yaw_rate_err) */
        float K_yaw = 0.2f; /* Steer torque gain */
        /* Average yaw rate is proportional to difference in wheel speeds: 
         * omega_yaw = (v_right - v_left) / track_width. 
         * But since we want to follow target yaw rate, a simple differential torque is: */
        steer_torque = K_yaw * (controller->target_yaw_rate);
    }

    float tau_l_cmd = (total_torque_lqr / 2.0f) - steer_torque + delta_tau_l;
    float tau_r_cmd = (total_torque_lqr / 2.0f) + steer_torque + delta_tau_r;

    /* 7. Clamp commands to DDSM315 peak stall torque limit */
    if (tau_l_cmd > DDSM_MAX_TORQUE_NM)  tau_l_cmd = DDSM_MAX_TORQUE_NM;
    if (tau_l_cmd < -DDSM_MAX_TORQUE_NM) tau_l_cmd = -DDSM_MAX_TORQUE_NM;
    if (tau_r_cmd > DDSM_MAX_TORQUE_NM)  tau_r_cmd = DDSM_MAX_TORQUE_NM;
    if (tau_r_cmd < -DDSM_MAX_TORQUE_NM) tau_r_cmd = -DDSM_MAX_TORQUE_NM;

    *out_tau_l = tau_l_cmd;
    *out_tau_r = tau_r_cmd;
}

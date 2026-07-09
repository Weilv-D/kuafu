#ifndef LQR_CONTROLLER_H
#define LQR_CONTROLLER_H

typedef struct {
    float K[4];             /* LQR Gains: [Kp_pos, Kp_theta, Kd_pos, Kd_theta] */
    float pos_error;        /* Integrated position error e_x (meters) */
    float target_velocity;  /* Target forward velocity cmd (m/s) */
    float target_yaw_rate;  /* Target yaw angular velocity cmd (rad/s) */
} LQRController_t;

/**
 * @brief Initializes the LQR controller structure with default gains.
 * 
 * @param controller Pointer to the controller structure.
 */
void lqr_init(LQRController_t *controller);

/**
 * @brief Resets the integrated position error of the LQR controller.
 * 
 * @param controller Pointer to the controller structure.
 */
void lqr_reset(LQRController_t *controller);

/**
 * @brief Computes the LQR control torques for the left and right wheels.
 * 
 * @param controller Pointer to the LQR controller structure.
 * @param pitch_rad Present body pitch angle (radians, from Mahony fusion).
 * @param pitch_rate_rads Present body pitch rate (radians per second, gyro Y).
 * @param left_vel_rads Present left wheel angular velocity (rad/s).
 * @param right_vel_rads Present right wheel angular velocity (rad/s).
 * @param delta_tau_l Input residual command torque for Left wheel (N-m).
 * @param delta_tau_r Input residual command torque for Right wheel (N-m).
 * @param dt Loop time step (seconds, e.g., 0.004 for 250Hz).
 * @param out_tau_l Output: target torque command for Left wheel (N-m).
 * @param out_tau_r Output: target torque command for Right wheel (N-m).
 */
void lqr_update(LQRController_t *controller,
                float pitch_rad,
                float pitch_rate_rads,
                float left_vel_rads,
                float right_vel_rads,
                float delta_tau_l,
                float delta_tau_r,
                float dt,
                float *out_tau_l,
                float *out_tau_r);

#endif /* LQR_CONTROLLER_H */

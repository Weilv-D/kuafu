#ifndef LQR_CONTROLLER_H
#define LQR_CONTROLLER_H

typedef struct {
    float K[4];       /* F=-K[x-x_ref, pitch, vx-v_ref, pitch_rate]-Ki*x_int */
    float Ki;
    float x_est;
    float x_ref;
    float x_int;
    float v_ref;
    float v_accel;
    float yaw_ref;
    float w_ref;
    float w_accel;
} LQRController_t;

void lqr_init(LQRController_t *controller);
void lqr_reset(LQRController_t *controller, float x_est, float yaw_rad);

/* 250 Hz baseline controller. Commands are SI units; residuals are normalized
 * common/yaw actions in [-1,1].  Positive yaw residual means right torque > left
 * torque, which is +wz under the repository frame contract. */
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
                float *out_tau_r);

#endif

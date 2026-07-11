#include "mahony.h"
#include <math.h>

void mahony_init(MahonyFilter_t *filter, float Kp, float Ki) {
    filter->Kp = Kp;
    filter->Ki = Ki;
    mahony_reset(filter);
}

void mahony_reset(MahonyFilter_t *filter) {
    filter->q0 = 1.0f;
    filter->q1 = 0.0f;
    filter->q2 = 0.0f;
    filter->q3 = 0.0f;
    filter->eInt[0] = 0.0f;
    filter->eInt[1] = 0.0f;
    filter->eInt[2] = 0.0f;
    filter->roll = 0.0f;
    filter->pitch = 0.0f;
    filter->yaw = 0.0f;
}

void mahony_update(MahonyFilter_t *filter, float ax, float ay, float az, float gx, float gy, float gz, float dt) {
    float q0 = filter->q0;
    float q1 = filter->q1;
    float q2 = filter->q2;
    float q3 = filter->q3;

    /* Helper variables to avoid redundant math */
    float q0q0 = q0 * q0;
    float q0q1 = q0 * q1;
    float q0q2 = q0 * q2;
    float q0q3 = q0 * q3;
    float q1q1 = q1 * q1;
    float q1q2 = q1 * q2;
    float q1q3 = q1 * q3;
    float q2q2 = q2 * q2;
    float q2q3 = q2 * q3;
    float q3q3 = q3 * q3;

    /* Normalize accelerometer measurements */
    float norm = sqrtf(ax * ax + ay * ay + az * az);
    if (norm > 0.0f) {
        ax /= norm;
        ay /= norm;
        az /= norm;
    } else {
        return; /* Avoid division by zero if accelerometer readings are zero */
    }

    /* Estimated direction of gravity (v) in body frame */
    float vx = 2.0f * (q1q3 - q0q2);
    float vy = 2.0f * (q0q1 + q2q3);
    float vz = q0q0 - q1q1 - q2q2 + q3q3;

    /* Error is cross product between estimated gravity and measured acceleration direction */
    float ex = (ay * vz - az * vy);
    float ey = (az * vx - ax * vz);
    float ez = (ax * vy - ay * vx);

    /* Compute integral feedback if enabled */
    if (filter->Ki > 0.0f) {
        filter->eInt[0] += ex * dt;
        filter->eInt[1] += ey * dt;
        filter->eInt[2] += ez * dt;
        
        /* Apply feedback correction to gyro inputs */
        gx += filter->Kp * ex + filter->Ki * filter->eInt[0];
        gy += filter->Kp * ey + filter->Ki * filter->eInt[1];
        gz += filter->Kp * ez + filter->Ki * filter->eInt[2];
    } else {
        /* Clear integral error buffer */
        filter->eInt[0] = 0.0f;
        filter->eInt[1] = 0.0f;
        filter->eInt[2] = 0.0f;

        gx += filter->Kp * ex;
        gy += filter->Kp * ey;
        gz += filter->Kp * ez;
    }

    /* Integrate quaternion rate of change */
    float dq0 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    float dq1 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    float dq2 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    float dq3 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

    q0 += dq0 * dt;
    q1 += dq1 * dt;
    q2 += dq2 * dt;
    q3 += dq3 * dt;

    /* Normalize quaternion */
    norm = sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (norm > 0.0f) {
        filter->q0 = q0 / norm;
        filter->q1 = q1 / norm;
        filter->q2 = q2 / norm;
        filter->q3 = q3 / norm;
    }

    /* Recalculate Euler Angles */
    filter->roll = atan2f(2.0f * (filter->q0 * filter->q1 + filter->q2 * filter->q3), 1.0f - 2.0f * (filter->q1 * filter->q1 + filter->q2 * filter->q2));

    /* pitch = asin(2*(q0*q2 - q3*q1)), clamped to [-1,1] to avoid domain error.
     * Matches RL sim: arcsin(clip(2*(qw*qy - qx*qz), -0.999999, 0.999999)) */
    float pitch_arg = 2.0f * (filter->q0 * filter->q2 - filter->q3 * filter->q1);
    if (pitch_arg > 1.0f) pitch_arg = 1.0f;
    if (pitch_arg < -1.0f) pitch_arg = -1.0f;
    filter->pitch = asinf(pitch_arg);

    filter->yaw = atan2f(2.0f * (filter->q0 * filter->q3 + filter->q1 * filter->q2), 1.0f - 2.0f * (filter->q2 * filter->q2 + filter->q3 * filter->q3));
}

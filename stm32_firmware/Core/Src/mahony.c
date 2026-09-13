#include "mahony.h"
#include <math.h>
#include <stddef.h>

#define MAHONY_EINT_MAX 0.5f
#define MAHONY_DT_MIN 0.0001f
#define MAHONY_DT_MAX 0.05f
#define MAHONY_GYRO_ONLY_MAX_S 0.10f
#define GRAVITY_M_S2 9.80665f
#define ACCEL_MIN_NORM (0.50f * GRAVITY_M_S2)
#define ACCEL_MAX_NORM (1.50f * GRAVITY_M_S2)
#define MAHONY_MAX_INNOVATION 0.25f

static float clampf_local(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static uint8_t finite3(float x, float y, float z) {
    return (uint8_t)(isfinite(x) && isfinite(y) && isfinite(z));
}

static void update_euler(MahonyFilter_t *filter) {
    float pitch_arg;
    filter->roll = atan2f(2.0f * (filter->q0 * filter->q1 + filter->q2 * filter->q3),
                          1.0f - 2.0f * (filter->q1 * filter->q1 + filter->q2 * filter->q2));
    pitch_arg = 2.0f * (filter->q0 * filter->q2 - filter->q3 * filter->q1);
    pitch_arg = clampf_local(pitch_arg, -1.0f, 1.0f);
    filter->pitch = asinf(pitch_arg);
    filter->yaw = atan2f(2.0f * (filter->q0 * filter->q3 + filter->q1 * filter->q2),
                         1.0f - 2.0f * (filter->q2 * filter->q2 + filter->q3 * filter->q3));
}

void mahony_init(MahonyFilter_t *filter, float Kp, float Ki) {
    if (filter == NULL) return;
    filter->Kp = isfinite(Kp) && Kp >= 0.0f ? Kp : 0.0f;
    filter->Ki = isfinite(Ki) && Ki >= 0.0f ? Ki : 0.0f;
    mahony_reset(filter);
}

void mahony_reset(MahonyFilter_t *filter) {
    if (filter == NULL) return;
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
    filter->gyro_only_elapsed_s = 0.0f;
    filter->control_valid = 0U;
    filter->last_status = MAHONY_UPDATE_REJECTED;
}

MahonyUpdateStatus_t mahony_update_validated(MahonyFilter_t *filter,
                                             float ax, float ay, float az,
                                             float gx, float gy, float gz,
                                             float dt,
                                             uint8_t accel_valid,
                                             uint8_t gyro_valid) {
    float q0, q1, q2, q3, norm, ex = 0.0f, ey = 0.0f, ez = 0.0f;
    float vx, vy, vz, innovation;
    uint8_t use_accel;
    MahonyUpdateStatus_t status;
    if (filter == NULL || !gyro_valid || !finite3(gx, gy, gz) || !isfinite(dt)) {
        if (filter != NULL) {
            filter->control_valid = 0U;
            filter->last_status = MAHONY_UPDATE_REJECTED;
        }
        return MAHONY_UPDATE_REJECTED;
    }
    dt = clampf_local(dt, MAHONY_DT_MIN, MAHONY_DT_MAX);
    use_accel = (uint8_t)(accel_valid && finite3(ax, ay, az));
    norm = sqrtf(ax * ax + ay * ay + az * az);
    if (!use_accel || !isfinite(norm) || norm < ACCEL_MIN_NORM || norm > ACCEL_MAX_NORM) {
        use_accel = 0U;
    }
    if (!use_accel) {
        filter->gyro_only_elapsed_s += dt;
        if (filter->gyro_only_elapsed_s > MAHONY_GYRO_ONLY_MAX_S) {
            filter->control_valid = 0U;
            filter->last_status = MAHONY_UPDATE_REJECTED;
            return MAHONY_UPDATE_REJECTED;
        }
        status = MAHONY_UPDATE_GYRO_ONLY;
    } else {
        filter->gyro_only_elapsed_s = 0.0f;
        ax /= norm;
        ay /= norm;
        az /= norm;
        q0 = filter->q0; q1 = filter->q1; q2 = filter->q2; q3 = filter->q3;
        vx = 2.0f * (q1 * q3 - q0 * q2);
        vy = 2.0f * (q0 * q1 + q2 * q3);
        vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
        ex = ay * vz - az * vy;
        ey = az * vx - ax * vz;
        ez = ax * vy - ay * vx;
        innovation = sqrtf(ex * ex + ey * ey + ez * ez);
        if (!isfinite(innovation) || innovation > MAHONY_MAX_INNOVATION) {
            /* A large innovation is treated as a transient acceleration or
             * disagreement: propagate gyro only, still under the time bound. */
            filter->gyro_only_elapsed_s += dt;
            if (filter->gyro_only_elapsed_s > MAHONY_GYRO_ONLY_MAX_S) {
                filter->control_valid = 0U;
                filter->last_status = MAHONY_UPDATE_REJECTED;
                return MAHONY_UPDATE_REJECTED;
            }
            status = MAHONY_UPDATE_GYRO_ONLY;
        } else {
            if (filter->Ki > 0.0f) {
                filter->eInt[0] = clampf_local(filter->eInt[0] + ex * dt, -MAHONY_EINT_MAX, MAHONY_EINT_MAX);
                filter->eInt[1] = clampf_local(filter->eInt[1] + ey * dt, -MAHONY_EINT_MAX, MAHONY_EINT_MAX);
                filter->eInt[2] = clampf_local(filter->eInt[2] + ez * dt, -MAHONY_EINT_MAX, MAHONY_EINT_MAX);
                gx += filter->Kp * ex + filter->Ki * filter->eInt[0];
                gy += filter->Kp * ey + filter->Ki * filter->eInt[1];
                gz += filter->Kp * ez + filter->Ki * filter->eInt[2];
            } else {
                filter->eInt[0] = 0.0f; filter->eInt[1] = 0.0f; filter->eInt[2] = 0.0f;
                gx += filter->Kp * ex; gy += filter->Kp * ey; gz += filter->Kp * ez;
            }
            status = MAHONY_UPDATE_FULL;
        }
    }
    q0 = filter->q0; q1 = filter->q1; q2 = filter->q2; q3 = filter->q3;
    {
        float dq0 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
        float dq1 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
        float dq2 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
        float dq3 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);
        q0 += dq0 * dt; q1 += dq1 * dt; q2 += dq2 * dt; q3 += dq3 * dt;
    }
    norm = sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (!isfinite(norm) || norm < 0.5f) {
        filter->control_valid = 0U;
        filter->last_status = MAHONY_UPDATE_REJECTED;
        return MAHONY_UPDATE_REJECTED;
    }
    filter->q0 = q0 / norm; filter->q1 = q1 / norm;
    filter->q2 = q2 / norm; filter->q3 = q3 / norm;
    update_euler(filter);
    filter->control_valid = 1U;
    filter->last_status = status;
    return status;
}

void mahony_update(MahonyFilter_t *filter, float ax, float ay, float az,
                   float gx, float gy, float gz, float dt) {
    (void)mahony_update_validated(filter, ax, ay, az, gx, gy, gz, dt, 1U, 1U);
}

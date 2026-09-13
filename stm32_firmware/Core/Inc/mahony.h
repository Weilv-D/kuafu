#ifndef MAHONY_H
#define MAHONY_H

#include <stdint.h>

typedef enum {
    MAHONY_UPDATE_REJECTED = 0,
    MAHONY_UPDATE_FULL = 1,
    MAHONY_UPDATE_GYRO_ONLY = 2
} MahonyUpdateStatus_t;

typedef struct {
    float q0, q1, q2, q3;
    float Kp;
    float Ki;
    float eInt[3];
    float roll;
    float pitch;
    float yaw;
    float gyro_only_elapsed_s;
    uint8_t control_valid;
    MahonyUpdateStatus_t last_status;
} MahonyFilter_t;

void mahony_init(MahonyFilter_t *filter, float Kp, float Ki);
void mahony_reset(MahonyFilter_t *filter);

/* Safe update. Invalid gyro is always rejected. Invalid/out-of-range accel
 * temporarily permits bounded gyro-only propagation, without accel correction. */
MahonyUpdateStatus_t mahony_update_validated(MahonyFilter_t *filter,
                                             float ax, float ay, float az,
                                             float gx, float gy, float gz,
                                             float dt,
                                             uint8_t accel_valid,
                                             uint8_t gyro_valid);

/* Compatibility wrapper; callers that control actuators must use the return
 * status from mahony_update_validated instead. */
void mahony_update(MahonyFilter_t *filter, float ax, float ay, float az,
                   float gx, float gy, float gz, float dt);

#endif

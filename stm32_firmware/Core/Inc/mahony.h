#ifndef MAHONY_H
#define MAHONY_H

#include <math.h>

typedef struct {
    float q0, q1, q2, q3; /* Quaternion states */
    float Kp;             /* Proportional gain */
    float Ki;             /* Integral gain */
    float eInt[3];        /* Integral error buffer */
    float roll;           /* Roll angle (radians) */
    float pitch;          /* Pitch angle (radians) */
    float yaw;            /* Yaw angle (radians) */
} MahonyFilter_t;

/**
 * @brief Initializes the Mahony filter structure.
 * 
 * @param filter Pointer to the filter structure.
 * @param Kp Proportional gain.
 * @param Ki Integral gain.
 */
void mahony_init(MahonyFilter_t *filter, float Kp, float Ki);

/**
 * @brief Resets the filter to a known alignment.
 * 
 * @param filter Pointer to the filter structure.
 */
void mahony_reset(MahonyFilter_t *filter);

/**
 * @brief Updates the attitude filter with accelerometer and gyroscope measurements.
 * 
 * @param filter Pointer to the filter structure.
 * @param ax Accelerometer x reading (m/s^2 or g, normalized inside).
 * @param ay Accelerometer y reading.
 * @param az Accelerometer z reading.
 * @param gx Gyroscope x reading (radians per second).
 * @param gy Gyroscope y reading.
 * @param gz Gyroscope z reading.
 * @param dt Update time interval in seconds.
 */
void mahony_update(MahonyFilter_t *filter, float ax, float ay, float az, float gx, float gy, float gz, float dt);

#endif /* MAHONY_H */

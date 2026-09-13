#ifndef BMI088_H
#define BMI088_H

#include "stm32f4xx_hal.h"
#include "device_health.h"
#include <stdint.h>

#define BMI088_ACCEL_ADDR        0x18
#define BMI088_GYRO_ADDR         0x68

#define BMI088_ACC_CHIP_ID       0x00
#define BMI088_ACC_ERR_REG       0x02
#define BMI088_ACC_STATUS        0x03
#define BMI088_ACC_X_LSB         0x12
#define BMI088_ACC_TEMP_MSB      0x22
#define BMI088_ACC_TEMP_LSB      0x23
#define BMI088_ACC_CONF          0x40
#define BMI088_ACC_RANGE         0x41
#define BMI088_ACC_PWR_CONF      0x7C
#define BMI088_ACC_PWR_CTRL      0x7D
#define BMI088_ACC_SOFTRESET     0x7E

#define BMI088_GYRO_CHIP_ID      0x00
#define BMI088_GYRO_X_LSB        0x02
#define BMI088_GYRO_RANGE        0x0F
#define BMI088_GYRO_BANDWIDTH    0x10
#define BMI088_GYRO_INT_CTRL     0x15
#define BMI088_GYRO_INT3_INT4_IO_CONF 0x16
#define BMI088_GYRO_INT3_INT4_IO_MAP  0x18
#define BMI088_GYRO_SOFTRESET    0x14

typedef struct {
    float bias[3];
    float scale[3];
} BMI088AccelCalibration_t;

typedef struct {
    uint8_t accel_valid;
    uint8_t gyro_valid;
    uint8_t accel_fresh;
    uint8_t gyro_fresh;
    uint32_t accel_age_ms;
    uint32_t gyro_age_ms;
    uint32_t accel_sequence;
    uint32_t gyro_sequence;
} BMI088SampleValidity_t;

typedef struct {
    I2C_HandleTypeDef *hi2c;
    float accel[3];
    float gyro[3];
    float temperature;
    /* Legacy aggregate health is refreshed only after a valid accel+gyro pair. */
    DeviceHealth_t health;
    DeviceHealth_t accel_health;
    DeviceHealth_t gyro_health;
    BMI088AccelCalibration_t accel_calibration;
    uint32_t accel_last_valid_ms;
    uint32_t gyro_last_valid_ms;
    uint32_t accel_sequence;
    uint32_t gyro_sequence;
    uint8_t accel_sample_valid;
    uint8_t gyro_sample_valid;
    uint8_t init_state;
    uint8_t initialized;
    uint8_t init_attempts;
    uint32_t init_deadline_ms;
} BMI088_t;

#define BMI088_MAX_INIT_ATTEMPTS 5U
#define BMI088_DEFAULT_MAX_AGE_MS 20U

void bmi088_begin_init(BMI088_t *imu, I2C_HandleTypeDef *hi2c, uint32_t now_ms);
void bmi088_recover_bus(BMI088_t *imu);
int bmi088_init_step(BMI088_t *imu, uint32_t now_ms);
int bmi088_read_accel(BMI088_t *imu);
int bmi088_read_gyro(BMI088_t *imu);
int bmi088_read_temp(BMI088_t *imu);

void bmi088_set_accel_calibration(BMI088_t *imu,
                                  const BMI088AccelCalibration_t *calibration);
void bmi088_get_sample_validity(const BMI088_t *imu,
                                uint32_t now_ms,
                                uint32_t max_age_ms,
                                BMI088SampleValidity_t *validity);
uint8_t bmi088_accel_healthy(const BMI088_t *imu, uint32_t now_ms, uint32_t max_age_ms);
uint8_t bmi088_gyro_healthy(const BMI088_t *imu, uint32_t now_ms, uint32_t max_age_ms);

#endif

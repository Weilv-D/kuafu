#ifndef BMI088_H
#define BMI088_H

#include "stm32f4xx_hal.h"
#include "device_health.h"

/* I2C Device Addresses */
#define BMI088_ACCEL_ADDR        0x18
#define BMI088_GYRO_ADDR         0x68

/* Accel Register Map */
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

/* Gyro Register Map */
#define BMI088_GYRO_CHIP_ID      0x00
#define BMI088_GYRO_X_LSB        0x02
#define BMI088_GYRO_RANGE        0x0F
#define BMI088_GYRO_BANDWIDTH    0x10
#define BMI088_GYRO_INT_CTRL     0x15
#define BMI088_GYRO_INT3_INT4_IO_CONF 0x16
#define BMI088_GYRO_INT3_INT4_IO_MAP  0x18
#define BMI088_GYRO_SOFTRESET    0x14

typedef struct {
    I2C_HandleTypeDef *hi2c;
    float accel[3];          /* Accel X, Y, Z in m/s^2 */
    float gyro[3];           /* Gyro X, Y, Z in rad/s */
    float temperature;       /* Chip temperature in degC */
    DeviceHealth_t health;
    uint32_t init_deadline_ms;
    uint8_t init_state;
    uint8_t initialized;
    uint8_t init_attempts;   /* I2C failures seen during the current init sequence */
} BMI088_t;

#define BMI088_MAX_INIT_ATTEMPTS 5U   /* I2C failures tolerated before giving up */

/**
 * @brief Initializes the BMI088 IMU over I2C.
 *
 * @param imu Pointer to the device structure.
 * @param hi2c Pointer to initialized STM32 HAL I2C handle.
 * @return int 0 on success, -1 on failure.
 */
void bmi088_begin_init(BMI088_t *imu, I2C_HandleTypeDef *hi2c, uint32_t now_ms);

/* Recovers a locked I2C bus: deinitializes the peripheral, toggles 9 SCL
 * clocks as a GPIO to release a slave holding SDA low, then re-initializes.
 * Called automatically on init failure; also safe to call on demand. */
void bmi088_recover_bus(BMI088_t *imu);

/* Advances at most one register transaction when its deadline is reached.
 * Returns 1 when initialized, 0 while in progress, and -1 on a failed attempt. */
int bmi088_init_step(BMI088_t *imu, uint32_t now_ms);

/**
 * @brief Reads the raw accelerometer values and converts to m/s^2.
 * 
 * @param imu Pointer to the device structure.
 * @return int 0 on success, -1 on failure.
 */
int bmi088_read_accel(BMI088_t *imu);

/**
 * @brief Reads the raw gyroscope values and converts to rad/s.
 * 
 * @param imu Pointer to the device structure.
 * @return int 0 on success, -1 on failure.
 */
int bmi088_read_gyro(BMI088_t *imu);

/**
 * @brief Reads the accelerometer chip temperature and converts to degC.
 *
 * @param imu Pointer to the device structure.
 * @return int 0 on success, -1 on failure.
 */
int bmi088_read_temp(BMI088_t *imu);

#endif /* BMI088_H */

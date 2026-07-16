#include "bmi088.h"

#define GRAVITY_M_S2         9.80665f
#define PI                   3.14159265f

/* Conversion factors */
#define ACCEL_24G_SCALE      ((24.0f / 32768.0f) * GRAVITY_M_S2)
#define GYRO_2000_SCALE      ((2000.0f / 32768.0f) * (PI / 180.0f))

/* Helper write register function */
static HAL_StatusTypeDef bmi088_write_reg(I2C_HandleTypeDef *hi2c, uint8_t dev_addr, uint8_t reg_addr, uint8_t data) {
    return HAL_I2C_Mem_Write(hi2c, dev_addr << 1, reg_addr, I2C_MEMADD_SIZE_8BIT, &data, 1, 5);
}

/* Helper read register function */
static HAL_StatusTypeDef bmi088_read_reg(I2C_HandleTypeDef *hi2c, uint8_t dev_addr, uint8_t reg_addr, uint8_t *data) {
    return HAL_I2C_Mem_Read(hi2c, dev_addr << 1, reg_addr, I2C_MEMADD_SIZE_8BIT, data, 1, 5);
}

enum {
    BMI_INIT_IDLE = 0,
    BMI_INIT_ACC_RESET,
    BMI_INIT_ACC_PWR_CONF,
    BMI_INIT_ACC_PWR_CTRL,
    BMI_INIT_ACC_ID,
    BMI_INIT_ACC_RANGE,
    BMI_INIT_ACC_CONF,
    BMI_INIT_GYRO_RESET,
    BMI_INIT_GYRO_ID,
    BMI_INIT_GYRO_RANGE,
    BMI_INIT_GYRO_BW,
    BMI_INIT_GYRO_INT_CTRL,
    BMI_INIT_GYRO_IO_CONF,
    BMI_INIT_GYRO_MAP,
    BMI_INIT_DONE,
    BMI_INIT_FAILED
};

static int bmi_init_write(BMI088_t *imu, uint8_t addr, uint8_t reg, uint8_t value,
                          uint8_t next_state, uint32_t now_ms, uint32_t delay_ms) {
    if (bmi088_write_reg(imu->hi2c, addr, reg, value) != HAL_OK) {
        imu->init_state = BMI_INIT_FAILED;
        return -1;
    }
    imu->init_state = next_state;
    imu->init_deadline_ms = now_ms + delay_ms;
    return 0;
}

void bmi088_begin_init(BMI088_t *imu, I2C_HandleTypeDef *hi2c, uint32_t now_ms) {
    device_health_init(&imu->health);
    imu->hi2c = hi2c;
    imu->initialized = 0U;
    imu->init_state = BMI_INIT_ACC_RESET;
    imu->init_deadline_ms = now_ms;
}

int bmi088_init_step(BMI088_t *imu, uint32_t now_ms) {
    uint8_t chip_id = 0U;
    if (imu->initialized) {
        return 1;
    }
    if (imu->init_state == BMI_INIT_IDLE || imu->init_state == BMI_INIT_FAILED) {
        return -1;
    }
    if ((int32_t)(now_ms - imu->init_deadline_ms) < 0) {
        return 0;
    }

    switch (imu->init_state) {
        case BMI_INIT_ACC_RESET:
            return bmi_init_write(imu, BMI088_ACCEL_ADDR, BMI088_ACC_SOFTRESET, 0xB6,
                                  BMI_INIT_ACC_PWR_CONF, now_ms, 50U);
        case BMI_INIT_ACC_PWR_CONF:
            return bmi_init_write(imu, BMI088_ACCEL_ADDR, BMI088_ACC_PWR_CONF, 0x00,
                                  BMI_INIT_ACC_PWR_CTRL, now_ms, 5U);
        case BMI_INIT_ACC_PWR_CTRL:
            return bmi_init_write(imu, BMI088_ACCEL_ADDR, BMI088_ACC_PWR_CTRL, 0x04,
                                  BMI_INIT_ACC_ID, now_ms, 50U);
        case BMI_INIT_ACC_ID:
            if (bmi088_read_reg(imu->hi2c, BMI088_ACCEL_ADDR, BMI088_ACC_CHIP_ID, &chip_id) != HAL_OK ||
                chip_id != 0x1E) {
                imu->init_state = BMI_INIT_FAILED;
                return -1;
            }
            imu->init_state = BMI_INIT_ACC_RANGE;
            return 0;
        case BMI_INIT_ACC_RANGE:
            return bmi_init_write(imu, BMI088_ACCEL_ADDR, BMI088_ACC_RANGE, 0x03,
                                  BMI_INIT_ACC_CONF, now_ms, 2U);
        case BMI_INIT_ACC_CONF:
            return bmi_init_write(imu, BMI088_ACCEL_ADDR, BMI088_ACC_CONF, 0xAC,
                                  BMI_INIT_GYRO_RESET, now_ms, 2U);
        case BMI_INIT_GYRO_RESET:
            return bmi_init_write(imu, BMI088_GYRO_ADDR, BMI088_GYRO_SOFTRESET, 0xB6,
                                  BMI_INIT_GYRO_ID, now_ms, 100U);
        case BMI_INIT_GYRO_ID:
            if (bmi088_read_reg(imu->hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_CHIP_ID, &chip_id) != HAL_OK ||
                chip_id != 0x0F) {
                imu->init_state = BMI_INIT_FAILED;
                return -1;
            }
            imu->init_state = BMI_INIT_GYRO_RANGE;
            return 0;
        case BMI_INIT_GYRO_RANGE:
            return bmi_init_write(imu, BMI088_GYRO_ADDR, BMI088_GYRO_RANGE, 0x00,
                                  BMI_INIT_GYRO_BW, now_ms, 2U);
        case BMI_INIT_GYRO_BW:
            return bmi_init_write(imu, BMI088_GYRO_ADDR, BMI088_GYRO_BANDWIDTH, 0x02,
                                  BMI_INIT_GYRO_INT_CTRL, now_ms, 2U);
        case BMI_INIT_GYRO_INT_CTRL:
            return bmi_init_write(imu, BMI088_GYRO_ADDR, BMI088_GYRO_INT_CTRL, 0x80,
                                  BMI_INIT_GYRO_IO_CONF, now_ms, 1U);
        case BMI_INIT_GYRO_IO_CONF:
            return bmi_init_write(imu, BMI088_GYRO_ADDR, BMI088_GYRO_INT3_INT4_IO_CONF, 0x01,
                                  BMI_INIT_GYRO_MAP, now_ms, 1U);
        case BMI_INIT_GYRO_MAP:
            if (bmi_init_write(imu, BMI088_GYRO_ADDR, BMI088_GYRO_INT3_INT4_IO_MAP, 0x01,
                               BMI_INIT_DONE, now_ms, 1U) != 0) {
                return -1;
            }
            return 0;
        case BMI_INIT_DONE:
            imu->initialized = 1U;
            return 1;
        default:
            imu->init_state = BMI_INIT_FAILED;
            return -1;
    }
}

int bmi088_read_accel(BMI088_t *imu) {
    uint8_t buffer[6];

    /* Read 6 bytes of raw accelerometer data starting at BMI088_ACC_X_LSB (0x12) */
    if (HAL_I2C_Mem_Read(imu->hi2c, BMI088_ACCEL_ADDR << 1, BMI088_ACC_X_LSB, I2C_MEMADD_SIZE_8BIT, buffer, 6, 10) != HAL_OK) {
        device_health_mark_failure(&imu->health, DEVICE_FAILURE_TIMEOUT, 3U);
        return -1;
    }

    /* Combine raw values (two's complement int16) and scale to m/s^2 */
    int16_t raw_x = (int16_t)(((uint16_t)buffer[1] << 8) | buffer[0]);
    int16_t raw_y = (int16_t)(((uint16_t)buffer[3] << 8) | buffer[2]);
    int16_t raw_z = (int16_t)(((uint16_t)buffer[5] << 8) | buffer[4]);

    imu->accel[0] = (float)raw_x * ACCEL_24G_SCALE;
    imu->accel[1] = (float)raw_y * ACCEL_24G_SCALE;
    imu->accel[2] = (float)raw_z * ACCEL_24G_SCALE;

    return 0;
}

int bmi088_read_gyro(BMI088_t *imu) {
    uint8_t buffer[6];

    /* Read 6 bytes of raw gyroscope data starting at BMI088_GYRO_X_LSB (0x02) */
    if (HAL_I2C_Mem_Read(imu->hi2c, BMI088_GYRO_ADDR << 1, BMI088_GYRO_X_LSB, I2C_MEMADD_SIZE_8BIT, buffer, 6, 10) != HAL_OK) {
        device_health_mark_failure(&imu->health, DEVICE_FAILURE_TIMEOUT, 3U);
        return -1;
    }

    /* Combine raw values (two's complement int16) and scale to rad/s */
    int16_t raw_x = (int16_t)(((uint16_t)buffer[1] << 8) | buffer[0]);
    int16_t raw_y = (int16_t)(((uint16_t)buffer[3] << 8) | buffer[2]);
    int16_t raw_z = (int16_t)(((uint16_t)buffer[5] << 8) | buffer[4]);

    imu->gyro[0] = (float)raw_x * GYRO_2000_SCALE;
    imu->gyro[1] = (float)raw_y * GYRO_2000_SCALE;
    imu->gyro[2] = (float)raw_z * GYRO_2000_SCALE;
    device_health_mark_valid(&imu->health, HAL_GetTick());

    return 0;
}

int bmi088_read_temp(BMI088_t *imu) {
    uint8_t buffer[2];

    /* Temperature lives on the accelerometer die (TEMP_MSB:0x22, TEMP_LSB:0x23) */
    if (HAL_I2C_Mem_Read(imu->hi2c, BMI088_ACCEL_ADDR << 1, BMI088_ACC_TEMP_MSB, I2C_MEMADD_SIZE_8BIT, buffer, 2, 10) != HAL_OK) {
        device_health_mark_failure(&imu->health, DEVICE_FAILURE_TIMEOUT, 3U);
        return -1;
    }

    /* 11-bit signed value: MSB holds bits[10:3], LSB[7:5] hold bits[2:0] */
    uint16_t temp_uint11 = ((uint16_t)buffer[0] << 3) | (buffer[1] >> 5);
    int16_t temp_int11 = (temp_uint11 > 1023) ? (int16_t)temp_uint11 - 2048 : (int16_t)temp_uint11;

    /* Datasheet: Temperature = Temp_int11 * 0.125 degC + 23 degC */
    imu->temperature = (float)temp_int11 * 0.125f + 23.0f;

    return 0;
}

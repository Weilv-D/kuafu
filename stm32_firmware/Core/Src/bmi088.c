#include "bmi088.h"

#define GRAVITY_M_S2         9.80665f
#define PI                   3.14159265f

/* Conversion factors */
#define ACCEL_24G_SCALE      ((24.0f / 32768.0f) * GRAVITY_M_S2)
#define GYRO_2000_SCALE      ((2000.0f / 32768.0f) * (PI / 180.0f))

/* Helper write register function */
static HAL_StatusTypeDef bmi088_write_reg(I2C_HandleTypeDef *hi2c, uint8_t dev_addr, uint8_t reg_addr, uint8_t data) {
    return HAL_I2C_Mem_Write(hi2c, dev_addr << 1, reg_addr, I2C_MEMADD_SIZE_8BIT, &data, 1, 100);
}

/* Helper read register function */
static HAL_StatusTypeDef bmi088_read_reg(I2C_HandleTypeDef *hi2c, uint8_t dev_addr, uint8_t reg_addr, uint8_t *data) {
    return HAL_I2C_Mem_Read(hi2c, dev_addr << 1, reg_addr, I2C_MEMADD_SIZE_8BIT, data, 1, 100);
}

int bmi088_init(BMI088_t *imu, I2C_HandleTypeDef *hi2c) {
    imu->hi2c = hi2c;
    uint8_t chip_id = 0;

    /* ------------------------------------------------------------- */
    /* 1. Accelerometer Initialization                               */
    /* ------------------------------------------------------------- */
    /* Soft Reset Accelerometer */
    bmi088_write_reg(hi2c, BMI088_ACCEL_ADDR, BMI088_ACC_SOFTRESET, 0xB6);
    HAL_Delay(50); /* Recommended delay after reset */

    /* Power up Accelerometer (write 0x00 to ACC_PWR_CONF) */
    bmi088_write_reg(hi2c, BMI088_ACCEL_ADDR, BMI088_ACC_PWR_CONF, 0x00);
    HAL_Delay(5);

    /* Enable Accelerometer (write 0x03 to ACC_PWR_CTRL) */
    bmi088_write_reg(hi2c, BMI088_ACCEL_ADDR, BMI088_ACC_PWR_CTRL, 0x03);
    HAL_Delay(50);

    /* Read and verify Accel Chip ID */
    if (bmi088_read_reg(hi2c, BMI088_ACCEL_ADDR, BMI088_ACC_CHIP_ID, &chip_id) != HAL_OK || chip_id != 0x1E) {
        return -1; /* Accel identification failed */
    }

    /* Configure Accel Range to ±24g */
    bmi088_write_reg(hi2c, BMI088_ACCEL_ADDR, BMI088_ACC_RANGE, 0x03);
    HAL_Delay(2);

    /* Configure Accel ODR to 1600Hz and Bandwidth to 280Hz (Normal mode) */
    bmi088_write_reg(hi2c, BMI088_ACCEL_ADDR, BMI088_ACC_CONF, 0xAC);
    HAL_Delay(2);

    /* ------------------------------------------------------------- */
    /* 2. Gyroscope Initialization                                  */
    /* ------------------------------------------------------------- */
    /* Soft Reset Gyroscope */
    bmi088_write_reg(hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_SOFTRESET, 0xB6);
    HAL_Delay(100); /* Recommended delay after reset */

    /* Read and verify Gyro Chip ID */
    if (bmi088_read_reg(hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_CHIP_ID, &chip_id) != HAL_OK || chip_id != 0x0F) {
        return -1; /* Gyro identification failed */
    }

    /* Configure Gyro Range to ±2000 deg/s */
    bmi088_write_reg(hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_RANGE, 0x00);
    HAL_Delay(2);

    /* Configure Gyro ODR to 1000Hz, Filter Bandwidth to 116Hz */
    bmi088_write_reg(hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_BANDWIDTH, 0x02);
    HAL_Delay(2);

    /* Configure INT3 Pin for Gyroscope Data Ready Interrupt (Active High, Push-Pull) */
    bmi088_write_reg(hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_INT_CTRL, 0x80);        /* Enable Gyro DRDY int */
    HAL_Delay(1);
    bmi088_write_reg(hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_INT3_INT4_IO_CONF, 0x01); /* Set INT3: Push-Pull, Act High */
    HAL_Delay(1);
    bmi088_write_reg(hi2c, BMI088_GYRO_ADDR, BMI088_GYRO_INT3_INT4_IO_MAP, 0x01);  /* Map Gyro DRDY to INT3 */
    HAL_Delay(1);

    return 0;
}

int bmi088_read_accel(BMI088_t *imu) {
    uint8_t buffer[6];

    /* Read 6 bytes of raw accelerometer data starting at BMI088_ACC_X_LSB (0x12) */
    if (HAL_I2C_Mem_Read(imu->hi2c, BMI088_ACCEL_ADDR << 1, BMI088_ACC_X_LSB, I2C_MEMADD_SIZE_8BIT, buffer, 6, 10) != HAL_OK) {
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
        return -1;
    }

    /* Combine raw values (two's complement int16) and scale to rad/s */
    int16_t raw_x = (int16_t)(((uint16_t)buffer[1] << 8) | buffer[0]);
    int16_t raw_y = (int16_t)(((uint16_t)buffer[3] << 8) | buffer[2]);
    int16_t raw_z = (int16_t)(((uint16_t)buffer[5] << 8) | buffer[4]);

    imu->gyro[0] = (float)raw_x * GYRO_2000_SCALE;
    imu->gyro[1] = (float)raw_y * GYRO_2000_SCALE;
    imu->gyro[2] = (float)raw_z * GYRO_2000_SCALE;

    return 0;
}

int bmi088_read_temp(BMI088_t *imu) {
    uint8_t buffer[2];

    /* Temperature lives on the accelerometer die (TEMP_MSB:0x22, TEMP_LSB:0x23) */
    if (HAL_I2C_Mem_Read(imu->hi2c, BMI088_ACCEL_ADDR << 1, BMI088_ACC_TEMP_MSB, I2C_MEMADD_SIZE_8BIT, buffer, 2, 10) != HAL_OK) {
        return -1;
    }

    /* 11-bit signed value: MSB holds bits[10:3], LSB[7:5] hold bits[2:0] */
    uint16_t temp_uint11 = ((uint16_t)buffer[0] << 3) | (buffer[1] >> 5);
    int16_t temp_int11 = (temp_uint11 > 1023) ? (int16_t)temp_uint11 - 2048 : (int16_t)temp_uint11;

    /* Datasheet: Temperature = Temp_int11 * 0.125 degC + 23 degC */
    imu->temperature = (float)temp_int11 * 0.125f + 23.0f;

    return 0;
}

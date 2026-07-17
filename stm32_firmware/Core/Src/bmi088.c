#include "bmi088.h"
#include "pin_config.h"

#define GRAVITY_M_S2         9.80665f
#define PI                   3.14159265f

/* Conversion factors */
#define ACCEL_24G_SCALE      ((24.0f / 32768.0f) * GRAVITY_M_S2)
#define GYRO_2000_SCALE      ((2000.0f / 32768.0f) * (PI / 180.0f))

/* Delay between SCL edges during the 9-clock bus recovery sequence. */
#define BMI_BUS_RECOVERY_DELAY_US  10U

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

/* STM32F4 I2C bus recovery: releases a slave that holds SDA low.
 * On host-test builds (no GPIO model) this degrades to a DeInit/Init cycle. */
static void bmi_bus_recovery_delay_us(uint32_t us) {
    /* A tight spin using HAL_GetTick would be 1ms-granular and too coarse for
     * a 10us SCL edge. Use the Cortex-M cycle counter-free busy loop instead. */
    (void)us;
#ifdef STM32F4
    {
        /* ~168 cycles/us at 168 MHz; each loop iteration is ~3 cycles. */
        uint32_t cycles = us * 56U;
        volatile uint32_t i;
        for (i = 0U; i < cycles; ++i) {
            __NOP();
        }
    }
#endif
}

void bmi088_recover_bus(BMI088_t *imu) {
    I2C_HandleTypeDef *hi2c;
    if (imu == NULL || imu->hi2c == NULL) {
        return;
    }
    hi2c = imu->hi2c;

    (void)HAL_I2C_DeInit(hi2c);

#ifdef STM32F4
    /* Take SCL/SDA back as plain open-drain GPIO outputs to bit-bang clocks. */
    {
        GPIO_InitTypeDef gpio = {0};
        gpio.Pin = IMU_SCL_PIN | IMU_SDA_PIN;
        gpio.Mode = GPIO_MODE_OUTPUT_OD;
        gpio.Pull = GPIO_PULLUP;
        gpio.Speed = GPIO_SPEED_FREQ_LOW;
        HAL_GPIO_Init(IMU_SCL_PORT, &gpio);

        /* Make sure SDA is high so we can detect/release it, then clock SCL. */
        HAL_GPIO_WritePin(IMU_SCL_PORT, IMU_SDA_PIN, GPIO_PIN_SET);
        for (uint8_t i = 0U; i < 9U; ++i) {
            HAL_GPIO_WritePin(IMU_SCL_PORT, IMU_SCL_PIN, GPIO_PIN_SET);
            bmi_bus_recovery_delay_us(BMI_BUS_RECOVERY_DELAY_US);
            /* If SDA is still low after clocking, the slave has not released;
             * the remaining clocks give it more chances. */
            HAL_GPIO_WritePin(IMU_SCL_PORT, IMU_SCL_PIN, GPIO_PIN_RESET);
            bmi_bus_recovery_delay_us(BMI_BUS_RECOVERY_DELAY_US);
        }
        /* Generate a STOP: SCL high, then SDA low->high. */
        HAL_GPIO_WritePin(IMU_SCL_PORT, IMU_SCL_PIN, GPIO_PIN_SET);
        bmi_bus_recovery_delay_us(BMI_BUS_RECOVERY_DELAY_US);
        HAL_GPIO_WritePin(IMU_SCL_PORT, IMU_SDA_PIN, GPIO_PIN_RESET);
        bmi_bus_recovery_delay_us(BMI_BUS_RECOVERY_DELAY_US);
        HAL_GPIO_WritePin(IMU_SCL_PORT, IMU_SDA_PIN, GPIO_PIN_SET);
        bmi_bus_recovery_delay_us(BMI_BUS_RECOVERY_DELAY_US);

        /* Hand the pins back to the I2C alternate function. */
        gpio.Mode = GPIO_MODE_AF_OD;
        gpio.Pull = GPIO_PULLUP;
        gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        gpio.Alternate = IMU_I2C_AF;
        HAL_GPIO_Init(IMU_SCL_PORT, &gpio);
    }
#endif

    (void)HAL_I2C_Init(hi2c);
}

/* Called from the init state machine whenever an I2C transaction fails.
 * Returns 0 to let the caller restart the init sequence (retry), or -1 once
 * BMI088_MAX_INIT_ATTEMPTS is exhausted (give up -> FAILED). */
static int bmi_init_retry_or_fail(BMI088_t *imu, uint32_t now_ms) {
    if (imu->init_attempts < UINT8_MAX) {
        ++imu->init_attempts;
    }
    if (imu->init_attempts >= BMI088_MAX_INIT_ATTEMPTS) {
        imu->init_state = BMI_INIT_FAILED;
        return -1;
    }
    bmi088_recover_bus(imu);
    imu->init_state = BMI_INIT_ACC_RESET;
    imu->init_deadline_ms = now_ms + 50U;  /* let the bus/settling recover */
    return 0;
}

static int bmi_init_write(BMI088_t *imu, uint8_t addr, uint8_t reg, uint8_t value,
                          uint8_t next_state, uint32_t now_ms, uint32_t delay_ms) {
    if (bmi088_write_reg(imu->hi2c, addr, reg, value) != HAL_OK) {
        return bmi_init_retry_or_fail(imu, now_ms);
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
    imu->init_attempts = 0U;
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
                return bmi_init_retry_or_fail(imu, now_ms);
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
                return bmi_init_retry_or_fail(imu, now_ms);
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

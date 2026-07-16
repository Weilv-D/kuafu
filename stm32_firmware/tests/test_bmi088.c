#include "bmi088.h"
#include "test_support.h"

#include <string.h>

void run_bmi088_tests(void) {
    BMI088_t imu;
    I2C_HandleTypeDef i2c;
    int result = 0;

    memset(&imu, 0, sizeof(imu));
    memset(&i2c, 0, sizeof(i2c));
    test_i2c_reset();
    bmi088_begin_init(&imu, &i2c, 0U);
    TEST_EQ_INT(0, test_i2c_operation_count());

    result = bmi088_init_step(&imu, 0U);
    TEST_EQ_INT(0, result);
    TEST_EQ_INT(1, test_i2c_operation_count());
    (void)bmi088_init_step(&imu, 49U);
    TEST_EQ_INT(1, test_i2c_operation_count());

    for (uint32_t now = 50U; now <= 300U && result == 0; ++now) {
        test_set_time_ms(now);
        result = bmi088_init_step(&imu, now);
    }
    TEST_EQ_INT(1, result);
    TEST_TRUE(imu.initialized);
    TEST_EQ_INT(13, test_i2c_operation_count());

    memset(&imu, 0, sizeof(imu));
    test_i2c_reset();
    test_i2c_set_chip_ids(0x00U, 0x0FU);
    bmi088_begin_init(&imu, &i2c, 0U);
    result = 0;
    for (uint32_t now = 0U; now <= 120U && result == 0; ++now) {
        result = bmi088_init_step(&imu, now);
    }
    TEST_EQ_INT(-1, result);
    TEST_TRUE(!imu.initialized);
}

#include "bmi088.h"
#include "test_support.h"

#include <string.h>

/* Drive the init state machine forward, skipping its inter-step deadline
 * waits, until it returns non-zero (initialized=1 or FAILED=-1) or the step
 * cap is hit. Mirrors how main.c advances one step per loop tick. */
static int run_init_to_completion(BMI088_t *imu, uint32_t step_cap) {
    int result = 0;
    uint32_t now = 0U;
    while (result == 0 && now < step_cap) {
        result = bmi088_init_step(imu, now);
        now += 60U;  /* larger than the longest inter-step deadline (50ms) */
    }
    return result;
}

void run_bmi088_tests(void) {
    BMI088_t imu;
    I2C_HandleTypeDef i2c;
    int result = 0;

    memset(&imu, 0, sizeof(imu));
    memset(&i2c, 0, sizeof(i2c));
    test_i2c_reset();
    bmi088_begin_init(&imu, &i2c, 0U);
    TEST_EQ_INT(0, imu.init_attempts);
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
    TEST_EQ_INT(0, imu.init_attempts);  /* happy path never retries */

    /* A wrong accel chip ID retries BMI088_MAX_INIT_ATTEMPTS times then fails. */
    memset(&imu, 0, sizeof(imu));
    test_i2c_reset();
    test_i2c_set_chip_ids(0x00U, 0x0FU);
    bmi088_begin_init(&imu, &i2c, 0U);
    result = run_init_to_completion(&imu, 5000U);
    TEST_EQ_INT(-1, result);
    TEST_TRUE(!imu.initialized);
    TEST_EQ_INT(BMI088_MAX_INIT_ATTEMPTS, imu.init_attempts);

    /* A transient I2C failure mid-init is recovered by a retry: the first
     * write fails, the bus is "recovered" (stubbed), and the sequence restarts
     * and completes normally with init_attempts == 1. */
    memset(&imu, 0, sizeof(imu));
    test_i2c_reset();
    bmi088_begin_init(&imu, &i2c, 0U);
    test_i2c_fail_next(1);  /* exactly one I2C op fails, then succeeds */
    result = run_init_to_completion(&imu, 5000U);
    TEST_EQ_INT(1, result);
    TEST_TRUE(imu.initialized);
    TEST_EQ_INT(1, imu.init_attempts);

    /* When failures persist forever, init gives up after the cap. */
    memset(&imu, 0, sizeof(imu));
    test_i2c_reset();
    bmi088_begin_init(&imu, &i2c, 0U);
    test_i2c_fail_next(1000);  /* every I2C op fails */
    result = run_init_to_completion(&imu, 5000U);
    TEST_EQ_INT(-1, result);
    TEST_TRUE(!imu.initialized);
    TEST_EQ_INT(BMI088_MAX_INIT_ATTEMPTS, imu.init_attempts);
}

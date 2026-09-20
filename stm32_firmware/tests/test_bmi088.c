#include "bmi088.h"
#include "device_health.h"
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

    /* Channel health is independent; an unchanged static sample is not a
     * freshness fault, while an explicit read failure revokes only that channel. */
    memset(&imu, 0, sizeof(imu));
    test_i2c_reset();
    test_set_time_ms(100U);
    bmi088_begin_init(&imu, &i2c, 0U);
    TEST_EQ_INT(0, bmi088_read_pair(&imu, 100U));
    {
        BMI088SampleValidity_t validity;
        bmi088_get_sample_validity(&imu, 100U, 20U, &validity);
        TEST_TRUE(validity.accel_valid && validity.gyro_valid);
        TEST_TRUE(validity.accel_fresh && validity.gyro_fresh);
        TEST_EQ_INT(1, (int)validity.accel_sequence);
        TEST_EQ_INT(1, (int)validity.gyro_sequence);
        /* The aggregate health must have been refreshed by the shared-stamp
         * pair: it is the freshness source for the safety machine. */
        TEST_TRUE(device_health_is_fresh(&imu.health, 100U, 20U));
        bmi088_get_sample_validity(&imu, 121U, 20U, &validity);
        TEST_TRUE(!validity.accel_fresh && !validity.gyro_fresh);
    }
    test_i2c_fail_next(1U);
    TEST_EQ_INT(-1, bmi088_read_gyro(&imu, 100U));
    TEST_TRUE(!bmi088_gyro_healthy(&imu, 100U, 20U));
    TEST_TRUE(bmi088_accel_healthy(&imu, 100U, 20U));
    /* A failed gyro read must NOT refresh the aggregate pair health. */
    TEST_TRUE(!device_health_is_fresh(&imu.health, 121U, 20U));

    /* Shared-timestamp pair contract: when the accel and gyro transactions
     * would straddle a millisecond boundary under per-read HAL_GetTick()
     * stamping, the pair still validates because both channels carry the
     * caller's single instant.  The host HAL_GetTick is advanced between the
     * two channel reads to model that interleaving. */
    memset(&imu, 0, sizeof(imu));
    test_i2c_reset();
    test_set_time_ms(500U);
    bmi088_begin_init(&imu, &i2c, 500U);
    TEST_EQ_INT(0, bmi088_read_accel(&imu, 500U));
    test_set_time_ms(501U);
    TEST_EQ_INT(0, bmi088_read_gyro(&imu, 500U));
    TEST_TRUE(device_health_is_fresh(&imu.health, 500U, 20U));
    TEST_EQ_INT(1, (int)imu.accel_sequence);
    TEST_EQ_INT(1, (int)imu.gyro_sequence);
    /* Repeating the pair with a fresh instant keeps the aggregate fresh
     * even when the caller's clock has moved on between the reads. */
    TEST_EQ_INT(0, bmi088_read_pair(&imu, 502U));
    TEST_TRUE(device_health_is_fresh(&imu.health, 502U, 20U));
    TEST_EQ_INT(2, (int)imu.accel_sequence);
    TEST_EQ_INT(2, (int)imu.gyro_sequence);

    /* The final init write (gyro interrupt map) follows the same retry
     * contract as every other state: a retryable failure restarts the
     * sequence within the same round instead of aborting it.  Drive the
     * twelve preceding operations, then fail exactly the last one.  The
     * first init_step only arms the inter-step deadline, so the writes
     * land on every 60 ms step after the first. */
    memset(&imu, 0, sizeof(imu));
    test_i2c_reset();
    bmi088_begin_init(&imu, &i2c, 0U);
    {
        uint32_t now = 0U;
        for (int i = 0; i < 13; ++i) {
            result = bmi088_init_step(&imu, now);
            TEST_EQ_INT(0, result);
            now += 60U;
        }
        TEST_EQ_INT(12, (int)test_i2c_operation_count());
        test_i2c_fail_next(1U);
        result = bmi088_init_step(&imu, now);
        TEST_EQ_INT(0, result);  /* retryable: sequence restarted, not aborted */
        now += 60U;
        for (; now < 5000U && result == 0; now += 60U) {
            result = bmi088_init_step(&imu, now);
        }
        TEST_EQ_INT(1, result);
        TEST_TRUE(imu.initialized);
        TEST_EQ_INT(1, imu.init_attempts);
    }
}

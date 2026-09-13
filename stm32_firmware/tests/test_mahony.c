#include "mahony.h"
#include "test_support.h"

#include <math.h>
#include <string.h>

void run_mahony_tests(void) {
    MahonyFilter_t filter;
    MahonyUpdateStatus_t status;

    memset(&filter, 0, sizeof(filter));
    mahony_init(&filter, 2.0f, 0.1f);
    status = mahony_update_validated(&filter, 0.0f, 0.0f, 9.80665f,
                                     0.0f, 0.0f, 0.0f, 0.004f, 1U, 1U);
    TEST_EQ_INT(MAHONY_UPDATE_FULL, status);
    TEST_TRUE(filter.control_valid);
    TEST_NEAR(1.0f, sqrtf(filter.q0 * filter.q0 + filter.q1 * filter.q1 +
                          filter.q2 * filter.q2 + filter.q3 * filter.q3), 1.0e-5f);

    /* Invalid gyro is control-invalid and cannot keep integrating old data. */
    status = mahony_update_validated(&filter, 0.0f, 0.0f, 9.80665f,
                                     0.0f, 0.0f, 0.0f, 0.004f, 1U, 0U);
    TEST_EQ_INT(MAHONY_UPDATE_REJECTED, status);
    TEST_TRUE(!filter.control_valid);

    /* Accel loss permits only a bounded gyro-only interval. */
    mahony_reset(&filter);
    status = mahony_update_validated(&filter, 0.0f, 0.0f, 0.0f,
                                     0.1f, 0.0f, 0.0f, 0.004f, 0U, 1U);
    TEST_EQ_INT(MAHONY_UPDATE_GYRO_ONLY, status);
    TEST_TRUE(filter.control_valid);
    for (int i = 0; i < 30; ++i) {
        status = mahony_update_validated(&filter, 0.0f, 0.0f, 0.0f,
                                         0.1f, 0.0f, 0.0f, 0.004f, 0U, 1U);
    }
    TEST_EQ_INT(MAHONY_UPDATE_REJECTED, status);
    TEST_TRUE(!filter.control_valid);

    /* Non-finite inputs and an excessive innovation are rejected/degraded. */
    mahony_reset(&filter);
    status = mahony_update_validated(&filter, 0.0f, 0.0f, 9.80665f,
                                     NAN, 0.0f, 0.0f, 0.004f, 1U, 1U);
    TEST_EQ_INT(MAHONY_UPDATE_REJECTED, status);
    status = mahony_update_validated(&filter, 9.80665f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f, 0.004f, 1U, 1U);
    TEST_TRUE(status == MAHONY_UPDATE_GYRO_ONLY || status == MAHONY_UPDATE_FULL);
}

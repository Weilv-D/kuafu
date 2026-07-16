#ifndef KUAFU_TEST_SUPPORT_H
#define KUAFU_TEST_SUPPORT_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>

extern int g_test_failures;

void test_set_time_ms(uint32_t now_ms);
uint32_t test_get_time_ms(void);

#define TEST_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: TEST_TRUE failed: %s\n", __FILE__, __LINE__, #expr); \
        ++g_test_failures; \
    } \
} while (0)

#define TEST_EQ_INT(expected, actual) do { \
    int test_expected_ = (int)(expected); \
    int test_actual_ = (int)(actual); \
    if (test_expected_ != test_actual_) { \
        fprintf(stderr, "%s:%d: expected %d, got %d\n", \
                __FILE__, __LINE__, test_expected_, test_actual_); \
        ++g_test_failures; \
    } \
} while (0)

#define TEST_EQ_U8(expected, actual) \
    TEST_EQ_INT((uint8_t)(expected), (uint8_t)(actual))

#define TEST_NEAR(expected, actual, tolerance) do { \
    double test_expected_ = (double)(expected); \
    double test_actual_ = (double)(actual); \
    double test_tolerance_ = (double)(tolerance); \
    if (fabs(test_expected_ - test_actual_) > test_tolerance_) { \
        fprintf(stderr, "%s:%d: expected %.9g +/- %.9g, got %.9g\n", \
                __FILE__, __LINE__, test_expected_, test_tolerance_, test_actual_); \
        ++g_test_failures; \
    } \
} while (0)

#endif

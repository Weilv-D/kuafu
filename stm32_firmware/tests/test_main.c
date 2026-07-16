#include "crc8.h"
#include "kinematics.h"
#include "test_support.h"

void run_servo_mapping_tests(void);
void run_device_health_tests(void);
void run_safety_state_tests(void);
void run_startup_manager_tests(void);
void run_bmi088_tests(void);
void run_ddsm315_tests(void);

static void test_crc8_maxim_known_vector(void) {
    const uint8_t bytes[] = {0x01, 0x02, 0x03, 0x04};
    const uint8_t standard[] = "123456789";
    TEST_EQ_U8(0xF4, crc8_calculate(bytes, sizeof(bytes)));
    TEST_EQ_U8(0xA1, crc8_calculate(standard, sizeof(standard) - 1U));
}

static void test_fivebar_dwell_and_extension_signs(void) {
    float q_a = 1.0f;
    float q_b = 1.0f;

    TEST_EQ_INT(0, kinematics_solve_ik(0.058f, &q_a, &q_b));
    TEST_NEAR(0.0f, q_a, 1.0e-5f);
    TEST_NEAR(0.0f, q_b, 1.0e-5f);

    TEST_EQ_INT(0, kinematics_solve_ik(0.207f, &q_a, &q_b));
    TEST_TRUE(q_a < 0.0f);
    TEST_TRUE(q_b > 0.0f);
}

int main(void) {
    test_crc8_maxim_known_vector();
    test_fivebar_dwell_and_extension_signs();
    run_servo_mapping_tests();
    run_device_health_tests();
    run_safety_state_tests();
    run_startup_manager_tests();
    run_bmi088_tests();
    run_ddsm315_tests();

    if (g_test_failures != 0) {
        fprintf(stderr, "%d firmware host test(s) failed\n", g_test_failures);
        return 1;
    }
    printf("firmware host tests passed\n");
    return 0;
}

#include "servo_mapping.h"
#include "test_support.h"

void run_servo_mapping_tests(void) {
    TEST_EQ_INT(275, servo_angle_to_tick(0.0f, 0));
    TEST_EQ_INT(1097, servo_angle_to_tick(0.0f, 1));
    TEST_EQ_INT(2809, servo_angle_to_tick(0.0f, 2));
    TEST_EQ_INT(1023, servo_angle_to_tick(0.0f, 3));

    TEST_TRUE(servo_angle_to_tick(-0.1f, 0) < 275);
    TEST_TRUE(servo_angle_to_tick(-0.1f, 1) > 1097);
    TEST_TRUE(servo_angle_to_tick(+0.1f, 2) > 2809);
    TEST_TRUE(servo_angle_to_tick(+0.1f, 3) < 1023);

    TEST_NEAR(-0.1f,
              servo_tick_to_angle((uint16_t)servo_angle_to_tick(-0.1f, 0), 0),
              0.002f);
    TEST_NEAR(+0.1f,
              servo_tick_to_angle((uint16_t)servo_angle_to_tick(+0.1f, 3), 3),
              0.002f);
    TEST_TRUE(servo_tick_is_valid(0));
    TEST_TRUE(servo_tick_is_valid(4095));
    TEST_TRUE(!servo_tick_is_valid(-1));
    TEST_TRUE(!servo_tick_is_valid(4096));
    TEST_EQ_INT(+1, servo_direction(0));
    TEST_EQ_INT(-1, servo_direction(1));
    TEST_EQ_INT(+1, servo_direction(2));
    TEST_EQ_INT(-1, servo_direction(3));
}

#include "servo_mapping.h"
#include "pin_config.h"

static const int8_t k_servo_direction[SERVO_MAPPING_COUNT] = SERVO_DIR_INIT;
static const int16_t k_servo_center[SERVO_MAPPING_COUNT] = SERVO_CENTER_INIT;

int16_t servo_angle_to_tick(float angle_rad, uint8_t index) {
    int32_t delta;
    if (index >= SERVO_MAPPING_COUNT) {
        return 0;
    }
    delta = (int32_t)((float)k_servo_direction[index] * angle_rad * SERVO_TICKS_PER_RAD);
    return (int16_t)((int32_t)k_servo_center[index] + delta);
}

float servo_tick_to_angle(uint16_t raw_tick, uint8_t index) {
    if (index >= SERVO_MAPPING_COUNT) {
        return 0.0f;
    }
    return (float)k_servo_direction[index] *
           ((float)raw_tick - (float)k_servo_center[index]) /
           SERVO_TICKS_PER_RAD;
}

uint8_t servo_tick_is_valid(int32_t raw_tick) {
    return (uint8_t)(raw_tick >= 0 && raw_tick <= 4095);
}

int8_t servo_direction(uint8_t index) {
    return index < SERVO_MAPPING_COUNT ? k_servo_direction[index] : 0;
}

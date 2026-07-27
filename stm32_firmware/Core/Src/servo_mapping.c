#include "servo_mapping.h"
#include "pin_config.h"

static const int8_t k_servo_direction[SERVO_MAPPING_COUNT] = SERVO_DIR_INIT;
static const int16_t k_servo_center[SERVO_MAPPING_COUNT] = SERVO_CENTER_INIT;

int16_t servo_angle_to_tick(float angle_rad, uint8_t index) {
    int32_t delta;
    int32_t tick;
    if (index >= SERVO_MAPPING_COUNT) {
        return 0;
    }
    delta = (int32_t)((float)k_servo_direction[index] * angle_rad * SERVO_TICKS_PER_RAD);
    tick = (int32_t)k_servo_center[index] + delta;
    /* Defensive saturation: IK inputs are already workspace-clamped, so
     * reaching these bounds means a contract violation upstream.  Clamp here
     * (sync_write would clamp silently anyway) to keep one clear policy. */
    if (tick < 0) tick = 0;
    if (tick > 4095) tick = 4095;
    return (int16_t)tick;
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

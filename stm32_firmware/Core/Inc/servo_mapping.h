#ifndef SERVO_MAPPING_H
#define SERVO_MAPPING_H

#include <stdint.h>

#define SERVO_MAPPING_COUNT 4U

int16_t servo_angle_to_tick(float angle_rad, uint8_t index);
float servo_tick_to_angle(uint16_t raw_tick, uint8_t index);
uint8_t servo_tick_is_valid(int32_t raw_tick);
int8_t servo_direction(uint8_t index);

#endif

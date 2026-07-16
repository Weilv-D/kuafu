#include "test_support.h"
#include "stm32f4xx_hal.h"

int g_test_failures = 0;
static uint32_t g_fake_time_ms = 0;

void test_set_time_ms(uint32_t now_ms) {
    g_fake_time_ms = now_ms;
}

uint32_t test_get_time_ms(void) {
    return g_fake_time_ms;
}

uint32_t HAL_GetTick(void) {
    return g_fake_time_ms;
}

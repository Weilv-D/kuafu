#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../Comm/crc8.h"
#include "../Control/mahony.h"
#include "../Control/kinematics.h"
#include "../Control/lqr_controller.h"

void test_crc8(void) {
    printf("[Test] Testing CRC-8/MAXIM...\n");
    crc8_init();

    /* Test Case 1: DDSM315 ID Broadcast command: C8 64 00 00 00 00 00 00 00 -> CRC8 should be DE */
    uint8_t pkt1[9] = {0xC8, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t crc1 = crc8_calculate(pkt1, 9);
    printf("  Broadcast command calculated CRC: 0x%02X (Expected: 0xDE)\n", crc1);
    assert(crc1 == 0xDE);

    /* Test Case 2: DDSM315 Speed command for 30 RPM (300 raw speed): 01 64 01 2C 00 00 00 00 00 -> CRC8 should be A6 */
    uint8_t pkt2[9] = {0x01, 0x64, 0x01, 0x2C, 0x00, 0x00, 0x00, 0x00, 0x00};
    uint8_t crc2 = crc8_calculate(pkt2, 9);
    printf("  Speed control calculated CRC: 0x%02X (Expected: 0xA6)\n", crc2);
    assert(crc2 == 0xA6);

    printf("[Success] CRC-8/MAXIM tests passed!\n\n");
}

void test_mahony(void) {
    printf("[Test] Testing Mahony Filter...\n");
    MahonyFilter_t filter;
    mahony_init(&filter, 2.0f, 0.005f);

    /* Feed 100 iterations of static gravity (accelerometer pointing up) */
    /* ax=0, ay=0, az=9.81 m/s^2, gx=0, gy=0, gz=0 */
    for (int i = 0; i < 100; i++) {
        mahony_update(&filter, 0.0f, 0.0f, 9.81f, 0.0f, 0.0f, 0.0f, 0.001f);
    }
    printf("  Static State - Roll: %.4f rad, Pitch: %.4f rad, Yaw: %.4f rad\n",
           filter.roll, filter.pitch, filter.yaw);
    assert(fabs(filter.roll) < 1e-3);
    assert(fabs(filter.pitch) < 1e-3);

    /* Feed a pitch rate rotation of 0.1 rad/s for 100ms (100 steps) */
    for (int i = 0; i < 100; i++) {
        /* Gyro rates: gx=0, gy=0.1, gz=0 */
        mahony_update(&filter, 0.0f, 0.0f, 9.81f, 0.0f, 0.1f, 0.0f, 0.001f);
    }
    printf("  Pitch Rotate State - Roll: %.4f rad, Pitch: %.4f rad, Yaw: %.4f rad\n",
           filter.roll, filter.pitch, filter.yaw);
    assert(filter.pitch > 0.0f); /* Should have pitched up */

    printf("[Success] Mahony Filter tests passed!\n\n");
}

void test_kinematics(void) {
    printf("[Test] Testing 5-Bar Linkage Kinematics...\n");
    float alpha1 = 0.0f;
    float alpha2 = 0.0f;

    /* Test Case 1: Minimum leg height (dwell posture 58mm = 0.058m) */
    int res1 = kinematics_solve_ik(0.058f, &alpha1, &alpha2);
    printf("  Dwell Pose (58mm) -> Res: %d, Alpha Back (LB/RB): %.4f rad (%.2f deg), Alpha Front (LF/RF): %.4f rad (%.2f deg)\n",
           res1, alpha1, alpha1 * 180.0f / 3.14159f, alpha2, alpha2 * 180.0f / 3.14159f);
    assert(res1 == 0);

    /* Test Case 2: Intermediate leg height (150mm = 0.150m) */
    int res2 = kinematics_solve_ik(0.150f, &alpha1, &alpha2);
    printf("  Middle Pose (150mm) -> Res: %d, Alpha Back (LB/RB): %.4f rad (%.2f deg), Alpha Front (LF/RF): %.4f rad (%.2f deg)\n",
           res2, alpha1, alpha1 * 180.0f / 3.14159f, alpha2, alpha2 * 180.0f / 3.14159f);
    assert(res2 == 0);

    /* Test Case 3: Out of bound height (should be clamped safely) */
    int res3 = kinematics_solve_ik(0.300f, &alpha1, &alpha2);
    printf("  Out-of-Bound Pose (300mm) -> Res: %d (clamped internally)\n", res3);
    assert(res3 == 0);

    printf("[Success] Kinematics tests passed!\n\n");
}

void test_lqr(void) {
    printf("[Test] Testing LQR Controller...\n");
    LQRController_t lqr;
    lqr_init(&lqr);

    float tau_l = 0.0f;
    float tau_r = 0.0f;

    /* Test Case 1: Perfectly balanced state, zero commanded residual */
    lqr_update(&lqr, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.004f, &tau_l, &tau_r);
    printf("  Balanced state -> tau_l: %.4f N-m, tau_r: %.4f N-m\n", tau_l, tau_r);
    assert(fabs(tau_l) < 1e-4);
    assert(fabs(tau_r) < 1e-4);

    /* Test Case 2: Tilted pitch state (pitch = 0.1 rad, i.e. falling forward) */
    /* LQR gain K[1] is -61.18. So force F = -(-61.18 * 0.1) = 6.118 N.
     * Torque total = 6.118 * 0.03908 = 0.239 N-m.
     * Torque per wheel = 0.1195 N-m. */
    lqr_update(&lqr, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.004f, &tau_l, &tau_r);
    printf("  Tilted forward state -> tau_l: %.4f N-m, tau_r: %.4f N-m\n", tau_l, tau_r);
    assert(tau_l > 0.05f);
    assert(tau_r > 0.05f);

    printf("[Success] LQR Controller tests passed!\n\n");
}

int main(void) {
    printf("=== Starting STM32 Firmware Math Library Unit Tests ===\n\n");
    test_crc8();
    test_mahony();
    test_kinematics();
    test_lqr();
    printf("=== All Math Library Tests Completed Successfully! ===\n");
    return 0;
}

#include "lqr_controller.h"
#include "kuafu_generated.h"
#include "pin_config.h"
#include "ddsm315.h"
#include "test_support.h"

#include <math.h>

/* No demand, no motion -> both wheel commands are zero. */
static void test_lqr_zero_state_zero_output(void) {
    LQRController_t c;
    float tau_l = 1.0f, tau_r = 1.0f;
    lqr_init(&c);
    lqr_update(&c, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 0.0f, &tau_l, &tau_r);
    TEST_NEAR(0.0f, tau_l, 1.0e-6f);
    TEST_NEAR(0.0f, tau_r, 1.0e-6f);
}

/* Large pitch + position demand saturates at the stall-torque envelope. */
static void test_lqr_torque_envelope_saturation(void) {
    LQRController_t c;
    float tau_l = 0.0f, tau_r = 0.0f;
    lqr_init(&c);
    c.x_est = 0.5f;   /* persistent position error adds K0 demand */
    c.x_ref = 0.0f;
    for (int i = 0; i < 4; ++i) {
        lqr_update(&c, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    TEST_NEAR(-DDSM_MAX_TORQUE_NM, tau_l, 1.0e-4f);
    TEST_NEAR(-DDSM_MAX_TORQUE_NM, tau_r, 1.0e-4f);
}

/* Speed safety limit: beyond 0.5 m/s a strong braking force is added.  At
 * high enough speed the brake term must dominate the velocity-error term and
 * flip the torque sign relative to a sub-limit cruise speed. */
static void test_lqr_speed_limit_braking(void) {
    LQRController_t c;
    float tau_l = 0.0f, tau_r = 0.0f;
    float slow_l, fast_l;

    /* Sub-limit: vx = 10 rad/s * R ~= 0.39 m/s, no brake term. */
    lqr_init(&c);
    lqr_update(&c, 0.0f, 0.0f, 10.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 0.0f, &tau_l, &tau_r);
    slow_l = tau_l;

    /* Over-limit: vx = 30 rad/s * R ~= 1.17 m/s, brake term dominates. */
    lqr_init(&c);
    lqr_update(&c, 0.0f, 0.0f, 30.0f, 30.0f, 0.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 0.0f, &tau_l, &tau_r);
    fast_l = tau_l;

    TEST_TRUE(slow_l < 0.0f);
    TEST_TRUE(fast_l > 0.0f);   /* sign flip = brake dominates */
    TEST_TRUE(tau_l <= DDSM_MAX_TORQUE_NM + 1.0e-6f);
    TEST_TRUE(tau_r <= DDSM_MAX_TORQUE_NM + 1.0e-6f);
}

/* Straight-line (wz_cmd == 0): yaw_ref tracks the measured heading, so a
 * fixed "heading offset" produces NO differential torque.  Regression test
 * for the in-place spin bug: Mahony yaw drifts without a magnetometer, and a
 * fixed yaw_ref turned that drift into a constant YAW_KP * yaw_error torque
 * that counter-rotated the wheels. */
static void test_lqr_yaw_tracks_heading_when_not_turning(void) {
    LQRController_t c;
    float tau_l = 1.0f, tau_r = 1.0f;
    lqr_init(&c);
    for (int i = 0; i < 10; ++i) {
        lqr_update(&c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    TEST_NEAR(0.0f, tau_l, 1.0e-6f);
    TEST_NEAR(0.0f, tau_r, 1.0e-6f);
    TEST_NEAR(1.0f, c.yaw_ref, 1.0e-6f);
}

/* While a turn is commanded the yaw PD still acts, and its torque is clamped
 * so the differential component cannot dominate pitch. */
static void test_lqr_yaw_torque_clamp(void) {
    LQRController_t c;
    float tau_l = 0.0f, tau_r = 0.0f;
    lqr_init(&c);
    /* Command a sustained turn; yaw_ref integrates away from the (fixed)
     * measured heading until YAW_KP * yaw_error exceeds the clamp. */
    for (int i = 0; i < 500; ++i) {
        lqr_update(&c, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    TEST_NEAR(0.1f, tau_l, 1.0e-5f);
    TEST_NEAR(-0.1f, tau_r, 1.0e-5f);
}

/* Airborne (wheels unloaded, body level): x_ref must track x_est and the
 * integral must freeze, otherwise the free-spinning wheels integrate a huge
 * x_error that slams maximum torque on touchdown ("落地冲撞"). */
static void test_lqr_airborne_reanchor(void) {
    LQRController_t c;
    float tau_l = 0.0f, tau_r = 0.0f;
    lqr_init(&c);
    /* Wheels at 40 rad/s (~1.6 m/s) with a level body for 2 s: airborne. */
    for (int i = 0; i < 500; ++i) {
        lqr_update(&c, 0.0f, 0.0f, 40.0f, 40.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    TEST_NEAR(c.x_est, c.x_ref, 1.0e-4f);
    TEST_NEAR(0.0f, c.x_int, 1.0e-9f);
    /* Ground driving at cruise speed with a small tilt must NOT re-anchor:
     * the position loop stays engaged. */
    lqr_init(&c);
    for (int i = 0; i < 250; ++i) {
        lqr_update(&c, 0.05f, 0.0f, 5.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    TEST_TRUE((c.x_est - c.x_ref) > 0.01f);
    /* A brief high-speed blip (< 250 ms sustain) must NOT re-anchor either:
     * the stand-entry ground transient crosses |vx|>0.8 momentarily, and
     * re-anchoring there drags the anchor along with the runaway robot and
     * kills the position loop mid-rush ("疯狂后退"). */
    lqr_init(&c);
    for (int i = 0; i < 60; ++i) {  /* 60 * 4 ms = 240 ms < 250 ms */
        lqr_update(&c, 0.0f, 0.0f, 40.0f, 40.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    /* 40 rad/s * 0.039 m * 0.24 s ≈ 0.375 m integrated, anchor unmoved. */
    TEST_NEAR(0.0f, c.x_ref, 1.0e-6f);
    TEST_TRUE((c.x_est - c.x_ref) > 0.3f);
    /* Bursty fast spin (hand-held air jitter crosses the threshold
     * intermittently) must still re-anchor via the leaky bucket — otherwise
     * the phantom x_est slams the wheels on touchdown. */
    lqr_init(&c);
    for (int i = 0; i < 500; ++i) {
        float w = ((i % 60) < 40) ? 40.0f : 0.0f;  /* 2/3 duty cycle */
        lqr_update(&c, 0.0f, 0.0f, w, w, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    TEST_NEAR(c.x_est, c.x_ref, 1.0e-4f);
    TEST_NEAR(0.0f, c.x_int, 1.0e-9f);
}

/* Position integral is clamped to +/-0.05 m s. */
static void test_lqr_integral_clamp(void) {
    LQRController_t c;
    float tau_l = 0.0f, tau_r = 0.0f;
    lqr_init(&c);
    c.x_est = 10.0f;
    c.x_ref = 0.0f;
    for (int i = 0; i < 10; ++i) {
        lqr_update(&c, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, &tau_l, &tau_r);
    }
    TEST_TRUE(c.x_int <= 0.05f + 1.0e-6f);
    TEST_TRUE(c.x_int >= -0.05f - 1.0e-6f);
}

static void test_lqr_elapsed_dt_api(void) {
    LQRController_t c;
    float tau_l = 0.0f, tau_r = 0.0f;
    float used;
    lqr_init(&c);
    used = lqr_update_elapsed_dt(&c, 0.008f, 0.0f, 0.0f,
                                 1.0f, 1.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f,
                                 &tau_l, &tau_r);
    TEST_NEAR(0.008f, used, 1.0e-7f);
    TEST_NEAR(0.008f * WHEEL_RADIUS_M, c.x_est, 1.0e-7f);
    used = lqr_update_elapsed_dt(&c, 0.100f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 0.0f,
                                 &tau_l, &tau_r);
    TEST_NEAR(LQR_ELAPSED_DT_MAX, used, 1.0e-7f);
}

/* Body-frame equal torque becomes opposite raw signs for the mirrored right
 * motor, while decoding remains one explicit provisional conversion. */
static void test_wheel_dir_raw_contract(void) {
    uint8_t packet_l[DDSM_FRAME_SIZE];
    uint8_t packet_r[DDSM_FRAME_SIZE];
    int16_t raw_l, raw_r;
    const float body_tau = 0.2f;
    ddsm_build_torque(packet_l, DDSM_LEFT_ID, WHEEL_DIR_L * body_tau);
    ddsm_build_torque(packet_r, DDSM_RIGHT_ID, WHEEL_DIR_R * body_tau);
    raw_l = (int16_t)(((uint16_t)packet_l[2] << 8) | packet_l[3]);
    raw_r = (int16_t)(((uint16_t)packet_r[2] << 8) | packet_r[3]);
    TEST_TRUE(raw_l > 0);
    TEST_TRUE(raw_r < 0);
    TEST_NEAR(body_tau, (float)raw_l * DDSM_RAW_TO_TORQUE, 2.0e-4f);
    TEST_NEAR(-body_tau, (float)raw_r * DDSM_RAW_TO_TORQUE, 2.0e-4f);
    TEST_TRUE(fabsf((float)raw_l * DDSM_RAW_TO_TORQUE) < 0.5f);
}

/* Tipping past the fade-out window clears the position integral. */
static void test_lqr_integral_cleared_when_fallen(void) {
    LQRController_t c;
    float tau_l = 0.0f, tau_r = 0.0f;
    lqr_init(&c);
    c.x_int = 0.05f;
    lqr_update(&c, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
               0.0f, 0.0f, &tau_l, &tau_r);
    TEST_NEAR(0.0f, c.x_int, 1.0e-6f);
}

/* Non-finite inputs coast instead of propagating NaN through the gains into
 * the torque output (NaN compares false against every clamp bound, so the
 * magnitude-only dispatch clamps would not stop it). */
static void test_lqr_non_finite_input_coasts(void) {
    LQRController_t c;
    float tau_l = 1.0f, tau_r = 1.0f;
    lqr_init(&c);
    lqr_update_elapsed_dt(&c, 0.004f, (float)NAN, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          &tau_l, &tau_r);
    TEST_TRUE(isfinite(tau_l));
    TEST_NEAR(0.0f, tau_l, 1.0e-6f);
    TEST_TRUE(isfinite(tau_r));
    TEST_NEAR(0.0f, tau_r, 1.0e-6f);

    tau_l = 1.0f; tau_r = 1.0f;
    lqr_update_elapsed_dt(&c, (float)NAN, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          &tau_l, &tau_r);
    TEST_TRUE(isfinite(tau_l));
    TEST_NEAR(0.0f, tau_l, 1.0e-6f);
    TEST_TRUE(isfinite(tau_r));
    TEST_NEAR(0.0f, tau_r, 1.0e-6f);

    tau_l = 1.0f; tau_r = 1.0f;
    lqr_update_elapsed_dt(&c, 0.004f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, (float)INFINITY, 0.0f,
                          &tau_l, &tau_r);
    TEST_TRUE(isfinite(tau_l));
    TEST_NEAR(0.0f, tau_l, 1.0e-6f);
    TEST_TRUE(isfinite(tau_r));
    TEST_NEAR(0.0f, tau_r, 1.0e-6f);
}

void run_lqr_tests(void) {
    test_lqr_zero_state_zero_output();
    test_lqr_torque_envelope_saturation();
    test_lqr_speed_limit_braking();
    test_lqr_yaw_tracks_heading_when_not_turning();
    test_lqr_yaw_torque_clamp();
    test_lqr_airborne_reanchor();
    test_lqr_integral_clamp();
    test_lqr_elapsed_dt_api();
    test_wheel_dir_raw_contract();
    test_lqr_integral_cleared_when_fallen();
    test_lqr_non_finite_input_coasts();
}

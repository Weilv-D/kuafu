#include "kinematics.h"
#include "pin_config.h"
#include <math.h>

/* Helper function for clipping float values */
static float clamp_float(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static float wrapped_delta(float angle, float reference) {
    return atan2f(sinf(angle - reference), cosf(angle - reference));
}

/**
 * @brief Solves the absolute crank angle of one 5-bar branch.
 *
 * Hip pivot at (hip_x, 0), foot output Q at (0, -d0). Crank length a, rod b.
 * branch_sign selects the assembly mode (+1 / -1) matching the mechanism side.
 *
 * @return 0 on success, -1 if the target is outside the reachable workspace.
 */
static int solve_crank(float hip_x, float qx, float d0, float branch_sign, float *out_angle) {
    const float a = KIN_LEG_A; /* crank length (m) */
    const float b = KIN_LEG_B; /* rod length (m)   */

    float dx = qx - hip_x;
    float dy = -d0;
    float L = sqrtf(dx * dx + dy * dy);

    if (L > (a + b) || L < fabsf(a - b)) {
        return -1; /* Outside workspace */
    }

    float cos_ang = (a * a + L * L - b * b) / (2.0f * a * L);
    cos_ang = clamp_float(cos_ang, -1.0f, 1.0f);
    float ang = acosf(cos_ang);
    float base = atan2f(dy, dx);

    *out_angle = base + branch_sign * ang;
    return 0;
}

int kinematics_solve_ik_xy(float qx, float D0, float *q_hip_A, float *q_hip_B) {
    /* Safety clamping on D0 virtual leg length */
    float d0_clamped = clamp_float(D0, KIN_MIN_LEG_D0, KIN_MAX_LEG_D0);

    /* Hip pivots: A chain at -c, B chain at +c, per sim model
     * (servo_LF/RF at x=-0.026, servo_LB/RB at x=+0.026). */
    const float hip_x_A = -(KIN_LEG_C);
    const float hip_x_B = +(KIN_LEG_C);

    /* Dwell reference angles (D0 = KIN_MIN_LEG_D0) so the returned angles are
     * relative to the dwell posture, matching the sim/servo zero convention
     * (dwell = 0). NOTE: refine against measured mechanical zero on bench. */
    static int ref_valid = 0;
    static float ref_A = 0.0f;
    static float ref_B = 0.0f;
    if (!ref_valid) {
        if (solve_crank(hip_x_A, 0.0f, KIN_MIN_LEG_D0, -1.0f, &ref_A) != 0 ||
            solve_crank(hip_x_B, 0.0f, KIN_MIN_LEG_D0, +1.0f, &ref_B) != 0) {
            return -1;
        }
        ref_valid = 1;
    }

    float abs_A, abs_B;
    if (solve_crank(hip_x_A, qx, d0_clamped, -1.0f, &abs_A) != 0) return -1;
    if (solve_crank(hip_x_B, qx, d0_clamped, +1.0f, &abs_B) != 0) return -1;

    /* Dwell-relative angle, sign-matched to the sim/RL joint convention
     * (extension: hip_A negative, hip_B positive; see rl/env hip_*_target). */
    *q_hip_A = -wrapped_delta(abs_A, ref_A); /* A chain, 0 at dwell */
    *q_hip_B = -wrapped_delta(abs_B, ref_B); /* B chain, 0 at dwell */

    return 0;
}

int kinematics_solve_ik(float D0, float *q_hip_A, float *q_hip_B) {
    return kinematics_solve_ik_xy(0.0f, D0, q_hip_A, q_hip_B);
}

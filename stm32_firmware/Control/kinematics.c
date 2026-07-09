#include "kinematics.h"
#include "pin_config.h"
#include <math.h>

/* Helper function for clipping float values */
static float clamp_float(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

int kinematics_solve_ik(float D0, float *alpha1, float *alpha2) {
    /* Safety clamping on D0 virtual leg length */
    float d0_clamped = clamp_float(D0, KIN_MIN_LEG_D0, KIN_MAX_LEG_D0);

    float a = KIN_LEG_A; /* 0.093 m */
    float b = KIN_LEG_B; /* 0.110 m (wait! in pin_config.h we wrote 0.110m for rod, 0.093m? No, wait!) */
    /* Let's double check the dimensions:
       From kuafu_physics.py: 
       A_LEN = 93.0 mm (0.093m)
       B_LEN = 149.0 mm (0.149m)
       Let's use the exact values from pin_config.h: 
       KIN_LEG_A is defined as 0.093f or 0.050f? 
       Wait! In our pin_config.h we wrote:
       #define KIN_LEG_A               0.050f  (Wait, this was 50mm, let's correct it to 0.093f!)
       #define KIN_LEG_B               0.110f  (Wait, this was 110mm, let's correct it to 0.149f!)
       Let's correct pin_config.h first or use the correct constants here.
       Actually, AX, BX = -26, 26, so half width is 26mm (0.026m).
       Let's write kinematics.c with the correct physical parameters (a = 0.093, b = 0.149, half-width = 0.026)
       and let's also update pin_config.h to match these correct dimensions to avoid discrepancies!
    */
    
    a = 0.093f; /* Crank length = 93mm */
    b = 0.149f; /* Rod length = 149mm */
    float half_width = 0.026f; /* Half spacing = 26mm */

    /* --- Back Servo (Left branch, branch = 1) --- */
    /* Hip position: P0 = [-half_width, 0.0], Foot position: Q = [0.0, -d0_clamped] */
    float dx1 = 0.0f - (-half_width); /* 0.026 */
    float dy1 = -d0_clamped - 0.0f;
    float L1 = sqrtf(dx1 * dx1 + dy1 * dy1);

    if (L1 > (a + b) || L1 < fabsf(a - b)) {
        return -1; /* Outside workspace */
    }

    float c1 = (a * a + L1 * L1 - b * b) / (2.0f * a * L1);
    c1 = clamp_float(c1, -1.0f, 1.0f);
    float ang1 = acosf(c1);
    float base1 = atan2f(dy1, dx1);
    *alpha1 = base1 - ang1; /* Left leg back servo / branch = 1 */

    /* --- Front Servo (Right branch, branch = 0) --- */
    /* Hip position: P0 = [half_width, 0.0], Foot position: Q = [0.0, -d0_clamped] */
    float dx2 = 0.0f - half_width; /* -0.026 */
    float dy2 = -d0_clamped - 0.0f;
    float L2 = sqrtf(dx2 * dx2 + dy2 * dy2);

    if (L2 > (a + b) || L2 < fabsf(a - b)) {
        return -1; /* Outside workspace */
    }

    float c2 = (a * a + L2 * L2 - b * b) / (2.0f * a * L2);
    c2 = clamp_float(c2, -1.0f, 1.0f);
    float ang2 = acosf(c2);
    float base2 = atan2f(dy2, dx2);
    *alpha2 = base2 + ang2; /* Left leg front servo / branch = 0 */

    return 0;
}

#ifndef KINEMATICS_H
#define KINEMATICS_H

/**
 * @brief Solves the inverse kinematics for a 5-bar linkage leg.
 *        Maps target virtual leg length D0 (in meters) to target crank angles (in radians).
 * 
 * @param D0 Target virtual leg length (meters, range: 0.058m to 0.207m).
 * @param alpha1 Out: Target angle for the Back crank/servo (radians).
 * @param alpha2 Out: Target angle for the Front crank/servo (radians).
 * @return int 0 on success, -1 on failure (out of reachable workspace).
 */
int kinematics_solve_ik(float D0, float *alpha1, float *alpha2);

#endif /* KINEMATICS_H */

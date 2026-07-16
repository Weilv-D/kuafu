#ifndef KINEMATICS_H
#define KINEMATICS_H

/**
 * @brief Solves the inverse kinematics for a 5-bar linkage leg.
 *        Maps target virtual leg length D0 (in meters) to dwell-relative crank
 *        angles (radians), matching the sim/servo zero convention (dwell = 0).
 *
 * @param D0 Target virtual leg length (meters, range: 0.058m to 0.207m).
 * @param q_hip_A Out: A-chain (pivot x=-c) crank angle, dwell-relative (radians).
 * @param q_hip_B Out: B-chain (pivot x=+c) crank angle, dwell-relative (radians).
 * @return int 0 on success, -1 on failure (out of reachable workspace).
 */
int kinematics_solve_ik(float D0, float *q_hip_A, float *q_hip_B);

/* Full five-bar workspace target. Qx is fore/aft foot displacement in metres in
 * the chassis frame; D0 is positive downward in metres. */
int kinematics_solve_ik_xy(float qx, float d0, float *q_hip_A, float *q_hip_B);

#endif /* KINEMATICS_H */

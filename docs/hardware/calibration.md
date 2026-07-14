# Hardware Calibration

Hardware calibration is a release prerequisite, not a post-deployment refinement. Record each result with serial number, firmware schema, physical-model hash, date, operator, raw data path, and fitted uncertainty.

## Required Measurements

1. Confirm positive left/right wheel torque produces positive chassis X motion.
2. Confirm right torque greater than left produces positive yaw.
3. Determine BMI088 axis permutation and signs for projected gravity, pitch rate, roll rate, and yaw rate.
4. Measure loaded wheel radius and wheel current/torque/speed curve for each DDSM315.
5. Measure all four ST3215 dwell zero ticks, mirror signs, range, velocity, acceleration, current/torque relation, and thermal limit.
6. Sweep the five-bar workspace at low torque. Verify D0=58–207 mm, Qx range, closure gap, and `dqA/dD0<0`, `dqB/dD0>0` in the actual assembly branch.
7. Measure mass, COM, pitch inertia, wheel-ground friction, battery sag, and telemetry age/jitter.

## Applying Results

Physical geometry and controller parameters change only in `kuafu_physics.py`; regenerate the firmware artifact and retrain after changes. Servo zero and mirror settings are board/robot calibration values in firmware configuration. Domain-randomization ranges are centered on measured distributions, not guessed broad intervals.

## Bring-Up Sequence

1. Firmware protocol loopback with motors disabled.
2. IMU and encoder sign checks.
3. Servo zero and range check at reduced torque.
4. Tethered baseline hold at dwell.
5. Low-speed forward/back/yaw tracking.
6. Residual action enable with watchdog fault injection.
7. D0 sweep, pushes, slopes, then single steps.

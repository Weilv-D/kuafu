# System Architecture

## Control Partition

The STM32 owns safety-critical stabilization. It runs a 250 Hz discrete LQR/LQI controller with reference tracking, yaw heading/rate tracking, roll leveling, actuator limits, and five-bar workspace projection. The Pi5 runs an ONNX Actor at 50 Hz and supplies bounded residual actions. The Actor never replaces the baseline controller.

```text
high-level command
  -> jerk/acceleration-limited references
  -> STM32 baseline: position, velocity, pitch, heading, roll, IK
  <- Pi5 Actor: bounded common/yaw torque and Qx/D0 residuals
  -> wheel/servo commands
```

Roll leveling applies `ΔD0 = -ROLL_KP·roll - ROLL_KD·ωx` with `ROLL_KP = 190 mm/rad` and `ROLL_KD = 5.0`, derived from the 196 mm wheel track geometry. A high-speed D0 gate limits D0 to 120 mm whenever `|v| > 0.3 m/s` or `|w| > 0.6 rad/s`, preventing COM-rise topple. On stale action the STM32 clears only the learned residual. On stale heartbeat it also commands zero velocity and yaw rate, retaining baseline position/heading hold. A nonrecoverable tilt, thermal, or hardware fault is the only wheel-torque shutdown path.

## Learning Partition

The Actor receives four causal 35-dimensional proprioceptive frames, for a 140-dimensional input. Forward velocity and yaw rate come from wheel-odometry estimates, not simulation root truth. Every input has a sensor or estimator counterpart on real hardware. The `prev_applied_action` field carries the delayed action actually sent to actuators, so the policy observes the true effect of its previous output. A training Critic receives the Actor input plus 12 simulation-only values: nine static domain-randomization parameters and the three-dimensional applied push.

Friction domain randomization samples actual coefficients in `[0.3, 1.2]`. Push perturbation uses impulse-based sampling of 0-2 N·s with random timing. The first-frame history on reset is `[0, 0, 0, current]`, seeding the causal window with a valid current observation.

## Policy And Projection

The policy is a reparameterized tanh-squashed diagonal Gaussian:

```text
u ~ Normal(mu, sigma)
a = tanh(u)
log p(a) = log Normal(atanh(a); mu, sigma) - sum(log(1-a^2))
```

The Jacobian term `-log(1-a^2)` is included in `log_prob`, ensuring unbiased PPO ratios and entropy. PPO sampling, deterministic inference, ONNX, and Pi5 use this same transformation. PPO uses `init_noise_std = 0.4`, `gamma = 0.995`, and `num_steps_per_env = 96`. The six bounded actions are projected as common wheel torque, yaw wheel torque, left Qx, left D0, right Qx, and right D0. Qx and D0 are independently mapped through a two-dimensional five-bar IK grid.

## Curriculum

Training drives a 7-axis independent curriculum state machine: command, D0, domain randomization, latency/noise, slope, step, and push. Each axis has independent levels (0-4), evaluated every 25 PPO updates. An axis advances when survival rate is at least 90% and tracking passes at least 80% for two consecutive evaluations; it regresses on two consecutive failures. Upgrade decisions use full-episode velocity, yaw, and D0 tracking error with nonzero-command exposure, so a policy that remains still cannot pass the gate.

## Simulation

MJX advances ten 2 ms physics steps for each policy step. The five 4 ms base updates recompute LQR/LQI wheel torque while holding the policy residual. Native MuJoCo evaluation must use the same command projection and timing before it can be considered a release result.

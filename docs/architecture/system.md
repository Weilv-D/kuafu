# System Architecture

## Control Partition

The STM32 owns safety-critical stabilization. It runs a 250 Hz discrete LQR/LQI controller with reference tracking, heading tracking, actuator limits, and five-bar workspace projection. The Pi5 runs an ONNX Actor at 50 Hz and supplies bounded residual actions. The Actor never replaces the baseline controller.

```text
high-level command
  -> jerk/acceleration-limited references
  -> STM32 baseline: position, velocity, pitch, heading, roll, IK
  <- Pi5 Actor: bounded common/yaw torque and Qx/D0 residuals
  -> wheel/servo commands
```

On stale action the STM32 clears only the learned residual. On stale heartbeat it also commands zero velocity and yaw rate, retaining baseline position/heading hold. A nonrecoverable tilt, thermal, or hardware fault is the only wheel-torque shutdown path.

## Learning Partition

The Actor receives four causal 35-dimensional proprioceptive frames, for a 140-dimensional input. It is eligible for real hardware because every input has a sensor or estimator counterpart. A training Critic receives the Actor input plus 12 simulation-only values: nine static domain-randomization parameters and the three-dimensional applied push.

RMA latent adaptation and student distillation are not deployment dependencies. The former 157-dimensional RMA checkpoint format is marked `legacy-v0`.

## Policy And Projection

The policy is a tanh-squashed diagonal Gaussian:

```text
u ~ Normal(mu, sigma)
a = tanh(u)
log p(a) = log Normal(atanh(a); mu, sigma) - sum(log(1-a^2))
```

PPO sampling, deterministic inference, ONNX, and Pi5 use this same transformation. The six bounded actions are projected as common wheel torque, yaw wheel torque, left Qx, left D0, right Qx, and right D0. Qx and D0 are independently mapped through a two-dimensional five-bar IK grid.

## Simulation

MJX advances ten 2 ms physics steps for each policy step. The five 4 ms base updates recompute LQR/LQI wheel torque while holding the policy residual. Native MuJoCo evaluation must use the same command projection and timing before it can be considered a release result.

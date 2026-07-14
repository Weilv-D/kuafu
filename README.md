# KUAFU

KUAFU is a wheeled balancing robot with two closed-chain five-bar legs. This repository defines one control and deployment system spanning MuJoCo/MJX, PPO training, ONNX inference, a Pi5 runtime, STM32 firmware, and the UART contract between them.

The repository is gate-driven. A policy is eligible for deployment only after it passes the simulation, native-MuJoCo, ONNX, UART, HIL, and hardware gates in `docs/validation/acceptance.md`. The frozen S0-S7 scenario runner at `rl/verify/scenario_runner.py` produces the release-gate summary.

## Architecture

- `kuafu_physics.py`: canonical SI/mechanical parameters, discrete controller synthesis, five-bar geometry, generated-artifact hash.
- `rl/env/contract.py`: versioned frames, units, signs, observation/action interface, and protocol ranges. Current schema: `v1.1.0`.
- `rl/env/kuafu_mjx_env.py`: 500 Hz physics, 250 Hz baseline controller, 50 Hz residual policy.
- `rl/train/`: reparameterized tanh-squashed PPO, 7-axis independent curriculum, atomic schema-aware checkpoints with CPU-first loading.
- `rl/verify/scenario_runner.py`: frozen S0-S7 evaluation and release-gate summary.
- `rl/export/`: ONNX export with Torch/ONNX parity and manifest.
- `pi5_runtime/`: ONNX actor loop and versioned UART codec.
- `stm32_firmware/`: 250 Hz reference-tracking LQR/LQI, five-bar workspace projection, safety fallback, UART endpoint.

## Schema v1.1.0

The Actor observes a 140-dimensional input built from four causal frames of 35 proprioceptive values. Forward velocity and yaw rate come from wheel-odometry estimates, not simulation root truth. The `prev_applied_action` field is the delayed action actually sent to actuators, not the raw policy output. The first-frame history on reset is `[0, 0, 0, current]`. A high-speed D0 gate limits D0 to 120 mm when `|v| > 0.3 m/s` or `|w| > 0.6 rad/s`.

The policy is a tanh-squashed diagonal Gaussian with a reparameterized entropy and a numerically stable log-prob that accounts for the Jacobian term. PPO uses `init_noise_std = 0.4`, `gamma = 0.995`, and `num_steps_per_env = 96`. Curriculum operates over seven independent axes: command, D0, domain randomization, latency/noise, slope, step, and push. Friction domain randomization samples actual coefficients in `[0.3, 1.2]`. Push perturbation uses impulse-based sampling of 0-2 N·s with random timing. Roll leveling gains are `ROLL_KP = 190 mm/rad` and `ROLL_KD = 5.0`.

Checkpoints load CPU-first for cross-device resume compatibility. Schema `v1.0.0` checkpoints (157-dimensional RMA) are `legacy-v0` and cannot be resumed, exported, or deployed. Dependencies are version-pinned in `requirements.txt`.

## Verification

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/verify_controller_golden.py
rl/.venv/bin/python rl/verify/generate_artifacts.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated ROS plugins installed in the host Python environment.

## Quick Start

```bash
# Pilot run (100 updates, verify no NaN/stability issues)
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-pilot-s42 --num_envs 3072 --max_updates 100 --seed 42

# Full training
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-s42 --num_envs 3072 --seed 42

# Resume (must use a new run name)
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-s42-r01 \
  --resume rl/checkpoints/ppo-v1-s42/teacher/model_1000.pt --num_envs 3072 --seed 42

# Evaluate against frozen S0-S7 scenarios
rl/.venv/bin/python rl/verify/scenario_runner.py \
  --ckpt rl/checkpoints/ppo-v1-s42/teacher/model_5000.pt --out-dir eval_results

# Export deployable policy (requires passing release gate)
rl/.venv/bin/python rl/export/export_policy.py --ckpt <model.pt> --out <policy.onnx>
```

## Documentation

- `docs/architecture/system.md`: control, simulation, learning, and runtime architecture.
- `docs/contracts/interface.md`: normative coordinate, unit, observation, action, and UART contract.
- `docs/operations/training.md`: training, resume, export, and release workflow.
- `docs/operations/deployment.md`: ONNX and Pi5/STM32 deployment procedure.
- `docs/validation/acceptance.md`: automated and hardware acceptance matrix.
- `docs/hardware/calibration.md`: required physical measurements and calibration records.
- `docs/KUAFU.md`: canonical physical-model reference.

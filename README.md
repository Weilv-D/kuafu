# KUAFU

KUAFU is a wheeled balancing robot with two closed-chain five-bar legs. This repository defines one control and deployment system spanning MuJoCo/MJX, PPO training, ONNX inference, a Pi5 runtime, STM32 firmware, and the UART contract between them.

The repository is intentionally gate-driven. A policy is eligible for deployment only after it passes the simulation, native-MuJoCo, ONNX, UART, HIL, and hardware gates in `docs/validation/acceptance.md`.

## Architecture

- `kuafu_physics.py`: canonical SI/mechanical parameters, discrete controller synthesis, five-bar geometry, generated-artifact hash.
- `rl/env/contract.py`: versioned frames, units, signs, observation/action interface, and protocol ranges.
- `rl/env/kuafu_mjx_env.py`: 500 Hz physics, 250 Hz baseline controller, 50 Hz residual policy.
- `rl/train/`: tanh-squashed PPO and atomic schema-aware checkpoints.
- `rl/export/`: ONNX export with Torch/ONNX parity and manifest.
- `pi5_runtime/`: ONNX actor loop and versioned UART codec.
- `stm32_firmware/`: 250 Hz reference-tracking LQR/LQI, five-bar workspace projection, safety fallback, UART endpoint.

## Current Status

Software contracts, JAX execution, source-generated control constants, policy export interfaces, and host-side protocol tests are implemented. Current checkpoints from the former 157-dimensional RMA interface are `legacy-v0`; they must not be resumed, evaluated as release candidates, exported, or deployed.

The repository does not claim hardware acceptance without physical measurements and HIL results. Required bench calibration and P9 gates are listed in `docs/hardware/calibration.md` and `docs/validation/acceptance.md`.

## Verification

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/verify_controller_golden.py
rl/.venv/bin/python rl/verify/calibrate_fivebar.py
rl/.venv/bin/python rl/verify/generate_artifacts.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated ROS plugins installed in the host Python environment.

## Documentation

- `docs/architecture/system.md`: control, simulation, learning, and runtime architecture.
- `docs/contracts/interface.md`: normative coordinate, unit, observation, action, and UART contract.
- `docs/operations/training.md`: training, resume, export, and release workflow.
- `docs/operations/deployment.md`: ONNX and Pi5/STM32 deployment procedure.
- `docs/validation/acceptance.md`: automated and hardware acceptance matrix.
- `docs/hardware/calibration.md`: required physical measurements and calibration records.
- `docs/KUAFU.md`: canonical physical-model reference.

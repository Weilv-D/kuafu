# Training Operations

## Preconditions

Run source and geometry verification before any PPO job:

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/verify_controller_golden.py
rl/.venv/bin/python rl/verify/calibrate_fivebar.py
rl/.venv/bin/python rl/verify/generate_artifacts.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q
```

## PPO

```bash
rl/.venv/bin/python rl/train/train.py --run_name stage-s0 --num_envs 3072
```

Training uses a tanh-squashed policy and fixed physical-scale Actor inputs. `legacy-v0` checkpoints have incompatible 157-dimensional RMA inputs and are not resumable.

Resume into a new run name so the source checkpoint remains immutable:

```bash
rl/.venv/bin/python rl/train/train.py --run_name stage-s1 --num_envs 3072 \
  --resume rl/checkpoints/stage-s0/teacher/model_1000.pt
```

Checkpoints atomically store model, optimizer, iteration, adaptive learning rate, entropy coefficient, curriculum buffers, full vectorized MJX state/delay buffers/integrators, Python/NumPy/Torch/JAX RNG state, schema version, and physical-model hash. Smoke runs always use an isolated `_smoke/` directory.

## Curriculum

Each environment independently samples command, DR, terrain, push, and D0 difficulty axes. Upgrade decisions use full-episode velocity, yaw, and D0 MAE with nonzero-command exposure, not survival alone. A policy that remains still cannot pass the upgrade gate.

Stage gates and frozen holdout evaluation are defined in `docs/validation/acceptance.md`. Training does not establish a release: native simulation, ONNX, HIL, and hardware gates remain mandatory.

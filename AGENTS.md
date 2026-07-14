# KUAFU Development Guide

## Sources Of Truth

- `kuafu_physics.py`: physical parameters, timing, LQR/LQI synthesis, five-bar geometry, model hash, generated constants.
- `rl/env/contract.py`: schema, frame, units, signs, observation/action contract, protocol ranges.
- `docs/contracts/interface.md`: readable contract; it must agree with the executable contract.

Physical changes are made in `kuafu_physics.py`, followed by `rl/verify/generate_artifacts.py`. Do not hand-edit generated firmware constants.

## Environment

Use `rl/.venv/bin/python`. The Actor has 140 inputs (35 values × four causal frames); the Critic has 152 inputs. The policy is tanh-squashed in PPO, inference, ONNX, and Pi5. Old 157-dimensional RMA checkpoints are `legacy-v0`.

## Required Checks

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/verify_controller_golden.py
rl/.venv/bin/python rl/verify/calibrate_fivebar.py
rl/.venv/bin/python rl/verify/generate_artifacts.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q
```

For controller or environment changes, also run a JAX reset/step/auto-reset smoke. For firmware changes, run the host C syntax check where available and then a target build/HIL test.

## Training And Export

```bash
rl/.venv/bin/python rl/train/train.py --run_name s0 --num_envs 3072
rl/.venv/bin/python rl/train/train.py --run_name s1 --num_envs 3072 \
  --resume rl/checkpoints/s0/teacher/model_1000.pt
rl/.venv/bin/python rl/export/export_policy.py --ckpt <model.pt> --out <policy.onnx>
```

Resume to a new run name. Checkpoints are immutable sources and carry optimizer, learning rate, entropy, curriculum, RNG, schema, and model hash. Follow the release gates in `docs/validation/acceptance.md`; simulation completion is not hardware acceptance.

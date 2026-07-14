# KUAFU Development Guide

## Sources Of Truth

- `kuafu_physics.py`: physical parameters, timing, LQR/LQI synthesis, five-bar geometry, roll/yaw gains, model hash, generated constants.
- `rl/env/contract.py`: schema version, frame, units, signs, observation/action contract, protocol ranges. Current schema: `v1.1.0`.
- `rl/train/train_config.py`: PPO hyperparameters, network architecture, training scale.
- `rl/train/curriculum.py`: 7-axis independent curriculum state machine.
- `docs/contracts/interface.md`: readable contract; it must agree with the executable contract.

Physical changes are made in `kuafu_physics.py`, followed by `rl/verify/calibrate_fivebar.py` and `rl/verify/generate_artifacts.py`. Do not hand-edit generated firmware constants.

## Environment

Use `rl/.venv/bin/python`. The Actor has 140 inputs (35 values × four causal frames); the Critic has 152 inputs. The policy is tanh-squashed in PPO, inference, ONNX, and Pi5. Actor observations use wheel-odometry velocity estimates, not simulation root truth. The `prev_applied_action` field is the delayed action actually sent to actuators. Schema `v1.0.0` checkpoints (157-dimensional RMA) are `legacy-v0` and cannot be resumed, exported, or deployed.

## Required Checks

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/verify_controller_golden.py
rl/.venv/bin/python rl/verify/generate_artifacts.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q
```

For controller or environment changes, also run a JAX reset/step/auto-reset smoke. For firmware changes, run the host C syntax check where available and then a target build/HIL test.

## Training And Export

```bash
# Pilot run (100 updates, verify no NaN/stability issues)
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-pilot-s42 --num_envs 3072 --max_updates 100 --seed 42

# Full training
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-s42 --num_envs 3072 --seed 42

# Resume (must use a new run name)
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-s42-r01 \
  --resume rl/checkpoints/ppo-v1-s42/teacher/model_1000.pt --num_envs 3072 --seed 42

# Evaluate against frozen scenarios
rl/.venv/bin/python rl/verify/scenario_runner.py \
  --ckpt rl/checkpoints/ppo-v1-s42/teacher/model_5000.pt --out-dir eval_results

# Export deployable policy (requires passing release gate)
rl/.venv/bin/python rl/export/export_policy.py --ckpt <model.pt> --out <policy.onnx>
```

Resume to a new run name. Checkpoints carry model, optimizer, learning rate, entropy, curriculum, full vectorized environment state, all RNG states, schema, and model hash. CPU-first checkpoint loading ensures cross-device resume compatibility. Follow the release gates in `docs/validation/acceptance.md`; simulation completion is not hardware acceptance.

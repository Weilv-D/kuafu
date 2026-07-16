# KUAFU Development Guide

## Sources Of Truth

- `kuafu_physics.py`: physical parameters, timing, LQR/LQI synthesis, five-bar geometry, roll/yaw gains, model hash, generated constants.
- `rl/env/contract.py`: schema version, frame, units, signs, observation/action contract, protocol ranges. Current schema: `v1.1.0`.
- `rl/train/curriculum.py`: 8-axis independent curriculum (`AXES`, `DIFF_INDICES`, `AXIS_CONFIG`) — the single source of truth for difficulty axis order, per-axis gates, and level state. `kuafu_mjx_env.py` and `train.py` import from it.
- `rl/train/train_config.py`: PPO hyperparameters, network architecture, training scale.
- `docs/contracts/interface.md`: readable contract; it must agree with the executable contract.

Physical changes are made in `kuafu_physics.py`, followed by `rl/verify/calibrate_fivebar.py` and `rl/verify/generate_artifacts.py`. Do not hand-edit generated firmware constants.

## Environment

Use `rl/.venv/bin/python`. The Actor has 140 inputs (35 values × four causal frames); the Critic has 152 inputs. The policy is tanh-squashed in PPO, inference, ONNX, and Pi5. Actor observations use wheel-odometry velocity estimates, not simulation root truth. The `prev_applied_action` field is the delayed action actually sent to actuators.

Training difficulty is an 8-axis independent curriculum (`rl/train/curriculum.py`): `command, d0, dr, latency, slope, step, rough, push`. Each axis advances/falls back on per-axis done-env episode statistics (`AXIS_CONFIG`). Terrain/perturbation axes use a pure survival gate; `command`/`d0` add a tracking anti-cheat gate calibrated to the physical noise floor. Difficulty is sampled per env in a band `[0.8*level, level]` per axis. Rough terrain is a MuJoCo heightfield whose data is rewritten each reset (MJX supports `hfield×capsule`; wheels are capsules).

Checkpoint compatibility: schema `v1.0.0` (157-dim RMA) and the pre-8-axis `model_1650.pt` (5-axis curriculum, `kuafu_curriculum_schema != "v2"`) are `legacy` and cannot be resumed. `--resume_ignore_hash` permits resuming a checkpoint whose `model_hash` differs after a benign asset change (e.g. terrain XML), but never bypasses the obs/action schema check.

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
rl/.venv/bin/python rl/train/train.py --run_name ppo-v2-pilot-s42 --num_envs 3072 --iterations 100 --seed 42

# Full training
rl/.venv/bin/python rl/train/train.py --run_name ppo-v2-s42 --num_envs 3072 --seed 42

# Resume (must use a new run name)
rl/.venv/bin/python rl/train/train.py --run_name ppo-v2-s42-r01 \
  --resume rl/checkpoints/ppo-v2-s42/teacher/model_1000.pt --num_envs 3072 --seed 42

# Evaluate against frozen scenarios
rl/.venv/bin/python rl/verify/scenario_runner.py \
  --ckpt rl/checkpoints/ppo-v1-s42/teacher/model_5000.pt --out-dir eval_results

# Export deployable policy (requires passing release gate)
rl/.venv/bin/python rl/export/export_policy.py --ckpt <model.pt> --out <policy.onnx>
```

Resume to a new run name. Checkpoints carry model, optimizer, learning rate, entropy, curriculum, full vectorized environment state, all RNG states, schema, and model hash. CPU-first checkpoint loading ensures cross-device resume compatibility. Follow the release gates in `docs/validation/acceptance.md`; simulation completion is not hardware acceptance.

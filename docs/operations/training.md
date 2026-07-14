# Training Operations

## Preconditions

Run source and geometry verification before any PPO job:

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/verify_controller_golden.py
rl/.venv/bin/python rl/verify/generate_artifacts.py --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q
```

Dependencies are version-pinned in `rl/requirements.txt`.

## PPO

```bash
# Pilot run (100 updates, verify no NaN/stability issues)
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-pilot-s42 --num_envs 3072 --max_updates 100 --seed 42

# Full training
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-s42 --num_envs 3072 --seed 42
```

The policy is a reparameterized tanh-squashed diagonal Gaussian whose `log_prob` includes the `-log(1-a^2)` Jacobian term, yielding unbiased PPO ratios and entropy. Hyperparameters: `init_noise_std = 0.4`, `gamma = 0.995`, `lam = 0.95`, `num_steps_per_env = 96`, `clip = 0.2`, `lr = 3e-4` adaptive on KL (`desired_kl = 0.01`), actor/critic MLP `[512,512,512]` ELU. Actor inputs use fixed physical-scale values from the contract; no empirical normalization. `legacy-v0` checkpoints have incompatible 157-dimensional RMA inputs and are not resumable.

## Resume

Resume into a new run name so the source checkpoint remains immutable:

```bash
rl/.venv/bin/python rl/train/train.py --run_name ppo-v1-s42-r01 --num_envs 3072 \
  --resume rl/checkpoints/ppo-v1-s42/teacher/model_1000.pt --seed 42
```

Checkpoints load CPU-first (`map_location="cpu"`) for cross-device resume compatibility. They atomically store model, optimizer, iteration, adaptive learning rate, entropy coefficient, 7-axis curriculum buffers, full vectorized MJX state/delay buffers/integrators, Python/NumPy/Torch/JAX RNG state, schema version, and physical-model hash. Smoke runs always use an isolated `_smoke/` directory.

## Curriculum

A 7-axis independent curriculum state machine drives difficulty: command, D0, domain randomization, latency/noise, slope, step, and push. Each axis has five levels (0-4), evaluated every 25 PPO updates. An axis advances after two consecutive evaluations with survival rate at least 90% and tracking pass at least 80%; it regresses after two consecutive failures. Friction domain randomization samples actual coefficients in `[0.3, 1.2]`. Push perturbation uses impulse-based sampling of 0-2 N·s with random timing. Upgrade decisions use full-episode velocity, yaw, and D0 MAE with nonzero-command exposure, not survival alone.

## Evaluation And Release

Evaluate a checkpoint against frozen S0-S7 scenarios:

```bash
rl/.venv/bin/python rl/verify/scenario_runner.py \
  --ckpt rl/checkpoints/ppo-v1-s42/teacher/model_5000.pt --out-dir eval_results
```

The runner writes per-episode JSONL and a `release_gate`-compatible `summary.json`. Stage gates and frozen holdout evaluation are defined in `docs/validation/acceptance.md`. Training does not establish a release: native simulation, ONNX, HIL, and hardware gates remain mandatory.

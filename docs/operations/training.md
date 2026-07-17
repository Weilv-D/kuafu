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

Checkpoints load CPU-first (`map_location="cpu"`) for cross-device resume compatibility. They atomically store model, optimizer, iteration, adaptive learning rate, entropy coefficient, the 8-axis curriculum state (`state_dict` per axis: level/streak/fail_streak) plus per-axis episode buffers, full vectorized MJX state/delay buffers/integrators, Python/NumPy/Torch/JAX RNG state, schema version, and physical-model hash. Legacy 5-axis checkpoints (`kuafu_curriculum_schema != "v2"`) cannot be resumed; `--resume_ignore_hash` permits resuming a checkpoint whose `model_hash` differs after a benign asset change (e.g. terrain XML) but never bypasses the obs/action schema check. Smoke runs always use an isolated `_smoke/` directory.

## Curriculum

An 8-axis independent curriculum state machine (`rl/train/curriculum.py`) drives difficulty: `command, d0, dr, latency, slope, step, rough, push`. Each axis has five levels (0-4) and is evaluated independently by done-env episode count (triggered when an axis accumulates at least `min_episodes` done episodes near its current level). Terrain/perturbation axes (`dr, latency, slope, step, rough, push`) advance on a pure survival gate (best practice: never block terrain progress on velocity tracking that is physically impossible on rough ground). The `command` axis is **pinned at max level** (fixed full-range velocity commands, legged_gym / MuJoCo Playground / RMA standard) and never re-evaluated — velocity tracking is learned from step one under the LQR baseline. Only `d0` keeps a tracking anti-cheat gate (calibrated to the physical noise floor) to reject "do-nothing" policies. Difficulty is sampled per env in a band `[0.8*level, level]` per axis. Rough terrain is a MuJoCo heightfield whose data is rewritten each reset (MJX supports `hfield×capsule`; wheels are capsules). Non-pinned axes advance after two consecutive passing evaluations and regress after two consecutive failures.

## Evaluation And Release

Evaluate a checkpoint against frozen S0-S7 scenarios:

```bash
rl/.venv/bin/python rl/verify/scenario_runner.py \
  --ckpt rl/checkpoints/ppo-v1-s42/teacher/model_5000.pt --out-dir eval_results
```

The runner writes per-episode JSONL and a `release_gate`-compatible `summary.json`. Stage gates and frozen holdout evaluation are defined in `docs/validation/acceptance.md`. Training does not establish a release: native simulation, ONNX, HIL, and hardware gates remain mandatory.

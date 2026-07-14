# KUAFU Reinforcement Learning

The RL stack trains a residual policy around the 250 Hz baseline controller. It is not a standalone balance controller.

## Interface

- Actor: 35 hardware-observable values over four causal frames, input dimension 140.
- Critic: Actor input plus 12 simulation-only values, input dimension 152.
- Action: `[dtau_common, dtau_yaw, dQx_L, dD0_L, dQx_R, dD0_R]`, each bounded by tanh.
- Physics/base/policy rates: 500/250/50 Hz.

The authoritative specification is `../docs/contracts/interface.md`. 157-dimensional RMA checkpoints are legacy and cannot be resumed or exported by this stack.

## Commands

```bash
rl/.venv/bin/python rl/verify/verify_physics_source.py
rl/.venv/bin/python rl/verify/calibrate_fivebar.py
rl/.venv/bin/python rl/verify/generate_artifacts.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 rl/.venv/bin/python -m pytest rl/train/tests -q

rl/.venv/bin/python rl/train/train.py --run_name s0 --num_envs 3072
rl/.venv/bin/python rl/train/train.py --run_name s1 --num_envs 3072 \
  --resume rl/checkpoints/s0/teacher/model_1000.pt

rl/.venv/bin/python rl/export/export_policy.py \
  --ckpt rl/checkpoints/s7/teacher/model_5000.pt --out artifacts/kuafu-policy.onnx
```

Resume destinations must use a new run name. Checkpoints are atomically written and include optimizer, dynamic learning state, curriculum, RNG, schema, and physical-model hash.

## Verification Scope

`verify_physics_source.py` covers sign, controller-pole, geometry, and generation guards. `calibrate_fivebar.py` produces a validated dwell-relative IK artifact. `train/tests` covers DLPack, the tanh distribution, and the protocol codec. Native policy and hardware tests are release gates, not optional visual checks.

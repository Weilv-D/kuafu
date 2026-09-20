# KUAFU Documentation

This directory distinguishes normative interfaces from historical design records.

| Document | Role |
|---|---|
| `HANDOFF.md` | onboarding handoff: two-board architecture, safety/startup state machines, build/flash/SWD procedures, review history, current known limits |
| `architecture/system.md` | normative runtime and learning partition |
| `contracts/interface.md` | readable form of the executable cross-layer contract |
| `KUAFU.md` | physical-model reference for `kuafu_physics.py` |
| `operations/training.md` | reproducible training, resume, and export workflow |
| `operations/deployment.md` | ONNX, Pi5, STM32, teleop, and HIL procedure |
| `validation/acceptance.md` | release and hardware gates |
| `validation/stm32-firmware-*.md` | dated firmware review records: defect lists and verification evidence per review round |
| `hardware/calibration.md` | required physical calibration records |
| `plans/` | dated historical proposals; they record design rationale at a point in time and never override the contract or source of truth |

Normative content lives in `rl/env/contract.py`, `kuafu_physics.py`, and the source files they govern. When a document disagrees with the source, the source is correct.

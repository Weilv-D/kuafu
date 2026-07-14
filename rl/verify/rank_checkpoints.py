# -*- coding: utf-8 -*-
"""Rank KUAFU checkpoints by frozen evaluation performance.

Expands a glob pattern, loads each checkpoint, runs a minimal S0 evaluation
(deterministic hold + domain-randomised episodes), and ranks the survivors by
survival rate (descending) then tilt RMS (ascending).  Non-loadable or
schema-mismatched checkpoints are reported but excluded from the ranking.

Usage::

    rl/.venv/bin/python rl/verify/rank_checkpoints.py \
        --pattern 'rl/checkpoints/*/teacher/model_*.pt' \
        --out ranking.json --episodes 10
"""
import os
import sys
import json
import glob
import argparse

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import numpy as np
import mujoco

import kuafu_physics as P


def evaluate_checkpoint(ckpt, model, data, episodes=10, seed=42):
    """Run a minimal S0 evaluation and return summary metrics.

    Half the episodes are deterministic; the other half use domain
    randomisation so the ranking reflects mild robustness, not just the
    nominal case.
    """
    from rl.train.teacher_model import TeacherInferenceModel
    from rl.verify.eval_policy import run_episode

    teacher = TeacherInferenceModel.from_checkpoint(ckpt)
    cmd = np.array([0.0, 0.0, P.D0_MIN])
    steps = int(round(20.0 / P.RL_DT))
    rng = np.random.default_rng(seed)
    n_det = max(1, episodes // 2)
    n_dr = max(1, episodes - n_det)

    results = []
    for _ in range(n_det):
        results.append(run_episode(model, data, teacher, cmd, steps))
    for _ in range(n_dr):
        results.append(run_episode(model, data, teacher, cmd, steps, rng=rng, dr=True))

    survival = sum(1 for r in results if not r["fallen"]) / len(results)
    tilt = float(np.mean([r["pitch_rms_deg"] for r in results]))
    return {
        "survival_rate": float(survival),
        "tilt_rms_deg": tilt,
        "episodes": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="Rank KUAFU checkpoints by S0 performance")
    parser.add_argument("--pattern", required=True, help="glob pattern for checkpoints")
    parser.add_argument("--out", default="ranking.json", help="output ranking JSON path")
    parser.add_argument("--episodes", type=int, default=10,
                        help="evaluation episodes per checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for DR episodes")
    args = parser.parse_args()

    ckpts = sorted(glob.glob(args.pattern))
    if not ckpts:
        print(f"No checkpoints matching {args.pattern}")
        return 1

    xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "kuafu.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    results = []
    for ckpt in ckpts:
        print(f"Evaluating {ckpt} ...")
        try:
            metrics = evaluate_checkpoint(ckpt, model, data,
                                          episodes=args.episodes, seed=args.seed)
            entry = {"checkpoint": ckpt, "loadable": True, **metrics}
            print(f"  survival={metrics['survival_rate']:.2f}  "
                  f"tilt={metrics['tilt_rms_deg']:.3f} deg")
        except Exception as exc:
            entry = {"checkpoint": ckpt, "loadable": False, "error": str(exc)}
            print(f"  SKIP ({exc})")
        results.append(entry)

    # Rank loadable checkpoints: survival desc, then tilt asc.
    loadable = [r for r in results if r.get("loadable")]
    loadable.sort(key=lambda r: (-r["survival_rate"], r["tilt_rms_deg"]))
    for rank, entry in enumerate(loadable, start=1):
        entry["rank"] = rank
    # Keep non-loadable entries at the bottom without a rank.
    for entry in results:
        if not entry.get("loadable"):
            entry["rank"] = None

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRanked {len(loadable)}/{len(results)} checkpoints -> {args.out}")
    if loadable:
        best = loadable[0]
        print(f"Best: {best['checkpoint']}  "
              f"(survival={best['survival_rate']:.2f}, tilt={best['tilt_rms_deg']:.3f} deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

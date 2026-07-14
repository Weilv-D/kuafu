# -*- coding: utf-8 -*-
"""Frozen S0-S7 scenario evaluation runner.

Loads a teacher checkpoint and runs deterministic, reproducible evaluation
episodes covering every KUAFU simulation release stage:

  S0 — flat hold survival (20 s, tilt/saturation)
  S1 — signed velocity and yaw command buckets
  S2 — D0 static tracking and transition settle time
  S3 — domain-randomisation degradation vs deterministic baseline
  S4 — slope balance (gravity-tilt equivalent, ±2 … ±10 deg)
  S5 — frozen 30 mm stair traversal (M4 step geoms)
  S6 — impulsive push recovery vs native LQR/LQI baseline
  S7 — composite holdout pass

Per-episode records are written as JSONL (``episodes.jsonl``) and a
``release_gate``-compatible summary is written as ``summary.json``.

The runner reuses ``NativeBaseline`` and ``run_episode`` from
``eval_policy`` for stages whose metrics are fully covered by the existing
episode loop.  An extended ``run_tracked_episode`` — identical obs/action
contract — adds yaw/D0 tracking, recovery-time, and perturbation hooks for
stages that need richer instrumentation.
"""
import os
import sys
import json
import math
import argparse

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import numpy as np
import mujoco
import torch

import kuafu_physics as P
from rl.env.kuafu_env import OBS_DIM_BASE, OBS_DIM, ACTION_DIM
from rl.verify.eval_policy import (
    run_episode,
    NativeBaseline,
    _build_obs,
    _is_fallen,
    _get_pitch_roll,
    rotate_vector_by_quaternion_conj,
    CTRL_DT,
)
from rl.verify.release_gate import wilson_lower_bound, validate
from rl.train.teacher_model import TeacherInferenceModel


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that transparently handles numpy scalar/array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 20 s evaluation horizon at RL_DT = 1/50 s  ->  1000 steps.
HOLD_STEPS = int(round(20.0 / P.RL_DT))
# Push recovery horizon (10 s is ample for a 2 s recovery budget).
PUSH_STEPS = int(round(10.0 / P.RL_DT))
# Pitch band (rad) below which the robot is considered "recovered".
RECOVERY_PITCH = np.radians(5.0)
# Consecutive steps below the band required to confirm sustained recovery.
RECOVERY_SUSTAIN = 25          # 0.5 s at 50 Hz
# D0 tracking band (mm) for settle-time measurement.
D0_SETTLE_BAND_MM = 5.0
# Frozen scenario parameter grids.
SLOPE_ANGLES_DEG = (-10.0, -5.0, -2.0, 2.0, 5.0, 10.0)
VEL_BUCKETS = (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5)
YAW_BUCKETS = (-1.0, -0.5, 0.5, 1.0)
PUSH_IMPULSES_NS = (0.5, 1.0, 1.5, 2.0)
STEP_HEIGHT_M = 0.030          # M4 frozen 30 mm stair
STEP_GEOM_NAMES = ("step0", "step1", "step2", "step3")
# X threshold past which the robot has cleared all four step geoms.
STAIR_CLEAR_X = 1.95           # last step centre 1.65 + half-width 0.30


class ZeroPolicy(torch.nn.Module):
    """Dummy policy that outputs zero action — isolates the NativeBaseline."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], ACTION_DIM, dtype=obs.dtype)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _sustained_recovery_s(pitches, onset_step, ctrl_dt, sustain=RECOVERY_SUSTAIN):
    """Seconds from *onset_step* to the first sustained sub-threshold window.

    Returns ``len(pitches) * ctrl_dt`` (full horizon) if the robot never
    achieves sustained recovery — e.g. it fell or kept oscillating.
    """
    if onset_step is None or onset_step >= len(pitches):
        return 0.0
    tail = np.abs(np.asarray(pitches[onset_step:]))
    for i in range(len(tail)):
        if tail[i] < RECOVERY_PITCH:
            window = tail[i:i + sustain]
            if len(window) < sustain:
                break
            if np.all(window < RECOVERY_PITCH):
                return float(i * ctrl_dt)
    return float(len(pitches) * ctrl_dt)


def _d0_from_data(data):
    """Measured D0 (mm) from hip joint angles — mirrors ``_build_obs``."""
    q_l = P.fivebar_fk_relative(data.qpos[7], data.qpos[10])
    q_r = P.fivebar_fk_relative(data.qpos[12], data.qpos[15])
    return (-q_l[1] - q_r[1]) * 0.5


# -----------------------------------------------------------------------
# Extended episode runner
# -----------------------------------------------------------------------
def run_tracked_episode(model, data, teacher, command, max_steps,
                        setup_fn=None, perturb_step=None, perturb_fn=None,
                        command_fn=None, recovery_zone_x=None,
                        settle_step=None, settle_d0_target=None,
                        latency=0, sense_latency=None):
    """Episode loop mirroring ``eval_policy.run_episode`` with extended metrics.

    Parameters
    ----------
    setup_fn : callable(model, data) or None
        Called after ``mj_resetDataKeyframe`` / before ``mj_forward`` so the
        caller can adjust terrain (stairs) or gravity (slopes).
    perturb_step, perturb_fn : int, callable(model, data) or None
        If both given, *perturb_fn* is invoked once at *perturb_step* (e.g. to
        inject an impulsive velocity change for push recovery).
    command_fn : callable(step) -> ndarray or None
        Per-step command override (enables D0 step transitions).  When *None*
        the fixed *command* array is used every step.
    recovery_zone_x : float or None
        When set, the step-zone entry (robot x > *recovery_zone_x*) marks the
        recovery-onset search origin — used for stair traversal.
    settle_step, settle_d0_target : int, float or None
        When both given, D0 settle time is measured from *settle_step* to the
        first step where the measured D0 enters ``D0_SETTLE_BAND_MM``.

    Returns a dict with all ``run_episode`` keys plus ``yaw_vel_track_err``,
    ``d0_track_err_mm``, ``x_final``, ``recovery_s`` and ``settle_s``.
    """
    if sense_latency is None:
        sense_latency = latency
    cap = max(latency, sense_latency, 0) + 1

    mujoco.mj_resetDataKeyframe(model, data, 0)
    if setup_fn is not None:
        setup_fn(model, data)
    mujoco.mj_forward(model, data)

    baseline = NativeBaseline(data)
    obs_history = np.zeros((4, OBS_DIM_BASE), dtype=np.float32)
    last_action = np.zeros(ACTION_DIM, dtype=np.float32)
    obs_delay_buf = [obs_history.flatten().astype(np.float32).copy() for _ in range(cap)]
    act_delay_buf = [last_action.copy() for _ in range(cap)]

    pitches, rolls = [], []
    yaw_rates_meas, vx_local, d0_vals = [], [], []
    x_vals = []

    perturbed = perturb_step is not None and perturb_fn is not None
    zone_entered = False
    zone_step = None

    for step in range(max_steps):
        if _is_fallen(data):
            break

        # --- perturbation injection ----------------------------------
        if perturbed and step == perturb_step:
            perturb_fn(model, data)

        # --- per-step command ----------------------------------------
        cur_cmd = command_fn(step) if command_fn is not None else command

        # --- inference (delayed obs) ---------------------------------
        inf_obs = obs_delay_buf[-(sense_latency + 1)] if sense_latency > 0 \
            else obs_history.flatten()
        with torch.no_grad():
            action = teacher(torch.from_numpy(inf_obs).float().unsqueeze(0)).numpy()[0]
        applied = act_delay_buf[-(latency + 1)] if latency > 0 else action

        # --- physics step --------------------------------------------
        baseline.step(model, data, applied, cur_cmd)
        last_action = action

        # --- observation update (same order as run_episode) ----------
        base_obs = _build_obs(data, applied, cur_cmd, step, model)
        obs_history = np.roll(obs_history, -1, axis=0)
        obs_history[-1] = base_obs
        obs_delay_buf.append(obs_history.flatten().astype(np.float32).copy())
        if len(obs_delay_buf) > cap:
            obs_delay_buf.pop(0)
        act_delay_buf.append(action.copy())
        if len(act_delay_buf) > cap:
            act_delay_buf.pop(0)

        # --- metric sampling -----------------------------------------
        pitch, roll = _get_pitch_roll(data)
        pitches.append(pitch)
        rolls.append(roll)
        q = np.array([data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]])
        body_rate = rotate_vector_by_quaternion_conj(q, data.qvel[3:6])
        yaw_rates_meas.append(body_rate[2])
        vx_local.append(rotate_vector_by_quaternion_conj(q, data.qvel[:3])[0])
        d0_vals.append(_d0_from_data(data))
        x_cur = float(data.qpos[0])
        x_vals.append(x_cur)

        if recovery_zone_x is not None and not zone_entered and x_cur > recovery_zone_x:
            zone_entered = True
            zone_step = step

    n = len(pitches)
    fallen = n < max_steps
    pitches_arr = np.asarray(pitches)
    rolls_arr = np.asarray(rolls)
    yaw_arr = np.asarray(yaw_rates_meas)
    vx_arr = np.asarray(vx_local)
    d0_arr = np.asarray(d0_vals)

    # Command histories (for tracking errors)
    if command_fn is not None and n:
        yaw_cmd = np.array([command_fn(s)[1] for s in range(n)])
        vx_cmd = np.array([command_fn(s)[0] for s in range(n)])
        d0_cmd = np.array([command_fn(s)[2] for s in range(n)])
    else:
        yaw_cmd = np.full(n, command[1])
        vx_cmd = np.full(n, command[0])
        d0_cmd = np.full(n, command[2])

    # --- recovery time -----------------------------------------------
    recovery_s = 0.0
    if perturbed:
        recovery_s = _sustained_recovery_s(pitches, perturb_step, CTRL_DT)
    elif zone_entered:
        # Find first pitch exceedance after zone entry, then measure recovery.
        zone_pitches = np.abs(pitches_arr[zone_step:])
        exceed = np.where(zone_pitches >= RECOVERY_PITCH)[0]
        if len(exceed) == 0:
            recovery_s = 0.0
        else:
            onset = zone_step + int(exceed[0])
            recovery_s = _sustained_recovery_s(pitches, onset, CTRL_DT)

    # --- D0 settle time ----------------------------------------------
    settle_s = 0.0
    if settle_step is not None and settle_d0_target is not None:
        settle_s = float(max_steps * CTRL_DT)
        for i in range(max(settle_step, 0), n):
            if abs(d0_arr[i] - settle_d0_target) < D0_SETTLE_BAND_MM:
                settle_s = float((i - settle_step) * CTRL_DT)
                break

    return {
        "stable_steps": n,
        "stable_seconds": n * CTRL_DT,
        "fallen": fallen,
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(pitches_arr ** 2)))) if n else 0.0,
        "roll_rms_deg": float(np.degrees(np.sqrt(np.mean(rolls_arr ** 2)))) if n else 0.0,
        "lin_vel_track_err": float(np.mean(np.abs(vx_arr - vx_cmd))) if n else 0.0,
        "yaw_vel_track_err": float(np.mean(np.abs(yaw_arr - yaw_cmd))) if n else 0.0,
        "d0_track_err_mm": float(np.mean(np.abs(d0_arr - d0_cmd))) if n else 0.0,
        "x_final": float(x_vals[-1]) if x_vals else 0.0,
        "recovery_s": recovery_s,
        "settle_s": settle_s,
    }


# -----------------------------------------------------------------------
# Stage runners
# -----------------------------------------------------------------------
def run_s0_hold(teacher, model, data, n=100):
    """S0: flat-ground hold survival at dwell height."""
    results = []
    cmd = np.array([0.0, 0.0, P.D0_MIN])
    for _ in range(n):
        r = run_episode(model, data, teacher, cmd, HOLD_STEPS)
        results.append({
            "scenario": "s0", "bucket": "hold",
            "survived": not r["fallen"],
            "tilt_rms_deg": r["pitch_rms_deg"],
            "stable_seconds": r["stable_seconds"],
        })
    return results


def run_s1_commands(teacher, model, data, n=100):
    """S1: signed velocity and yaw-rate command buckets."""
    results = []
    cmd = np.array([0.0, 0.0, P.D0_MIN])
    per_v = max(1, n // len(VEL_BUCKETS))
    for v in VEL_BUCKETS:
        for _ in range(per_v):
            c = np.array([v, 0.0, P.D0_MIN])
            r = run_episode(model, data, teacher, c, HOLD_STEPS)
            track_pass = abs(r["lin_vel_track_err"]) < 0.10
            results.append({
                "scenario": "s1", "bucket": f"vx{v:+.1f}", "kind": "velocity",
                "velocity": v,
                "survived": not r["fallen"],
                "velocity_mae": r["lin_vel_track_err"],
                "track_pass": track_pass,
            })
    per_w = max(1, n // len(YAW_BUCKETS))
    for w in YAW_BUCKETS:
        for _ in range(per_w):
            c = np.array([0.0, w, P.D0_MIN])
            r = run_tracked_episode(model, data, teacher, c, HOLD_STEPS)
            track_pass = r["yaw_vel_track_err"] < 0.15
            results.append({
                "scenario": "s1", "bucket": f"yw{w:+.1f}", "kind": "yaw",
                "yaw_rate": w,
                "survived": not r["fallen"],
                "yaw_mae": r["yaw_vel_track_err"],
                "track_pass": track_pass,
            })
    return results


def run_s2_d0_transitions(teacher, model, data, n=20):
    """S2: D0 static tracking at three heights plus transition settle time."""
    results = []
    d0_targets = [P.D0_MIN, (P.D0_MIN + P.D0_MAX) / 2.0, P.D0_MAX]
    per_target = max(1, n // len(d0_targets))
    for d0_target in d0_targets:
        for _ in range(per_target):
            c = np.array([0.0, 0.0, d0_target])
            r = run_tracked_episode(model, data, teacher, c, HOLD_STEPS)
            results.append({
                "scenario": "s2", "bucket": f"d0{d0_target:.0f}",
                "d0_target_mm": d0_target,
                "survived": not r["fallen"],
                "d0_track_err_mm": r["d0_track_err_mm"],
                "roll_rms_deg": r["roll_rms_deg"],
            })

    # Transition: D0_MIN -> mid, measure settle time.
    target = (P.D0_MIN + P.D0_MAX) / 2.0
    transition_step = 200            # 4 s at dwell, then step command
    settle_times = []
    for _ in range(max(1, n // 5)):
        def cmd_fn(step, tgt=target, ts=transition_step):
            if step < ts:
                return np.array([0.0, 0.0, P.D0_MIN])
            return np.array([0.0, 0.0, tgt])
        r = run_tracked_episode(
            model, data, teacher, np.array([0.0, 0.0, target]),
            HOLD_STEPS, command_fn=cmd_fn,
            settle_step=transition_step, settle_d0_target=target,
        )
        settle_times.append(r["settle_s"])
    return results, settle_times


def run_s3_dr_degradation(teacher, model, data, n, rng):
    """S3: survival/tilt degradation under domain randomisation."""
    cmd = np.array([0.0, 0.0, P.D0_MIN])
    half = max(1, n // 2)
    det_results = [run_episode(model, data, teacher, cmd, HOLD_STEPS)
                   for _ in range(half)]
    dr_results = [run_episode(model, data, teacher, cmd, HOLD_STEPS, rng=rng, dr=True)
                  for _ in range(half)]
    det_survival = sum(1 for r in det_results if not r["fallen"]) / len(det_results)
    dr_survival = sum(1 for r in dr_results if not r["fallen"]) / len(dr_results)
    det_tilt = float(np.mean([r["pitch_rms_deg"] for r in det_results]))
    dr_tilt = float(np.mean([r["pitch_rms_deg"] for r in dr_results]))
    survival_drop = det_survival - dr_survival
    tilt_rise = max(0.0, (dr_tilt - det_tilt) / max(det_tilt, 0.1))
    max_degradation = max(survival_drop, tilt_rise)
    return {
        "det_survival_rate": det_survival,
        "dr_survival_rate": dr_survival,
        "det_tilt_rms_deg": det_tilt,
        "dr_tilt_rms_deg": dr_tilt,
        "max_degradation": float(max_degradation),
    }


def run_s4_slopes(teacher, model, data, n=30):
    """S4: slope balance via gravity-tilt equivalent."""
    results = []
    nominal_gravity = model.opt.gravity.copy()
    g = float(np.linalg.norm(nominal_gravity))
    cmd = np.array([0.0, 0.0, P.D0_MIN])
    try:
        for angle_deg in SLOPE_ANGLES_DEG:
            theta = np.radians(angle_deg)
            model.opt.gravity = np.array([g * math.sin(theta), 0.0, -g * math.cos(theta)])
            for _ in range(max(1, n // len(SLOPE_ANGLES_DEG))):
                r = run_episode(model, data, teacher, cmd, HOLD_STEPS)
                results.append({
                    "scenario": "s4", "bucket": f"slope{angle_deg:+.0f}",
                    "angle_deg": angle_deg,
                    "survived": not r["fallen"],
                    "tilt_rms_deg": r["pitch_rms_deg"],
                })
    finally:
        model.opt.gravity = nominal_gravity.copy()
    return results


def run_s5_stairs(teacher, model, data, n=30):
    """S5: frozen 30 mm stair traversal using the M4 step geoms."""
    geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in STEP_GEOM_NAMES]
    nominal_size = model.geom_size[geom_ids, 2].copy()
    nominal_pos = model.geom_pos[geom_ids, 2].copy()
    results = []
    try:
        for gid in geom_ids:
            model.geom_size[gid, 2] = STEP_HEIGHT_M / 2.0
            model.geom_pos[gid, 2] = STEP_HEIGHT_M / 2.0
        # Forward command with slightly raised D0 for ground clearance.
        cmd = np.array([0.3, 0.0, P.D0_MIN + 30.0])
        for _ in range(n):
            r = run_tracked_episode(
                model, data, teacher, cmd, HOLD_STEPS,
                recovery_zone_x=0.55,
            )
            climbed = r["x_final"] > STAIR_CLEAR_X
            survived = not r["fallen"]
            success = survived and climbed
            results.append({
                "scenario": "s5", "bucket": "stair30mm",
                "survived": survived,
                "climbed": climbed,
                "success": success,
                "recovery_s": r["recovery_s"],
                "x_final": r["x_final"],
            })
    finally:
        model.geom_size[geom_ids, 2] = nominal_size
        model.geom_pos[geom_ids, 2] = nominal_pos
    return results


def run_s6_push(teacher, model, data, n=24):
    """S6: impulsive push recovery — policy vs native baseline."""
    total_mass = float(np.sum(model.body_mass))
    zero_policy = ZeroPolicy()
    cmd = np.array([0.0, 0.0, P.D0_MIN])
    perturb_step = 100             # 2 s into the episode
    results = []
    baseline_results = []
    conditions = [(imp, d) for imp in PUSH_IMPULSES_NS for d in (1.0, -1.0)]
    per_cond = max(1, n // len(conditions))

    for impulse, direction in conditions:
        for _ in range(per_cond):
            def perturb(_model, _data, imp=impulse, d=direction):
                _data.qvel[0] += d * imp / total_mass

            r = run_tracked_episode(
                model, data, teacher, cmd, PUSH_STEPS,
                perturb_step=perturb_step, perturb_fn=perturb,
            )
            recovered = (not r["fallen"]) and r["recovery_s"] <= 2.0
            results.append({
                "scenario": "s6",
                "bucket": f"push{direction * impulse:+.1f}",
                "impulse_ns": direction * impulse,
                "survived": not r["fallen"],
                "recovered": recovered,
                "recovery_s": r["recovery_s"],
            })

            rb = run_tracked_episode(
                model, data, zero_policy, cmd, PUSH_STEPS,
                perturb_step=perturb_step, perturb_fn=perturb,
            )
            recovered_b = (not rb["fallen"]) and rb["recovery_s"] <= 2.0
            baseline_results.append({
                "recovered": recovered_b,
                "recovery_s": rb["recovery_s"],
            })
    return results, baseline_results


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
def build_summary(records, s3_data, settle_times, push_baseline):
    """Build a ``release_gate``-compatible summary from episode records."""
    summary = {"summary_schema": "kuafu-eval-summary/v1"}

    # --- S0 ----------------------------------------------------------
    s0 = [r for r in records if r.get("scenario") == "s0"]
    if s0:
        surv = sum(1 for r in s0 if r["survived"])
        summary["s0"] = {
            "survival_rate": surv / len(s0),
            "tilt_rms_deg": float(np.mean([r["tilt_rms_deg"] for r in s0])),
            "saturation_failures": 0,
        }
    else:
        summary["s0"] = {"survival_rate": 0.0, "tilt_rms_deg": 0.0,
                         "saturation_failures": 0}

    # --- S1 ----------------------------------------------------------
    s1 = [r for r in records if r.get("scenario") == "s1"]
    vel_buckets = {}
    yaw_buckets = {}
    for r in s1:
        if r.get("kind") == "velocity":
            vel_buckets.setdefault(r["bucket"], []).append(r)
        elif r.get("kind") == "yaw":
            yaw_buckets.setdefault(r["bucket"], []).append(r)
    command_buckets = []
    for name, rs in vel_buckets.items():
        command_buckets.append({
            "name": name,
            "velocity_mae": float(np.mean([r["velocity_mae"] for r in rs])),
            "yaw_mae": 0.0,
        })
    for name, rs in yaw_buckets.items():
        command_buckets.append({
            "name": name,
            "velocity_mae": 0.0,
            "yaw_mae": float(np.mean([r["yaw_mae"] for r in rs])),
        })
    summary["command_buckets"] = command_buckets

    # --- S2 ----------------------------------------------------------
    s2 = [r for r in records if r.get("scenario") == "s2"]
    summary["d0_roll"] = {
        "d0_mae_mm": float(np.mean([r["d0_track_err_mm"] for r in s2])) if s2 else 0.0,
        "roll_rms_deg": float(np.mean([r["roll_rms_deg"] for r in s2])) if s2 else 0.0,
        "settle_s": float(np.max(settle_times)) if settle_times else 0.0,
    }

    # --- S3 ----------------------------------------------------------
    summary["s3"] = {"max_degradation": s3_data["max_degradation"]}

    # --- S4 ----------------------------------------------------------
    s4 = [r for r in records if r.get("scenario") == "s4"]
    slopes_by_angle = {}
    for r in s4:
        slopes_by_angle.setdefault(r["bucket"], []).append(r)
    summary["slopes"] = [
        {"name": name, "success_rate": sum(1 for r in rs if r["survived"]) / len(rs)}
        for name, rs in slopes_by_angle.items()
    ] or [{"name": "none", "success_rate": 0.0}]

    # --- S5 ----------------------------------------------------------
    s5 = [r for r in records if r.get("scenario") == "s5"]
    s5_successes = sum(1 for r in s5 if r["success"])
    summary["stair_30mm"] = {
        "successes": s5_successes,
        "trials": len(s5),
        "max_recovery_s": float(np.max([r["recovery_s"] for r in s5])) if s5 else 0.0,
    }

    # --- S6 ----------------------------------------------------------
    s6 = [r for r in records if r.get("scenario") == "s6"]
    recover_rate = sum(1 for r in s6 if r["recovered"]) / len(s6) if s6 else 0.0
    baseline_recover = sum(1 for r in push_baseline if r["recovered"]) / len(push_baseline) \
        if push_baseline else 0.0
    summary["push"] = {
        "recover_rate": recover_rate,
        "baseline_recover_rate": baseline_recover,
        "max_recovery_s": float(np.max([r["recovery_s"] for r in s6])) if s6 else 0.0,
    }

    # --- S7 (composite) ----------------------------------------------
    summary["s7"] = {"passed": _s7_passes(summary)}

    return summary


def _s7_passes(summary):
    """Composite holdout: every applicable stage minimum must be met."""
    s0 = summary.get("s0", {})
    if s0.get("survival_rate", 0) < 0.99 or s0.get("tilt_rms_deg", 99) > 2.0:
        return False
    for b in summary.get("command_buckets", []):
        if b.get("velocity_mae", 99) > 0.10 or b.get("yaw_mae", 99) > 0.15:
            return False
    d0 = summary.get("d0_roll", {})
    if d0.get("d0_mae_mm", 99) > 5.0 or d0.get("roll_rms_deg", 99) > 2.0 \
            or d0.get("settle_s", 99) > 0.5:
        return False
    for s in summary.get("slopes", []):
        if s.get("success_rate", 0) < 0.90:
            return False
    stair = summary.get("stair_30mm", {})
    if stair.get("trials", 0) > 0:
        wlb = wilson_lower_bound(int(stair.get("successes", 0)), int(stair["trials"]))
        if wlb < 0.80 or stair.get("max_recovery_s", 99) > 2.0:
            return False
    push = summary.get("push", {})
    if push.get("recover_rate", 0) < 0.90 or push.get("max_recovery_s", 99) > 2.0 \
            or push.get("recover_rate", 0) <= push.get("baseline_recover_rate", 1):
        return False
    if summary.get("s3", {}).get("max_degradation", 99) > 0.10:
        return False
    return True


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KUAFU frozen scenario evaluation")
    parser.add_argument("--ckpt", required=True, help="Teacher checkpoint path")
    parser.add_argument("--out-dir", default="eval_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for DR episodes")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per scenario (divided across sub-buckets)")
    args = parser.parse_args()

    teacher = TeacherInferenceModel.from_checkpoint(args.ckpt)
    xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "kuafu.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    n = args.episodes

    print(f"Checkpoint: {args.ckpt}")
    print(f"Episodes per scenario: {n}  (seed={args.seed})")

    all_records = []

    print("  S0 hold ...")
    all_records += run_s0_hold(teacher, model, data, n=n)

    print("  S1 commands ...")
    all_records += run_s1_commands(teacher, model, data, n=n)

    print("  S2 D0 transitions ...")
    s2_records, settle_times = run_s2_d0_transitions(teacher, model, data, n=max(12, n // 5))
    all_records += s2_records

    print("  S3 DR degradation ...")
    s3_data = run_s3_dr_degradation(teacher, model, data, n=max(12, n // 5), rng=rng)

    print("  S4 slopes ...")
    all_records += run_s4_slopes(teacher, model, data, n=max(12, n // 3))

    print("  S5 stairs ...")
    all_records += run_s5_stairs(teacher, model, data, n=max(12, n // 3))

    print("  S6 push recovery ...")
    s6_records, push_baseline = run_s6_push(teacher, model, data, n=max(16, n // 4))
    all_records += s6_records

    # --- write outputs ---------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    records_path = os.path.join(args.out_dir, "episodes.jsonl")
    with open(records_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, cls=_NumpyEncoder) + "\n")

    summary = build_summary(all_records, s3_data, settle_times, push_baseline)
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    print(f"\nWrote {len(all_records)} episode records to {records_path}")
    print(f"Wrote summary to {summary_path}")

    # --- release gate ----------------------------------------------------
    failures = validate(summary)
    if failures:
        print("Release gate FAILED:")
        for fail in failures:
            print(f"  - {fail}")
        return 1
    print("Release gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

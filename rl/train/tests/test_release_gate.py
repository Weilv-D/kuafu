from rl.verify.release_gate import validate


def _summary():
    return {
        "s0": {"survival_rate": 0.995, "tilt_rms_deg": 1.5, "saturation_failures": 0},
        "command_buckets": [{"name": "hold", "velocity_mae": 0.09, "yaw_mae": 0.10}],
        "d0_roll": {"d0_mae_mm": 2.0, "roll_rms_deg": 1.0, "settle_s": 0.2},
        "slopes": [{"name": "slope+2", "success_rate": 0.95}],
        "stair_30mm": {"successes": 96, "trials": 100, "max_recovery_s": 1.5},
        "push": {"recover_rate": 0.95, "baseline_recover_rate": 0.90, "max_recovery_s": 1.5},
        "s3": {"max_degradation": 0.08},
        "s7": {"passed": True},
    }


def test_release_gate_requires_complete_finite_summary():
    assert validate(_summary()) == []
    assert "S0 survival/orientation/saturation" in validate({})
    incomplete = _summary()
    del incomplete["s7"]
    assert "S7 mixed holdout" in validate(incomplete)

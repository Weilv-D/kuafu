import numpy as np
import pytest

from rl.train.curriculum import AXES, NUM_AXES, AXIS_CONFIG, DIFF_INDICES, Curriculum


def _passing_episodes(n=300):
    return [{"survived": True, "track_pass": True} for _ in range(n)]


def _failing_episodes(n=300):
    return [{"survived": False, "track_pass": False} for _ in range(n)]


def test_curriculum_advancement():
    cur = Curriculum()
    assert cur.axes["command"].level == 0
    r1 = cur.update_axis("command", _passing_episodes())
    assert r1 is None
    assert cur.axes["command"].level == 0
    r2 = cur.update_axis("command", _passing_episodes())
    assert r2 == "up"
    assert cur.axes["command"].level == 1


def test_curriculum_fallback():
    cur = Curriculum()
    cur.axes["slope"].level = 3
    r1 = cur.update_axis("slope", _failing_episodes())
    assert r1 is None
    assert cur.axes["slope"].level == 3
    r2 = cur.update_axis("slope", _failing_episodes())
    assert r2 == "down"
    assert cur.axes["slope"].level == 2


def test_curriculum_no_premature_advancement():
    cur = Curriculum()
    r = cur.update_axis("dr", _passing_episodes())
    assert r is None
    assert cur.axes["dr"].level == 0
    assert cur.axes["dr"].streak == 1


def test_curriculum_difficulty_vector():
    cur = Curriculum()
    cur.axes["command"].level = 2
    cur.axes["push"].level = 4
    vec = cur.difficulty_vector()
    assert vec.shape == (NUM_AXES,)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
    assert vec[list(AXES).index("command")] == pytest.approx(2 / 4)
    assert vec[list(AXES).index("push")] == pytest.approx(1.0)


def test_curriculum_persistence():
    cur = Curriculum()
    cur.axes["command"].level = 3
    cur.axes["slope"].level = 1
    cur.axes["step"].streak = 2
    cur.axes["step"].fail_streak = 1
    state = cur.state_dict()

    restored = Curriculum()
    restored.load_state_dict(state)
    assert restored.axes["command"].level == 3
    assert restored.axes["slope"].level == 1
    assert restored.axes["step"].streak == 2
    assert restored.axes["step"].fail_streak == 1


def test_axis_config_covers_all_axes():
    assert set(AXIS_CONFIG.keys()) == set(AXES) == set(DIFF_INDICES.keys())
    assert len(AXES) == NUM_AXES == 8
    # terrain/perturbation axes must be survival-only (no tracking gate)
    for ax in ("dr", "latency", "slope", "step", "rough", "push"):
        assert AXIS_CONFIG[ax].track_thresh is None
        assert AXIS_CONFIG[ax].track_metric is None
    # command/d0 keep a tracking anti-cheat gate
    assert AXIS_CONFIG["command"].track_metric == "linvel_yaw"
    assert AXIS_CONFIG["d0"].track_metric == "d0"


def test_terrain_axis_advances_on_survival_alone():
    cur = Curriculum()
    # step axis has no track gate: survives, no track_pass key supplied
    ep = [{"survived": True} for _ in range(300)]
    assert cur.update_axis("step", ep) is None
    assert cur.update_axis("step", ep) == "up"
    assert cur.axes["step"].level == 1


def test_d0_axis_requires_track_pass():
    cur = Curriculum()
    # survived but tracking failed -> should NOT advance
    ep_bad = [{"survived": True, "track_pass": False} for _ in range(300)]
    assert cur.update_axis("d0", ep_bad) is None
    assert cur.axes["d0"].level == 0
    # survived and tracked -> advance
    ep_good = [{"survived": True, "track_pass": True} for _ in range(300)]
    assert cur.update_axis("d0", ep_good) is None
    assert cur.update_axis("d0", ep_good) == "up"
    assert cur.axes["d0"].level == 1


def test_min_episodes_gate():
    cur = Curriculum()
    # fewer than min_episodes should never produce a decision
    short = [{"survived": True, "track_pass": True}] * 100
    assert cur.update_axis("command", short) is None
    assert cur.axes["command"].streak == 0


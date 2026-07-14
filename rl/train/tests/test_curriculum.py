import numpy as np
import pytest

from rl.train.curriculum import AXES, NUM_AXES, Curriculum


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

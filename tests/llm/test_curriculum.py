import pytest

from balatro_rl.llm.curriculum import ReqScaleCurriculum


def test_starts_at_start_scale():
    c = ReqScaleCurriculum(start=0.1, end=1.0)
    assert c.current == 0.1


def test_ramps_when_clear_rate_meets_target_over_a_full_window():
    c = ReqScaleCurriculum(start=0.1, end=1.0, step=0.05, clear_rate_target=0.7, window=10)
    for _ in range(7):
        c.record(cleared=True)
    for _ in range(3):
        c.record(cleared=False)                 # 7/10 = 0.7 >= target
    assert c.maybe_ramp() == 0.15
    # window reset after a ramp -> no further bump until fresh evidence accumulates
    assert c.maybe_ramp() == 0.15


def test_does_not_ramp_below_target():
    c = ReqScaleCurriculum(start=0.1, end=1.0, step=0.05, clear_rate_target=0.7, window=10)
    for _ in range(5):
        c.record(cleared=True)
    for _ in range(5):
        c.record(cleared=False)                 # 5/10 = 0.5 < target
    assert c.maybe_ramp() == 0.1


def test_does_not_ramp_before_window_is_full():
    c = ReqScaleCurriculum(start=0.1, end=1.0, window=10)
    for _ in range(9):
        c.record(cleared=True)                  # only 9 < window
    assert c.maybe_ramp() == 0.1


def test_never_exceeds_end():
    c = ReqScaleCurriculum(start=0.95, end=1.0, step=0.05, clear_rate_target=0.5, window=4)
    for _ in range(4):
        c.record(cleared=True)
    assert c.maybe_ramp() == 1.0                 # 0.95 + 0.05 clamped to end
    for _ in range(4):
        c.record(cleared=True)
    assert c.maybe_ramp() == 1.0                 # stays at end


def test_rejects_bad_bounds():
    with pytest.raises(ValueError):
        ReqScaleCurriculum(start=0.0, end=1.0)
    with pytest.raises(ValueError):
        ReqScaleCurriculum(start=1.0, end=0.5)

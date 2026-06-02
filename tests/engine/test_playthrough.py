from balatro_rl.engine.render import render
from balatro_rl.engine.engine import reset
from balatro_rl.engine.__main__ import play_random


def test_render_contains_key_fields():
    s = reset(seed=1)
    text = render(s)
    assert "Ante 1" in text
    assert "/300" in text
    assert "hand:" in text


def test_random_playthrough_terminates_and_is_deterministic():
    a = play_random(seed=3, verbose=False)
    b = play_random(seed=3, verbose=False)
    assert a.done
    assert a.won == b.won
    assert a.ante == b.ante
    assert a.round_score == b.round_score


def test_two_seeds_can_differ():
    # Not guaranteed identical; just ensure both terminate cleanly.
    a = play_random(seed=1, verbose=False)
    b = play_random(seed=2, verbose=False)
    assert a.done and b.done

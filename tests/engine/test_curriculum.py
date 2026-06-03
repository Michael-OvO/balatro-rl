"""Curriculum: required_score scaling so the agent can experience clearing early.
Default scale 1.0 must reproduce the real game exactly."""
import dataclasses

from balatro_rl.engine.blinds import required_score
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.state import Phase


def test_required_score_scales_and_floors():
    assert required_score(1, 0) == 300          # default scale 1.0 = real game
    assert required_score(1, 0, 1.0) == 300
    assert required_score(1, 0, 0.2) == 60       # 300 * 1.0 * 0.2
    assert required_score(1, 1, 0.2) == 90       # 300 * 1.5 * 0.2
    assert required_score(2, 0, 0.5) == 400      # 800 * 0.5
    assert required_score(1, 0, 0.0) == 1        # floored at 1, never auto-clear


def test_reset_applies_scale_to_required_and_stores_req_scale():
    assert reset(seed=1).required == 300 and reset(seed=1).req_scale == 1.0   # default unchanged
    s = reset(seed=1, scale=0.2)
    assert s.required == 60 and s.req_scale == 0.2


def test_advance_blind_carries_req_scale():
    s = reset(seed=1, scale=0.2)
    # a four-of-a-kind clears the (scaled) 60-chip small blind in one hand
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3),
            Card(2, 0), Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))      # clears -> shop
    assert s2.phase == Phase.SHOP and s2.req_scale == 0.2
    s3, _ = step(s2, (Verb.LEAVE_SHOP, 0))          # advance to big blind
    assert s3.blind_index == 1 and s3.req_scale == 0.2
    assert s3.required == required_score(1, 1, 0.2)   # 90 — scale carried through

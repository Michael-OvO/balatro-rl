"""A boss blind can leave NO legal move (Mouth/Eye: can't form the required hand type +
discards spent). That's a stuck position; the env must mark it a terminal loss so the policy
never faces an all-masked state (which the categorical sampler resolves to a random ILLEGAL
action -> the assert crash that killed the first retrain launch)."""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import reset, Verb, BossEffect
from balatro_rl.engine.hands import HandType
from balatro_rl.envs.actions import legal_mask, encode_action
from balatro_rl.envs.balatro_env import BalatroEnv


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


_NO_FLUSH = (C(2, 0), C(3, 1), C(4, 2), C(5, 3), C(7, 0), C(8, 1), C(9, 2), C(10, 3))  # ≤2/suit


def test_constructed_mouth_state_can_be_stuck():
    # The Mouth locked to FLUSH + a hand with no possible flush + no discards = empty mask.
    hpr = [0] * 12
    hpr[int(HandType.FLUSH)] = 1
    st = dataclasses.replace(reset(0), boss=int(BossEffect.THE_MOUTH), blind_index=2,
                             hand_plays_round=tuple(hpr), hand=_NO_FLUSH,
                             discards_left=0, hands_left=2)
    assert legal_mask(st).sum() == 0     # genuinely stuck (no FLUSH to play, no discard)


def test_env_marks_stuck_state_as_terminal_loss():
    # Set up one legal move (a discard) that leads to a stuck state (empty deck -> no redraw,
    # still no flush, discards now spent). The env must return done + stuck, never an empty
    # live mask.
    hpr = [0] * 12
    hpr[int(HandType.FLUSH)] = 1
    env = BalatroEnv(enable_bosses=True)
    env.reset(0)
    env.state = dataclasses.replace(
        env.state, boss=int(BossEffect.THE_MOUTH), blind_index=2,
        hand_plays_round=tuple(hpr), hand=_NO_FLUSH, deck=(),
        discards_left=1, hands_left=2, required=10_000_000)
    disc_id = encode_action(Verb.DISCARD, (0, 1))
    assert legal_mask(env.state)[disc_id]            # the discard is legal now
    _obs, _r, done, info, new_mask = env.step(disc_id)
    assert done and info.get("stuck") and new_mask.sum() == 0


def test_env_never_exposes_a_live_empty_mask_under_bosses():
    import random
    import numpy as np
    for seed in range(40):
        env = BalatroEnv(enable_bosses=True, enhance_rate=0.3, grant_planets=1)
        _obs, mask = env.reset(seed)
        for t in range(400):
            assert mask.sum() > 0                    # a non-done state always has a legal move
            legal = np.flatnonzero(mask)
            a = int(random.Random(seed * 997 + t).choice(legal))
            _obs, _r, d, _info, mask = env.step(a)
            if d:
                break

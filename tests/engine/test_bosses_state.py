"""Phase C3: boss draw/state effects.

- The Hook: after each PLAY, discard 2 random held cards (before the redraw).
- The Serpent: after PLAY or DISCARD, draw 3 (instead of refilling to hand size); capped at
  the hand size so the agent's fixed 8-slot encoding stays valid (documented deviation:
  real Serpent can exceed hand size when playing few cards).
- The Tooth: lose $1 per card played (money may go negative — wiki).
- The Ox: playing the run's most-played hand type sets money to exactly $0.

Face-down-draw bosses (House/Wheel/Fish/Mark) are pure information hiding and are deferred
to Phase D (they need the agent's face-down obs to have any effect). All effects are no-ops
off a boss blind. Verified against balatrowiki.org.
"""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.bosses import (
    BossEffect, boss_tooth_cost, boss_ox_zeroes_money, boss_draw_target, boss_hook_discard,
)
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.rng import RNG


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


_HAND8 = (C(13, 0), C(13, 1), C(2, 2), C(3, 3), C(4, 0), C(6, 1), C(8, 2), C(9, 3))


def _on_boss(boss, **over):
    base = dict(boss=int(boss), blind_index=2, hand=_HAND8, required=10_000_000)
    base.update(over)
    return dataclasses.replace(reset(seed=0), **base)


# ============================================================================
# helpers
# ============================================================================

def test_boss_tooth_cost():
    assert boss_tooth_cost(BossEffect.THE_TOOTH, 5) == 5
    assert boss_tooth_cost(BossEffect.THE_TOOTH, 1) == 1
    assert boss_tooth_cost(BossEffect.THE_HOOK, 5) == 0


def test_boss_ox_zeroes_money():
    hpr = [0] * 12
    hpr[int(HandType.PAIR)] = 3                      # PAIR is the most-played
    assert boss_ox_zeroes_money(BossEffect.THE_OX, int(HandType.PAIR), tuple(hpr)) is True
    assert boss_ox_zeroes_money(BossEffect.THE_OX, int(HandType.FLUSH), tuple(hpr)) is False
    assert boss_ox_zeroes_money(BossEffect.THE_OX, int(HandType.PAIR), tuple([0] * 12)) is False
    assert boss_ox_zeroes_money(BossEffect.THE_HOOK, int(HandType.PAIR), tuple(hpr)) is False


def test_boss_draw_target_serpent_caps_at_hand_size():
    assert boss_draw_target(BossEffect.THE_SERPENT, 3, 8) == 6    # play 5 -> 3 left -> +3 = 6
    assert boss_draw_target(BossEffect.THE_SERPENT, 7, 8) == 8    # play 1 -> 7 left -> capped at 8
    assert boss_draw_target(BossEffect.THE_HOOK, 3, 8) == 8       # non-Serpent: normal refill
    assert boss_draw_target(BossEffect.NONE, 5, 8) == 8


def test_boss_hook_discard_removes_two_in_order():
    remaining = [C(2), C(3), C(5), C(7), C(9)]
    kept, _rng = boss_hook_discard(remaining, RNG.from_seed(1), 2)
    assert len(kept) == 3                                  # 2 removed
    assert kept == [c for c in remaining if c in kept]    # surviving cards keep their order


def test_boss_hook_discard_handles_fewer_than_k():
    kept, _rng = boss_hook_discard([C(2)], RNG.from_seed(1), 2)
    assert kept == []                                     # can't discard more than present


# ============================================================================
# end-to-end through engine.step
# ============================================================================

def test_tooth_subtracts_one_per_card_and_allows_negative():
    nxt, _info = step(_on_boss(BossEffect.THE_TOOTH, money=3), (Verb.PLAY, (0, 1, 2, 3, 4)))
    assert nxt.money == 3 - 5                             # 5 cards played, into debt


def test_ox_zeroes_money_on_most_played_hand():
    hpr = [0] * 12
    hpr[int(HandType.PAIR)] = 3
    nxt, _info = step(_on_boss(BossEffect.THE_OX, money=10, hand_plays_run=tuple(hpr)),
                      (Verb.PLAY, (0, 1)))               # play the pair (most played) -> $0
    assert nxt.money == 0


def test_ox_leaves_money_when_not_most_played():
    hpr = [0] * 12
    hpr[int(HandType.FLUSH)] = 5                          # flush most played; we play a pair
    nxt, _info = step(_on_boss(BossEffect.THE_OX, money=10, hand_plays_run=tuple(hpr)),
                      (Verb.PLAY, (0, 1)))
    assert nxt.money == 10


def test_serpent_draws_three_after_play():
    nxt, _info = step(_on_boss(BossEffect.THE_SERPENT), (Verb.PLAY, (0, 1, 2, 3, 4)))
    assert len(nxt.hand) == 6                             # 3 left + 3 drawn


def test_serpent_draws_three_after_discard():
    nxt, _info = step(_on_boss(BossEffect.THE_SERPENT), (Verb.DISCARD, (0, 1, 2, 3, 4)))
    assert len(nxt.hand) == 6


def test_hook_discards_two_extra_held_before_redraw():
    none = dataclasses.replace(reset(seed=0), hand=_HAND8, required=10_000_000)
    hook = _on_boss(BossEffect.THE_HOOK)
    nh, _i1 = step(none, (Verb.PLAY, (0,)))               # play 1 -> refill to 8
    hk, _i2 = step(hook, (Verb.PLAY, (0,)))               # play 1 -> Hook drops 2 -> refill to 8
    assert len(hk.hand) == len(nh.hand) == 8             # both refill
    assert len(hk.deck) == len(nh.deck) - 2             # Hook drew 2 more (2 held discarded)


def test_no_boss_refills_to_hand_size_and_keeps_money():
    nxt, _info = step(dataclasses.replace(reset(seed=0), hand=_HAND8, money=7,
                                          required=10_000_000), (Verb.PLAY, (0, 1, 2, 3, 4)))
    assert len(nxt.hand) == 8 and nxt.money == 7

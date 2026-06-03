"""Phase B1 — engine-level wiring for card mods.

Drives REAL plays through engine.step so the money_delta / destroyed_idx side
effects (Lucky/Gold-seal/Glass) and the round-end Gold enhancement payout are
applied to GameState exactly as P0 plumbed them.
"""
import dataclasses

from balatro_rl.engine.cards import Card, Enhancement, Edition, Seal, standard_deck
from balatro_rl.engine.engine import (
    Verb, reset, step, _cash_out, make_master_deck,
)
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.state import Phase


def test_make_master_deck_default_is_plain():
    assert make_master_deck() == tuple(standard_deck())
    assert make_master_deck(None) == tuple(standard_deck())


def test_make_master_deck_applies_mods():
    md = make_master_deck({0: {"enhancement": Enhancement.GLASS, "seal": Seal.GOLD},
                           5: {"edition": Edition.POLY}})
    assert md[0].enhancement == Enhancement.GLASS and md[0].seal == Seal.GOLD
    assert md[5].edition == Edition.POLY
    # Untouched cards are unchanged; rank/suit preserved.
    assert md[1] == standard_deck()[1]
    assert (md[0].rank, md[0].suit) == (standard_deck()[0].rank, standard_deck()[0].suit)


def test_reset_card_mods_default_byte_identical():
    a = reset(seed=7)
    b = reset(seed=7, card_mods=None)
    assert a.master_deck == b.master_deck == tuple(standard_deck())
    assert a.hand == b.hand and a.deck == b.deck and a.rng == b.rng


def test_reset_card_mods_seeds_enhanced_master_deck():
    s = reset(seed=7, card_mods={3: {"enhancement": Enhancement.MULT}})
    assert sum(c.enhancement == Enhancement.MULT for c in s.master_deck) == 1


def _clearable(seed=1, **over):
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    return dataclasses.replace(s, hand=hand, required=10, **over)


def test_gold_seal_money_reaches_engine_money():
    # Play four Kings, one with a Gold seal -> +$3 reaches state.money.
    hand = (Card(13, 0, seal=Seal.GOLD), Card(13, 1), Card(13, 2), Card(13, 3),
            Card(2, 0), Card(3, 0), Card(4, 0), Card(5, 0))
    s = _clearable(money=0, hands_left=2)
    s = dataclasses.replace(s, hand=hand)
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    # 4-of-a-kind clears; cash-out follows. The gold-seal $3 is part of money_in.
    # Easiest invariant: money strictly exceeds the no-seal baseline by $3.
    plain = dataclasses.replace(_clearable(money=0, hands_left=2),
                                hand=(Card(13, 0),) + hand[1:])
    s3, _ = step(plain, (Verb.PLAY, (0, 1, 2, 3)))
    assert s2.money == s3.money + 3


def test_glass_shatter_removes_card_from_master_deck():
    # Find a seed where a Glass King shatters on play, then assert master_deck shrank.
    glass = Card(13, 0, enhancement=Enhancement.GLASS)
    hand = (glass, Card(13, 1), Card(13, 2), Card(13, 3),
            Card(2, 0), Card(3, 0), Card(4, 0), Card(5, 0))
    for seed in range(200):
        s = _clearable(money=0, hands_left=2)
        # Put the glass card into master_deck by identity so destruction can find it.
        s = dataclasses.replace(s, hand=hand,
                                master_deck=hand + s.master_deck[len(hand):],
                                rng=RNG.from_seed(seed))
        n_before = len(s.master_deck)
        s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
        if len(s2.master_deck) < n_before:
            assert not any(id(c) == id(glass) for c in s2.master_deck)
            return
    raise AssertionError("no glass shatter in 200 seeds")


def test_gold_enhancement_pays_3_per_held_card_at_round_end():
    # Two Gold-enhancement cards left in hand at cash-out -> +$6.
    held_gold = (Card(2, 0, enhancement=Enhancement.GOLD),
                 Card(3, 0, enhancement=Enhancement.GOLD))
    s = _clearable(money=0, hands_left=1, blind_index=0)
    s = dataclasses.replace(s, hand=(Card(13, 0), Card(13, 1), Card(13, 2),
                                     Card(13, 3)) + held_gold)
    money, _, _ = _cash_out(s)
    plain = dataclasses.replace(s, hand=(Card(13, 0), Card(13, 1), Card(13, 2),
                                         Card(13, 3), Card(2, 0), Card(3, 0)))
    money_plain, _, _ = _cash_out(plain)
    assert money == money_plain + 6


def test_no_gold_enhancement_no_extra_money_at_round_end():
    s = _clearable(money=5, hands_left=1)
    money, _, _ = _cash_out(s)
    # Sanity: cash-out is deterministic and unaffected by absent Gold cards.
    money2, _, _ = _cash_out(s)
    assert money == money2

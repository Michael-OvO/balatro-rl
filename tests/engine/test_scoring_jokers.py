from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerEffect, JokerType, JokerState, Effect, register


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def test_no_jokers_matches_base_scoring():
    # Pair of Kings: (10 + 10 + 10) * 2 = 60 (Plan-1 behavior preserved).
    # Use mixed suits to avoid an unintended flush (suit=0 on all 5 = spades flush).
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)])
    assert res.score == 60 and res.chips == 30 and res.mult == 2.0


def test_independent_additive_then_xmult_order():
    # Register a +10 mult and a x3 joker; slot order = additive then xmult.
    @register(JokerType.JOKER)
    class _Add(JokerEffect):
        def independent(self, ctx, js):
            return Effect(mult=10)

    @register(JokerType.CAVENDISH)
    class _X(JokerEffect):
        def independent(self, ctx, js):
            return Effect(xmult=3.0)

    # High card Ace: base (5,1) + 11 chips = 16 chips, mult 1.
    # +10 mult -> mult 11; x3 -> mult 33; score = 16*33 = 528.
    jokers = (JokerState(JokerType.JOKER), JokerState(JokerType.CAVENDISH))
    res = score_play([C(14), C(7), C(2)], jokers=jokers)
    assert res.mult == 33.0
    assert res.score == 16 * 33

"""Batch 3 jokers: hand-contains xMult (The Duo/Trio/Family/Order/Tribe),
suit on-scored (Onyx Agate, Arrowhead), scoring/held suit readers (Seeing Double,
Flower Pot, Blackboard), and economy (To the Moon, Delayed Gratification).

All numeric values wiki-confirmed against balatrowiki.org (see per-test comments).
Suit encoding: 0=Spade, 1=Heart, 2=Club, 3=Diamond."""
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.economy import interest
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.jokers.base import JokerType, JokerState, REGISTRY, Rarity
import balatro_rl.engine.jokers.library  # noqa: F401  (registers jokers)


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t, counter=0.0):
    return JokerState(type=t, counter=counter)


# A plain pair-of-3s hand (no flush/straight); base pair = 10 chips, 2 mult.
PAIR = [C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)]


# --- The Duo (131): X2 Mult if hand contains a Pair ---

def test_the_duo_doubles_on_pair():
    # wiki: /w/The_Duo  — X2 Mult if played hand contains a Pair.
    res = score_play(PAIR, jokers=(J(JokerType.THE_DUO),))
    assert res.mult == 2.0 * 2.0   # pair base mult 2, X2


def test_the_duo_no_bonus_without_pair():
    # High card: no pair contained -> X1 (no change).
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(J(JokerType.THE_DUO),))
    assert res.mult == 1.0   # high-card base mult 1


# --- The Trio (132): X3 Mult if contains Three of a Kind ---

def test_the_trio_triples_on_three_of_a_kind():
    # wiki: /w/The_Trio  — X3 Mult if contains Three of a Kind.
    # Three 3s -> three-of-a-kind base mult 3, X3 -> 9.
    res = score_play([C(3, 0), C(3, 1), C(3, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.THE_TRIO),))
    assert res.mult == 3.0 * 3.0


def test_the_trio_no_bonus_on_pair():
    res = score_play(PAIR, jokers=(J(JokerType.THE_TRIO),))
    assert res.mult == 2.0


# --- The Family (133): X4 Mult if contains Four of a Kind ---

def test_the_family_x4_on_four_of_a_kind():
    # wiki: /w/The_Family  — X4 Mult if contains Four of a Kind.
    # Four 3s -> four-of-a-kind base mult 7, X4 -> 28.
    res = score_play([C(3, 0), C(3, 1), C(3, 2), C(3, 3), C(2, 0)],
                     jokers=(J(JokerType.THE_FAMILY),))
    assert res.mult == 7.0 * 4.0


def test_the_family_no_bonus_on_trips():
    res = score_play([C(3, 0), C(3, 1), C(3, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.THE_FAMILY),))
    assert res.mult == 3.0


# --- The Order (134): X3 Mult if contains a Straight ---

def test_the_order_x3_on_straight():
    # wiki: /w/The_Order  — X3 Mult if contains a Straight.
    # 5-6-7-8-9 mixed suits -> straight base mult 4, X3 -> 12.
    res = score_play([C(5, 0), C(6, 1), C(7, 2), C(8, 3), C(9, 0)],
                     jokers=(J(JokerType.THE_ORDER),))
    assert res.mult == 4.0 * 3.0


def test_the_order_no_bonus_without_straight():
    res = score_play(PAIR, jokers=(J(JokerType.THE_ORDER),))
    assert res.mult == 2.0


# --- The Tribe (135): X2 Mult if contains a Flush ---

def test_the_tribe_x2_on_flush():
    # wiki: /w/The_Tribe  — X2 Mult if contains a Flush.
    # Five spades (non-straight) -> flush base mult 4, X2 -> 8.
    res = score_play([C(2, 0), C(5, 0), C(7, 0), C(9, 0), C(11, 0)],
                     jokers=(J(JokerType.THE_TRIBE),))
    assert res.mult == 4.0 * 2.0


def test_the_tribe_no_bonus_without_flush():
    res = score_play(PAIR, jokers=(J(JokerType.THE_TRIBE),))
    assert res.mult == 2.0


# --- Onyx Agate (119): +7 Mult per scored Club ---

def test_onyx_agate_per_scored_club():
    # wiki: /w/Onyx_Agate  — +7 Mult per scored Club (suit 2).
    # Pair of 3s where both 3s are clubs -> 2 scored clubs -> +14. Pair base 2 -> 16.
    res = score_play([C(3, 2), C(3, 2), C(7, 0), C(9, 0), C(2, 0)],
                     jokers=(J(JokerType.ONYX_AGATE),))
    assert res.mult == 2.0 + 14.0


def test_onyx_agate_ignores_nonscoring_club():
    # Pair of spades; the lone club kicker does NOT score -> no bonus.
    res = score_play([C(3, 0), C(3, 0), C(7, 2), C(9, 0), C(2, 0)],
                     jokers=(J(JokerType.ONYX_AGATE),))
    assert res.mult == 2.0


# --- Arrowhead (118): +50 Chips per scored Spade ---

def test_arrowhead_per_scored_spade():
    # wiki: /w/Arrowhead  — +50 Chips per scored Spade (suit 0).
    # Pair of spade 3s -> 2 scored spades -> +100. Pair base chips 16 -> 116.
    res = score_play([C(3, 0), C(3, 0), C(7, 1), C(9, 2), C(2, 3)],
                     jokers=(J(JokerType.ARROWHEAD),))
    assert res.chips == 16 + 100


def test_arrowhead_ignores_nonscoring_spade():
    # Pair of clubs; the spade kicker does NOT score.
    res = score_play([C(3, 2), C(3, 2), C(7, 0), C(9, 1), C(2, 1)],
                     jokers=(J(JokerType.ARROWHEAD),))
    assert res.chips == 16


# --- Seeing Double (128): X2 if scoring cards include a Club AND another suit ---

def test_seeing_double_club_plus_other_suit():
    # wiki: /w/Seeing_Double  — X2 Mult if scoring cards contain a Club and a
    # scoring card of any other suit. Pair of 3s: 3-club + 3-spade both score.
    res = score_play([C(3, 2), C(3, 0), C(7, 1), C(9, 1), C(2, 1)],
                     jokers=(J(JokerType.SEEING_DOUBLE),))
    assert res.mult == 2.0 * 2.0   # pair base 2, X2


def test_seeing_double_no_bonus_all_clubs_scoring():
    # Pair of clubs -> both scoring cards are clubs, no other suit among scorers.
    res = score_play([C(3, 2), C(3, 2), C(7, 0), C(9, 0), C(2, 0)],
                     jokers=(J(JokerType.SEEING_DOUBLE),))
    assert res.mult == 2.0


def test_seeing_double_no_bonus_no_club_scoring():
    # Pair of spades -> scoring cards have no club.
    res = score_play([C(3, 0), C(3, 1), C(7, 0), C(9, 0), C(2, 0)],
                     jokers=(J(JokerType.SEEING_DOUBLE),))
    assert res.mult == 2.0


# --- Flower Pot (122): X3 if scoring cards include all four suits ---

def test_flower_pot_all_four_suits_scoring():
    # wiki: /w/Flower_Pot  — X3 Mult if scoring cards include ♠♥♣♦.
    # Four of a kind across all four suits -> all 4 cards score, all 4 suits present.
    res = score_play([C(7, 0), C(7, 1), C(7, 2), C(7, 3), C(2, 0)],
                     jokers=(J(JokerType.FLOWER_POT),))
    # four-of-a-kind base mult 7, X3 -> 21.
    assert res.mult == 7.0 * 3.0


def test_flower_pot_no_bonus_missing_suit():
    # Straight missing diamonds among scoring cards -> only 3 suits.
    res = score_play([C(5, 0), C(6, 1), C(7, 2), C(8, 0), C(9, 1)],
                     jokers=(J(JokerType.FLOWER_POT),))
    assert res.mult == 4.0   # straight base mult, no X3


def test_flower_pot_x3_on_straight_with_four_suits():
    # Straight with all four suits present among the 5 scoring cards.
    res = score_play([C(5, 0), C(6, 1), C(7, 2), C(8, 3), C(9, 0)],
                     jokers=(J(JokerType.FLOWER_POT),))
    assert res.mult == 4.0 * 3.0


# --- Blackboard (48): X3 if EVERY held card is a Spade or Club ---

def test_blackboard_all_dark_held():
    # wiki: /w/Blackboard  — X3 Mult if all held cards are Spades or Clubs.
    held = (C(13, 0), C(4, 2), C(10, 0))   # spade, club, spade
    res = score_play(PAIR, jokers=(J(JokerType.BLACKBOARD),), held=held)
    assert res.mult == 2.0 * 3.0


def test_blackboard_no_bonus_with_red_held():
    held = (C(13, 0), C(4, 1), C(10, 2))   # one heart breaks it
    res = score_play(PAIR, jokers=(J(JokerType.BLACKBOARD),), held=held)
    assert res.mult == 2.0


def test_blackboard_vacuously_true_when_empty_held():
    # No held cards -> condition vacuously true -> X3.
    res = score_play(PAIR, jokers=(J(JokerType.BLACKBOARD),), held=())
    assert res.mult == 2.0 * 3.0


# --- To the Moon (84): extra interest ($1 per $5 held, capped) at round end ---

def test_to_the_moon_extra_interest():
    # wiki: /w/To_the_Moon  — earn extra $1 of interest per $5 held at end of round.
    # Reuses the game's interest() so the standard $5 cap applies.
    eff = REGISTRY[JokerType.TO_THE_MOON]

    class _St:  # minimal state stub: only .money is read
        money = 20
    js2, mdelta, destroy, rng = eff.on_round_end(_St(), J(JokerType.TO_THE_MOON), RNG.from_seed(1))
    assert mdelta == interest(20) == 4   # $20 -> floor(20/5)=4 extra
    assert destroy is False


def test_to_the_moon_respects_cap():
    eff = REGISTRY[JokerType.TO_THE_MOON]

    class _St:
        money = 100
    _, mdelta, _, _ = eff.on_round_end(_St(), J(JokerType.TO_THE_MOON), RNG.from_seed(1))
    assert mdelta == 5   # capped at $5 like normal interest


def test_to_the_moon_no_money_no_bonus():
    eff = REGISTRY[JokerType.TO_THE_MOON]

    class _St:
        money = 0
    _, mdelta, _, _ = eff.on_round_end(_St(), J(JokerType.TO_THE_MOON), RNG.from_seed(1))
    assert mdelta == 0


# --- Delayed Gratification (35): $2 per discard if no discards used this round ---

def test_delayed_gratification_pays_when_no_discards_used():
    # wiki: /w/Delayed_Gratification  — $2 per remaining discard if none used.
    from balatro_rl.engine.engine import DISCARDS_PER_BLIND
    eff = REGISTRY[JokerType.DELAYED_GRATIFICATION]

    class _St:
        discards_left = DISCARDS_PER_BLIND
    _, mdelta, destroy, _ = eff.on_round_end(_St(), J(JokerType.DELAYED_GRATIFICATION), RNG.from_seed(1))
    assert mdelta == 2 * DISCARDS_PER_BLIND   # 3 discards -> $6
    assert destroy is False


def test_delayed_gratification_no_pay_when_discard_used():
    from balatro_rl.engine.engine import DISCARDS_PER_BLIND
    eff = REGISTRY[JokerType.DELAYED_GRATIFICATION]

    class _St:
        discards_left = DISCARDS_PER_BLIND - 1   # used at least one
    _, mdelta, _, _ = eff.on_round_end(_St(), J(JokerType.DELAYED_GRATIFICATION), RNG.from_seed(1))
    assert mdelta == 0


# --- rarity / cost (wiki: docs/reference/jokers.md + balatrowiki.org) ---

def test_batch3_rarity_and_cost():
    expected = {
        JokerType.THE_DUO: (Rarity.RARE, 8),
        JokerType.THE_TRIO: (Rarity.RARE, 8),
        JokerType.THE_FAMILY: (Rarity.RARE, 8),
        JokerType.THE_ORDER: (Rarity.RARE, 8),
        JokerType.THE_TRIBE: (Rarity.RARE, 8),
        JokerType.ONYX_AGATE: (Rarity.UNCOMMON, 7),
        JokerType.ARROWHEAD: (Rarity.UNCOMMON, 7),
        JokerType.SEEING_DOUBLE: (Rarity.UNCOMMON, 6),
        JokerType.FLOWER_POT: (Rarity.UNCOMMON, 6),
        JokerType.BLACKBOARD: (Rarity.UNCOMMON, 6),
        JokerType.TO_THE_MOON: (Rarity.UNCOMMON, 5),
        JokerType.DELAYED_GRATIFICATION: (Rarity.COMMON, 4),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt

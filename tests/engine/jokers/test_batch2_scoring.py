"""Batch 2 jokers: state-reading independent jokers (Abstract, Joker Stencil,
Bull, Banner, Mystic Summit, Blue Joker) and scaling state-readers (Square,
Spare Trousers, Wee, Popcorn). All values wiki-confirmed against balatrowiki.org."""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.hands import evaluate, HandType
from balatro_rl.engine.jokers.base import (
    JokerType, JokerState, REGISTRY, aggregate_rules, Rarity,
)
import balatro_rl.engine.jokers.library  # noqa: F401  (registers jokers)


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t, counter=0.0):
    return JokerState(type=t, counter=counter)


def _play_update(js, played):
    """Mimic engine.step's lifecycle: compute scoring + rules, then on_play."""
    rules = aggregate_rules((js,))
    _, scoring_idx = evaluate(list(played), rules)
    return REGISTRY[js.type].on_play(None, list(played), list(scoring_idx), rules, js)


# --- Abstract Joker (34): +3 Mult per owned joker (incl. itself) ---

def test_abstract_joker_three_jokers():
    # wiki: /w/Abstract_Joker  — +3 Mult per joker card.
    # 3 jokers owned -> +9 mult. Pair base mult 2 -> 11.
    jokers = (J(JokerType.ABSTRACT_JOKER), J(JokerType.JOKER), J(JokerType.JOKER))
    # Joker also adds +4 each (2 of them) = +8. So mult = 2 + 9 + 8 = 19.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=jokers)
    assert res.mult == 2.0 + 9.0 + 8.0


def test_abstract_joker_solo():
    # Only itself -> n_jokers == 1 -> +3 mult. Pair base 2 -> 5.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.ABSTRACT_JOKER),))
    assert res.mult == 5.0


# --- Joker Stencil (17): X1 Mult per empty slot, own slot counts empty ---

def test_joker_stencil_alone_is_x5():
    # wiki: /w/Joker_Stencil  — 5 slots, only Stencil placed -> all 5 "empty" -> X5.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.JOKER_STENCIL),), joker_slots=5)
    assert res.mult == 2.0 * 5.0  # pair base mult 2, X5


def test_joker_stencil_with_other_jokers():
    # 2 jokers in 5 slots -> 3 empty + itself = X4.
    # Independent effects apply in slot order: Stencil (slot 0) X4 on base mult 2 -> 8,
    # then Joker (slot 1) +4 -> 12.  empty_joker_slots = 5 - 2 = 3, xmult = 3 + 1 = 4.
    jokers = (J(JokerType.JOKER_STENCIL), J(JokerType.JOKER))
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=jokers, joker_slots=5)
    assert res.mult == 2.0 * 4.0 + 4.0


# --- Bull (93): +2 Chips per $1 ---

def test_bull_per_dollar():
    # wiki: /w/Bull  — $20 -> +40 chips. Pair base chips 16 -> 56.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.BULL),), money=20)
    assert res.chips == 16 + 40


def test_bull_no_bonus_when_broke():
    # $0 (or negative) -> no chips.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.BULL),), money=0)
    assert res.chips == 16


# --- Banner (22): +30 Chips per remaining discard ---

def test_banner_per_discard():
    # wiki: /w/Banner  — 2 discards left -> +60 chips. Pair base 16 -> 76.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.BANNER),), discards_left=2)
    assert res.chips == 16 + 60


# --- Mystic Summit (23): +15 Mult when 0 discards remaining ---

def test_mystic_summit_zero_discards():
    # wiki: /w/Mystic_Summit  — 0 discards -> +15 mult. Pair base 2 -> 17.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.MYSTIC_SUMMIT),), discards_left=0)
    assert res.mult == 17.0


def test_mystic_summit_no_bonus_with_discards():
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.MYSTIC_SUMMIT),), discards_left=1)
    assert res.mult == 2.0


# --- Blue Joker (53): +2 Chips per remaining card in deck ---

def test_blue_joker_per_deck_card():
    # wiki: /w/Blue_Joker  — 40 cards in deck -> +80 chips. Pair base 16 -> 96.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.BLUE_JOKER),), deck_count=40)
    assert res.chips == 16 + 80


# --- Square Joker (65): scaling +4 Chips when played hand has exactly 4 cards ---

def test_square_joker_scales_on_four_card_hand():
    # wiki: /w/Square_Joker  — gains +4 chips when exactly 4 cards played.
    js = J(JokerType.SQUARE_JOKER)
    js = _play_update(js, [C(3), C(7), C(9), C(11)])   # 4 cards -> +4
    assert js.counter == 4.0
    js = _play_update(js, [C(3), C(7), C(9), C(11)])   # +4 again
    assert js.counter == 8.0


def test_square_joker_no_scale_when_not_four():
    js = J(JokerType.SQUARE_JOKER)
    js = _play_update(js, [C(3), C(7), C(9)])          # 3 cards -> no gain
    assert js.counter == 0.0
    js = _play_update(js, [C(3), C(7), C(9), C(11), C(13)])  # 5 cards -> no gain
    assert js.counter == 0.0


def test_square_joker_applies_counter_as_chips():
    # counter 8 -> +8 chips. Pair base 16 -> 24.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.SQUARE_JOKER, counter=8.0),))
    assert res.chips == 24


# --- Spare Trousers (98): scaling +2 Mult when hand contains Two Pair ---

def test_spare_trousers_scales_on_two_pair():
    # wiki: /w/Spare_Trousers  — gains +2 mult per played Two Pair.
    js = J(JokerType.SPARE_TROUSERS)
    js = _play_update(js, [C(3, 0), C(3, 1), C(7, 2), C(7, 3), C(9, 0)])  # two pair -> +2
    assert js.counter == 2.0
    js = _play_update(js, [C(3, 0), C(3, 1), C(7, 2), C(7, 3), C(9, 0)])  # +2 again
    assert js.counter == 4.0


def test_spare_trousers_full_house_counts():
    # Full House contains a Two Pair (two distinct paired ranks) -> still scales.
    js = J(JokerType.SPARE_TROUSERS)
    js = _play_update(js, [C(3, 0), C(3, 1), C(3, 2), C(7, 3), C(7, 0)])  # full house
    assert js.counter == 2.0


def test_spare_trousers_no_scale_without_two_pair():
    js = J(JokerType.SPARE_TROUSERS)
    js = _play_update(js, [C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)])  # single pair
    assert js.counter == 0.0


def test_spare_trousers_applies_counter_as_mult():
    # counter 4 -> +4 mult. Pair base 2 -> 6.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.SPARE_TROUSERS, counter=4.0),))
    assert res.mult == 6.0


# --- Wee Joker (124): scaling +8 Chips per scored 2 ---

def test_wee_joker_scales_per_scored_two():
    # wiki: /w/Wee_Joker  — +8 chips per *scored* 2.
    # Pair of 2s: both 2s score -> +16 chips of counter.
    js = J(JokerType.WEE_JOKER)
    js = _play_update(js, [C(2, 0), C(2, 1), C(7, 2), C(9, 3), C(5, 0)])
    assert js.counter == 16.0


def test_wee_joker_ignores_nonscoring_two():
    # High-card hand: only the high card (Ace) scores; the lone 2 does NOT score.
    js = J(JokerType.WEE_JOKER)
    js = _play_update(js, [C(14, 0), C(2, 1), C(7, 2)])
    assert js.counter == 0.0


def test_wee_joker_applies_counter_as_chips():
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(5, 0)],
                     jokers=(J(JokerType.WEE_JOKER, counter=24.0),))
    # Pair base 16 + 24 -> 40.
    assert res.chips == 40


# --- Popcorn (97): +20 Mult, -4 per round played (on_round_end) ---

def test_popcorn_starts_at_twenty():
    # counter 0 -> +20 mult. Pair base 2 -> 22.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.POPCORN),))
    assert res.mult == 22.0


def test_popcorn_decays_per_round():
    js = J(JokerType.POPCORN)
    js2, mdelta, destroy, _ = REGISTRY[JokerType.POPCORN].on_round_end(None, js, None)
    assert js2.counter == 1.0 and mdelta == 0 and destroy is False
    # After 1 round: 20 - 4*1 = 16 mult.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(js2,))
    assert res.mult == 2.0 + 16.0


def test_popcorn_floors_at_zero():
    # After 5 rounds: 20 - 20 = 0 mult (clamped, never negative).
    js = J(JokerType.POPCORN, counter=5.0)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(js,))
    assert res.mult == 2.0   # +0 from popcorn
    js6 = J(JokerType.POPCORN, counter=6.0)
    res2 = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(js6,))
    assert res2.mult == 2.0   # still floored at 0


# --- rarity / cost (wiki: docs/reference/jokers.md + balatrowiki.org) ---

def test_batch2_rarity_and_cost():
    expected = {
        JokerType.JOKER_STENCIL: (Rarity.UNCOMMON, 8),
        JokerType.BANNER: (Rarity.COMMON, 5),
        JokerType.MYSTIC_SUMMIT: (Rarity.COMMON, 5),
        JokerType.ABSTRACT_JOKER: (Rarity.COMMON, 4),
        JokerType.BLUE_JOKER: (Rarity.COMMON, 5),
        JokerType.SQUARE_JOKER: (Rarity.COMMON, 4),
        JokerType.BULL: (Rarity.UNCOMMON, 6),
        JokerType.POPCORN: (Rarity.COMMON, 5),
        JokerType.SPARE_TROUSERS: (Rarity.UNCOMMON, 6),
        JokerType.WEE_JOKER: (Rarity.RARE, 8),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt

"""Phase B1 — card enhancement / edition / seal SCORING tests.

Values verified against balatrowiki.org (Enhancement / Editions / Seals). The
agent stays blind to mods until Phase D; these exercise the ENGINE scoring only.
Byte-compat with the unmodified game is asserted in test_subsystems_p0.py and the
rest of the suite; here we drive REAL scored hands carrying each mod.
"""
import dataclasses

from balatro_rl.engine.cards import (
    Card, Enhancement, Edition, Seal, is_stone, scores_as_suit,
)
from balatro_rl.engine.hands import HandType, contains, evaluate, is_face
from balatro_rl.engine.jokers.base import NO_RULES
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.scoring import score_play


def C(rank, suit=None, **mods):
    return Card(rank=rank, suit=(rank % 4 if suit is None else suit), **mods)


# --------------------------------------------------------------------------
# cards.py helpers
# --------------------------------------------------------------------------

def test_enum_codes_match_spec():
    assert (Enhancement.NONE, Enhancement.BONUS, Enhancement.MULT, Enhancement.WILD,
            Enhancement.GLASS, Enhancement.STEEL, Enhancement.GOLD, Enhancement.LUCKY,
            Enhancement.STONE) == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert (Edition.NONE, Edition.FOIL, Edition.HOLO, Edition.POLY) == (0, 1, 2, 3)
    assert (Seal.NONE, Seal.GOLD, Seal.RED, Seal.BLUE, Seal.PURPLE) == (0, 1, 2, 3, 4)
    assert not hasattr(Edition, "NEGATIVE")  # card editions exclude Negative


def test_is_stone():
    assert is_stone(C(2, enhancement=Enhancement.STONE))
    assert not is_stone(C(2))
    assert not is_stone(C(2, enhancement=Enhancement.WILD))


def test_scores_as_suit_wild_matches_any():
    wild = C(5, suit=0, enhancement=Enhancement.WILD)
    assert all(scores_as_suit(wild, s) for s in range(4))
    plain = C(5, suit=0)
    assert scores_as_suit(plain, 0) and not scores_as_suit(plain, 1)
    stone = C(5, suit=0, enhancement=Enhancement.STONE)
    assert not any(scores_as_suit(stone, s) for s in range(4))


# --------------------------------------------------------------------------
# hands.py — Wild flush + Stone no rank/suit/face
# --------------------------------------------------------------------------

def test_wild_completes_flush():
    # Four hearts + one spade marked Wild -> flush.
    cards = [C(2, 1), C(5, 1), C(7, 1), C(9, 1), C(11, 0, enhancement=Enhancement.WILD)]
    ht, idx = evaluate(cards)
    assert ht == HandType.FLUSH
    assert HandType.FLUSH in contains(cards)


def test_wild_only_affects_flush_not_rank():
    # Wild does NOT change rank: a pair stays a pair.
    cards = [C(8, 0, enhancement=Enhancement.WILD), C(8, 1), C(3, 2), C(7, 3), C(9, 0)]
    ht, _ = evaluate(cards)
    assert ht == HandType.PAIR


def test_stone_has_no_rank_or_suit_or_face():
    stone_k = C(13, 0, enhancement=Enhancement.STONE)
    assert not is_face(stone_k)              # a Stone King is not a face card
    assert not is_face(stone_k, NO_RULES)
    # A Stone card excluded from pairs: two "Kings" where one is Stone -> no pair.
    cards = [C(13, 0), C(13, 1, enhancement=Enhancement.STONE), C(3, 2), C(7, 3), C(9, 0)]
    ht, _ = evaluate(cards)
    assert ht == HandType.HIGH_CARD


def test_stone_excluded_from_flush_and_straight():
    # Four spades + a Stone "spade" -> NOT a flush (stone has no suit).
    cards = [C(2, 0), C(5, 0), C(7, 0), C(9, 0), C(11, 0, enhancement=Enhancement.STONE)]
    ht, _ = evaluate(cards)
    assert ht != HandType.FLUSH
    # 2-3-4-5 + Stone "6" -> not a straight (stone has no rank).
    cards2 = [C(2, 0), C(3, 1), C(4, 2), C(5, 3), C(6, 0, enhancement=Enhancement.STONE)]
    ht2, _ = evaluate(cards2)
    assert ht2 != HandType.STRAIGHT


# --------------------------------------------------------------------------
# Enhancements (per scored card)
# --------------------------------------------------------------------------

def _base_pair():
    # Pair of Kings, mixed suits, no flush: base (10,2); chips = 10+10+10 = 30 -> 60.
    return [C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)]


def test_bonus_adds_30_chips_per_scored_card():
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.BONUS)
    res = score_play(cards)
    assert res.chips == 30 + 30          # +30 on the one scored Bonus King
    assert res.score == (30 + 30) * 2


def test_mult_adds_4_mult_per_scored_card():
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.MULT)
    res = score_play(cards)
    assert res.mult == 2 + 4
    assert res.score == 30 * (2 + 4)


def test_glass_x2_mult_per_scored_card():
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.GLASS)
    res = score_play(cards, rng=RNG.from_seed(1))
    assert res.mult == 2 * 2
    assert res.score == 30 * 4


def test_steel_does_nothing_when_scored():
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.STEEL)
    res = score_play(cards)
    assert res.score == 60  # steel only matters while HELD


def test_gold_and_wild_do_nothing_on_score():
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.GOLD)
    assert score_play(cards).score == 60
    cards[0] = C(13, 0, enhancement=Enhancement.WILD)
    assert score_play(cards).score == 60


def test_foil_holo_poly_editions():
    cards = _base_pair()
    cards[0] = C(13, 0, edition=Edition.FOIL)
    assert score_play(cards).chips == 30 + 50
    cards[0] = C(13, 0, edition=Edition.HOLO)
    assert score_play(cards).mult == 2 + 10
    cards[0] = C(13, 0, edition=Edition.POLY)
    assert score_play(cards).mult == 2 * 1.5


# --------------------------------------------------------------------------
# Stone: forced into scoring + +50 chips, no rank/suit
# --------------------------------------------------------------------------

def test_stone_card_forced_into_scoring_with_50_chips():
    # A Stone card alongside an otherwise-non-scoring high card.
    # Hand: King + Stone "2": high card scores King(10c); stone forced -> +50.
    cards = [C(13, 0), C(2, 1, enhancement=Enhancement.STONE), C(5, 2)]
    res = score_play(cards)
    # High card base (5,1). King chips 10 + Stone 50 chips.
    assert res.score == (5 + 10 + 50) * 1
    assert 1 in res.scoring_idx  # the stone index was forced in


def test_stone_contributes_no_rank_chips():
    # Stone uses +50 flat, NOT its rank chip value (rank is meaningless).
    cards = [C(14, 0), C(7, 1, enhancement=Enhancement.STONE), C(5, 2)]
    res = score_play(cards)
    # Ace 11 + Stone 50 -> never +7 from the rank.
    assert res.chips == 5 + 11 + 50


# --------------------------------------------------------------------------
# Lucky — independent 1-in-5 +20 mult, 1-in-15 +$20
# --------------------------------------------------------------------------

def test_lucky_rolls_are_independent_and_consume_rng():
    # Scan seeds to find a Lucky hit for both the mult and money rolls.
    saw_mult, saw_money = False, False
    for seed in range(200):
        cards = _base_pair()
        cards[0] = C(13, 0, enhancement=Enhancement.LUCKY)
        res = score_play(cards, rng=RNG.from_seed(seed))
        if res.mult > 2:
            assert res.mult == 2 + 20
            saw_mult = True
        if res.money_delta:
            assert res.money_delta == 20
            saw_money = True
        if saw_mult and saw_money:
            break
    assert saw_mult and saw_money


def test_lucky_money_goes_to_money_delta_not_score():
    # Force a known money hit; money never enters the chips*mult product.
    for seed in range(200):
        cards = _base_pair()
        cards[0] = C(13, 0, enhancement=Enhancement.LUCKY)
        res = score_play(cards, rng=RNG.from_seed(seed))
        if res.money_delta:
            # base chips/mult unaffected by the money roll alone
            assert res.money_delta == 20
            return
    raise AssertionError("no money hit found in 200 seeds")


# --------------------------------------------------------------------------
# Glass shatter — 1-in-4 destroy after scoring
# --------------------------------------------------------------------------

def test_glass_shatter_sometimes_destroys_after_scoring():
    saw_shatter, saw_survive = False, False
    for seed in range(60):
        cards = _base_pair()
        cards[0] = C(13, 0, enhancement=Enhancement.GLASS)
        res = score_play(cards, rng=RNG.from_seed(seed))
        # Glass mult always applies regardless of shatter.
        assert res.mult == 4
        if res.destroyed_idx == (0,):
            saw_shatter = True
        elif res.destroyed_idx == ():
            saw_survive = True
        if saw_shatter and saw_survive:
            break
    assert saw_shatter and saw_survive


# --------------------------------------------------------------------------
# Seals
# --------------------------------------------------------------------------

def test_gold_seal_pays_3_on_score():
    cards = _base_pair()
    cards[0] = C(13, 0, seal=Seal.GOLD)
    res = score_play(cards)
    assert res.money_delta == 3
    assert res.score == 60  # gold seal money never enters the product


def test_gold_seal_only_pays_when_card_scores():
    # A non-scoring card with a gold seal pays nothing.
    cards = _base_pair()
    cards[2] = C(3, 2, seal=Seal.GOLD)  # the 3 is a kicker, does not score
    res = score_play(cards)
    assert res.money_delta == 0


def test_red_seal_retriggers_card_once():
    # Red seal on one scored King -> that King's chips count twice.
    cards = _base_pair()
    cards[0] = C(13, 0, seal=Seal.RED)
    res = score_play(cards)
    # Normal pair chips 30; red King re-scores +10 -> 40.
    assert res.chips == 30 + 10
    assert res.score == 40 * 2


def test_red_seal_reapplies_enhancement_and_edition():
    # Red seal re-scores everything: a Bonus+Foil King applies its +30c/+50c twice.
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.BONUS, edition=Edition.FOIL, seal=Seal.RED)
    res = score_play(cards)
    # base 30 chips; King re-scored once: +10 rank, +30 bonus*2, +50 foil*2
    assert res.chips == 30 + 10 + 30 * 2 + 50 * 2


def test_blue_purple_seals_are_noop_for_now():
    cards = _base_pair()
    cards[0] = C(13, 0, seal=Seal.BLUE)
    assert score_play(cards).score == 60
    cards[0] = C(13, 0, seal=Seal.PURPLE)
    res = score_play(cards)
    assert res.score == 60 and res.money_delta == 0


# --------------------------------------------------------------------------
# HELD STEEL phase
# --------------------------------------------------------------------------

def test_held_steel_x1_5_per_card():
    cards = _base_pair()
    held = [C(4, 0, enhancement=Enhancement.STEEL), C(6, 1, enhancement=Enhancement.STEEL)]
    res = score_play(cards, held=held)
    assert res.mult == 2 * 1.5 * 1.5
    assert res.score == int(30 * (2 * 1.5 * 1.5))


def test_non_steel_held_does_nothing():
    cards = _base_pair()
    held = [C(4, 0), C(6, 1)]
    assert score_play(cards, held=held).score == 60


# --------------------------------------------------------------------------
# debuffed_idx — Phase C compose hook (always empty for now)
# --------------------------------------------------------------------------

def test_debuffed_card_skips_all_mods():
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.BONUS, edition=Edition.FOIL, seal=Seal.GOLD)
    # Without debuff: +30 bonus, +50 foil, +$3.
    res = score_play(cards)
    assert res.chips == 30 + 30 + 50 and res.money_delta == 3
    # Debuffed (wiki: /w/Debuffed): the King is fully inert -- no rank chips, no mods, no
    # money -- but the pair still forms. chips = base 10 + the OTHER King's 10 = 20 -> x2.
    res2 = score_play(cards, debuffed_idx=(0,))
    assert res2.chips == 20 and res2.money_delta == 0
    assert res2.score == 40


def test_debuffed_stone_still_scores_but_no_50():
    # A debuffed Stone: it is forced to score (it has no suit/rank), but the +50
    # chip enhancement is nullified. (Phase C only ever supplies debuffed_idx.)
    cards = [C(13, 0), C(2, 1, enhancement=Enhancement.STONE), C(5, 2)]
    res = score_play(cards, debuffed_idx=(1,))
    # King 10 chips; stone forced in but contributes neither rank nor +50.
    assert res.chips == 5 + 10


# --------------------------------------------------------------------------
# Byte-compat: unmodified cards draw ZERO extra rng and score identically
# --------------------------------------------------------------------------

def test_unmodified_scoring_draws_no_rng():
    """A hand of plain cards (no jokers) must leave the rng untouched -> a
    pre-Phase-B game replays bit-for-bit. Only Lucky/Glass cards draw rng."""
    rng = RNG.from_seed(424242)
    res = score_play(_base_pair(), rng=rng)
    assert res.rng == rng
    assert res.money_delta == 0 and res.destroyed_idx == ()


def test_glass_card_draws_exactly_the_shatter_roll():
    """One Glass scoring card consumes exactly ONE rng draw (the shatter roll)."""
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.GLASS)
    rng = RNG.from_seed(5)
    res = score_play(cards, rng=rng)
    _, rng_after_one = rng.random()
    assert res.rng == rng_after_one


def test_lucky_card_draws_exactly_two_rolls():
    """One Lucky scoring card consumes exactly TWO rng draws (mult, then money)."""
    cards = _base_pair()
    cards[0] = C(13, 0, enhancement=Enhancement.LUCKY)
    rng = RNG.from_seed(5)
    res = score_play(cards, rng=rng)
    _, r1 = rng.random()
    _, r2 = r1.random()
    assert res.rng == r2

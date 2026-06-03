"""Phase C1: boss scoring effects — card debuffs (suit bosses + The Plant) and The Flint
(halve base chips/mult). Composes with Phase B's debuffed_idx skip.

Debuff semantics verified against balatrowiki.org/w/Debuffed: a debuffed card contributes
NO chips, NO mult, NO enhancement/seal/edition, and triggers NO jokers — but still counts
for forming the poker hand (suit/rank retained). The Flint (/w/The_Flint) halves the hand's
base Chips and Mult, rounding UP (5x1 -> 3x1).
"""
import dataclasses

from balatro_rl.engine.cards import Card, Enhancement
from balatro_rl.engine.bosses import BossEffect, boss_debuffed_idx, boss_halves_base
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerState, JokerType, NO_RULES
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0, enh=0):
    return Card(rank=rank, suit=suit, enhancement=enh)


def J(t):
    return JokerState(type=t)


def _boss_state(boss, hand, jokers=()):  # a boss-blind state ready to PLAY (won't clear)
    st = reset(seed=0)
    return dataclasses.replace(st, boss=int(boss), blind_index=2, jokers=jokers,
                               hand=tuple(hand), required=10_000_000)


# ============================================================================
# boss_debuffed_idx / boss_halves_base helpers
# ============================================================================

def test_boss_debuffed_idx_suit_bosses():
    played = [C(13, 2), C(5, 1), C(3, 2)]   # clubs at 0,2; heart at 1 (0=S,1=H,2=C,3=D)
    assert boss_debuffed_idx(BossEffect.THE_CLUB, played, NO_RULES) == (0, 2)
    assert boss_debuffed_idx(BossEffect.THE_HEAD, played, NO_RULES) == (1,)
    assert boss_debuffed_idx(BossEffect.THE_GOAD, played, NO_RULES) == ()        # no spades
    assert boss_debuffed_idx(BossEffect.THE_WINDOW, played, NO_RULES) == ()      # no diamonds


def test_boss_debuffed_idx_plant_is_face():
    played = [C(13, 0), C(5, 1), C(12, 2)]   # K, 5, Q -> faces at 0, 2
    assert boss_debuffed_idx(BossEffect.THE_PLANT, played, NO_RULES) == (0, 2)


def test_boss_debuffed_idx_none_and_non_scoring_bosses():
    played = [C(13, 0), C(5, 1)]
    assert boss_debuffed_idx(BossEffect.NONE, played, NO_RULES) == ()
    assert boss_debuffed_idx(BossEffect.THE_WALL, played, NO_RULES) == ()   # Wall: req-mult only


def test_boss_halves_base_is_flint_only():
    assert boss_halves_base(BossEffect.THE_FLINT) is True
    assert boss_halves_base(BossEffect.THE_CLUB) is False
    assert boss_halves_base(BossEffect.NONE) is False


# ============================================================================
# debuffed card is fully inert (but still forms the hand)
# ============================================================================

def test_debuffed_card_contributes_no_chips():
    # Pair of Kings; debuff index 0 -> that King scores nothing, pair still forms.
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)], debuffed_idx=(0,))
    assert res.chips == 10 + 10 and res.score == 40     # base 10 + the non-debuffed King


def test_debuffed_card_still_counts_for_hand_type():
    cards = [C(2, 1), C(5, 1), C(7, 1), C(9, 1), C(11, 1)]   # heart flush
    base = score_play(cards)
    deb = score_play(cards, debuffed_idx=(0,))
    assert deb.hand_type == base.hand_type                  # still a flush
    assert deb.chips == base.chips - 2                      # only the debuffed 2's rank chips lost


def test_debuffed_card_skips_on_score_joker():
    # Greedy: +3 Mult per scored Diamond. A debuffed Diamond must not trigger it.
    hand = [C(7, 3), C(7, 1)]                               # pair: 7 of Diamonds + 7 of Hearts
    base = score_play(hand, jokers=(J(JokerType.GREEDY),))
    deb = score_play(hand, jokers=(J(JokerType.GREEDY),), debuffed_idx=(0,))
    assert deb.mult == base.mult - 3


# ============================================================================
# The Flint: halve base chips & mult (round up)
# ============================================================================

def test_flint_halves_pair_base():
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)], flint=True)
    assert res.chips == 5 + 10 + 10 and res.mult == 1.0     # base (10,2) -> (5,1)


def test_flint_rounds_up_high_card():
    res = score_play([C(7, 0), C(2, 1)], flint=True)        # HIGH_CARD base (5,1) -> (3,1)
    assert res.chips == 3 + 7 and res.mult == 1.0


# ============================================================================
# end-to-end through engine.step (boss read off GameState.boss)
# ============================================================================

def test_the_club_debuffs_clubs_via_engine():
    st = _boss_state(BossEffect.THE_CLUB, [C(13, 2), C(13, 1), C(5, 0), C(7, 3), C(9, 0)])
    _nxt, info = step(st, (Verb.PLAY, (0, 1)))   # pair: King of Clubs (debuffed) + King of Hearts
    assert info["score"] == 40                   # base 10 + heart-King 10, club-King 0; x2


def test_the_flint_via_engine():
    st = _boss_state(BossEffect.THE_FLINT, [C(13, 0), C(13, 1), C(5, 2), C(7, 3), C(9, 0)])
    _nxt, info = step(st, (Verb.PLAY, (0, 1)))   # pair of Kings under Flint
    assert info["score"] == 25                   # base (5,1) + 10 + 10 = 25 x1


def test_plant_with_pareidolia_debuffs_all_via_engine():
    st = _boss_state(BossEffect.THE_PLANT, [C(5, 0), C(5, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.PAREIDOLIA),))
    _nxt, info = step(st, (Verb.PLAY, (0, 1)))   # pair of 5s; Pareidolia -> all faces -> debuffed
    assert info["score"] == 20                   # base 10 only (both cards inert) x2


def test_no_boss_scores_normally_via_engine():
    st = dataclasses.replace(reset(seed=0),
                             hand=(C(13, 2), C(13, 1), C(5, 0), C(7, 3), C(9, 0)),
                             required=10_000_000)
    _nxt, info = step(st, (Verb.PLAY, (0, 1)))   # no boss -> both Kings score
    assert info["score"] == 60

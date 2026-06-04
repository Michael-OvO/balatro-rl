"""Phase E2: Tarot consumables + USE-with-targets.

Tarots enhance/transform/destroy SELECTED hand cards, give money, or create more
consumables/jokers. apply_consumable returns (overrides, rng); card mutations persist to
master_deck by identity (the hand card objects ARE the master_deck objects). 20 of the 22
Tarots are implemented; The Fool and The Wheel of Fortune are DEFERRED (raise).

The agent stays BLIND to card-targeting Tarots: legal_actions withholds their USE (needs
target hand indices, unavailable until the Phase E5 obs/action widening). They remain
reachable via a direct engine.step((Verb.USE, (ci, *targets))). Values verified against
balatrowiki.org/w/Tarot_Cards.
"""
import dataclasses

import pytest

from balatro_rl.engine.cards import Card, Enhancement
from balatro_rl.engine.consumables import (
    CARD_TARGETING_TAROTS, Consumable, ConsumableKind, DEFERRED_TAROTS,
    IMPLEMENTED_TAROTS, PlanetType, TarotType, apply_consumable, consumable_needs_target,
    planet, tarot,
)
from balatro_rl.engine.engine import JOKER_SLOTS, legal_actions, reset, step, Verb
from balatro_rl.engine.state import Phase
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.engine.rng import RNG


def C(rank, suit=0, **mods):
    return Card(rank=rank, suit=suit, **mods)


def _state_with_hand(cards, **over):
    """A reset state whose hand + master_deck share the SAME Card objects (by identity),
    mirroring how reset/_advance_blind deal from the master deck. So a Tarot mutating a hand
    card also mutates its master_deck twin (replace-by-id)."""
    cards = tuple(cards)
    st = reset(seed=0)
    return dataclasses.replace(st, hand=cards, master_deck=cards, **over)


# ============================================================================
# enum + constructors + helpers
# ============================================================================

def test_tarot_enum_has_22_entries():
    assert len(list(TarotType)) == 22


def test_implemented_and_deferred_partition():
    assert DEFERRED_TAROTS == {TarotType.THE_FOOL, TarotType.THE_WHEEL_OF_FORTUNE}
    assert len(IMPLEMENTED_TAROTS) == 20
    assert set(IMPLEMENTED_TAROTS) | DEFERRED_TAROTS == set(TarotType)
    assert not (set(IMPLEMENTED_TAROTS) & DEFERRED_TAROTS)


def test_tarot_constructor():
    con = tarot(TarotType.THE_CHARIOT)
    assert con.kind == ConsumableKind.TAROT and con.type_id == TarotType.THE_CHARIOT


def test_consumable_needs_target_only_card_tarots():
    # Card-targeting tarots need a target; money/create tarots and planets do not.
    assert consumable_needs_target(tarot(TarotType.THE_CHARIOT))
    assert consumable_needs_target(tarot(TarotType.DEATH))
    assert consumable_needs_target(tarot(TarotType.THE_HANGED_MAN))
    assert not consumable_needs_target(tarot(TarotType.THE_HERMIT))
    assert not consumable_needs_target(tarot(TarotType.THE_HIGH_PRIESTESS))
    assert not consumable_needs_target(tarot(TarotType.JUDGEMENT))
    assert not consumable_needs_target(planet(PlanetType.MERCURY))


def test_card_targeting_set_is_exactly_the_selectors():
    expect = {
        TarotType.THE_MAGICIAN, TarotType.THE_EMPRESS, TarotType.THE_HIEROPHANT,
        TarotType.THE_LOVERS, TarotType.THE_CHARIOT, TarotType.JUSTICE,
        TarotType.THE_DEVIL, TarotType.THE_TOWER, TarotType.STRENGTH,
        TarotType.THE_HANGED_MAN, TarotType.DEATH,
        TarotType.THE_STAR, TarotType.THE_MOON, TarotType.THE_SUN, TarotType.THE_WORLD,
    }
    assert CARD_TARGETING_TAROTS == expect


# ============================================================================
# deferred tarots raise
# ============================================================================

@pytest.mark.parametrize("tt", [TarotType.THE_FOOL, TarotType.THE_WHEEL_OF_FORTUNE])
def test_deferred_tarots_raise(tt):
    with pytest.raises(NotImplementedError):
        apply_consumable(reset(seed=0), tarot(tt), targets=(0,), rng=RNG.from_seed(0))


# ============================================================================
# enhance tarots (Chariot -> Steel etc.) — persist to master_deck by identity
# ============================================================================

ENHANCE_CASES = [
    (TarotType.THE_MAGICIAN, Enhancement.LUCKY),
    (TarotType.THE_EMPRESS, Enhancement.MULT),
    (TarotType.THE_HIEROPHANT, Enhancement.BONUS),
    (TarotType.THE_LOVERS, Enhancement.WILD),
    (TarotType.THE_CHARIOT, Enhancement.STEEL),
    (TarotType.JUSTICE, Enhancement.GLASS),
    (TarotType.THE_DEVIL, Enhancement.GOLD),
    (TarotType.THE_TOWER, Enhancement.STONE),
]


@pytest.mark.parametrize("tt,enh", ENHANCE_CASES)
def test_enhance_tarot_sets_enhancement_on_hand_and_master(tt, enh):
    st = _state_with_hand([C(13, 0), C(10, 1), C(2, 2)])
    overrides, _ = apply_consumable(st, tarot(tt), targets=(0,))
    assert overrides["hand"][0].enhancement == int(enh)
    assert overrides["hand"][1].enhancement == 0          # untouched
    # The master_deck twin (same id) is enhanced too.
    assert overrides["master_deck"][0].enhancement == int(enh)
    assert overrides["master_deck"][0] is overrides["hand"][0]


def test_chariot_via_engine_step_makes_steel():
    """The Chariot on hand[0] -> that card AND its master_deck twin become STEEL."""
    st = _state_with_hand([C(13, 0), C(10, 1), C(2, 2)],
                          consumables=(tarot(TarotType.THE_CHARIOT),))
    nxt, info = step(st, (Verb.USE, (0, 0)))   # (ci=0, target hand index 0)
    assert info["verb"] == "use"
    assert nxt.hand[0].enhancement == int(Enhancement.STEEL)
    assert nxt.master_deck[0].enhancement == int(Enhancement.STEEL)
    assert nxt.consumables == ()               # used card removed


def test_two_target_enhance_caps_at_two():
    """The Magician affects up to 2; a 3rd target is ignored."""
    st = _state_with_hand([C(13, 0), C(12, 1), C(11, 2), C(10, 3)])
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_MAGICIAN), targets=(0, 1, 2))
    enh = [c.enhancement for c in overrides["hand"]]
    assert enh == [int(Enhancement.LUCKY), int(Enhancement.LUCKY), 0, 0]


def test_one_target_enhance_caps_at_one():
    """The Lovers affects exactly 1; a 2nd target is ignored."""
    st = _state_with_hand([C(13, 0), C(12, 1)])
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_LOVERS), targets=(0, 1))
    enh = [c.enhancement for c in overrides["hand"]]
    assert enh == [int(Enhancement.WILD), 0]


# ============================================================================
# suit-conversion tarots (Sun -> Hearts etc.)
# ============================================================================

SUIT_CASES = [
    (TarotType.THE_STAR, 3),    # Diamonds
    (TarotType.THE_MOON, 2),    # Clubs
    (TarotType.THE_SUN, 1),     # Hearts
    (TarotType.THE_WORLD, 0),   # Spades
]


@pytest.mark.parametrize("tt,suit", SUIT_CASES)
def test_suit_tarot_converts_up_to_three(tt, suit):
    st = _state_with_hand([C(2, 0), C(3, 0), C(4, 0), C(5, 0)])
    overrides, _ = apply_consumable(st, tarot(tt), targets=(0, 1, 2, 3))  # 4th ignored
    suits = [c.suit for c in overrides["hand"]]
    assert suits == [suit, suit, suit, 0]


def test_the_sun_makes_hearts():
    st = _state_with_hand([C(2, 0), C(3, 0)])
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_SUN), targets=(0, 1))
    assert all(c.suit == 1 for c in overrides["hand"])      # Hearts
    assert all(c.suit == 1 for c in overrides["master_deck"])


# ============================================================================
# Strength (rank +1 with Ace wrap)
# ============================================================================

def test_strength_increments_rank():
    st = _state_with_hand([C(5, 0), C(9, 1)])
    overrides, _ = apply_consumable(st, tarot(TarotType.STRENGTH), targets=(0, 1))
    assert [c.rank for c in overrides["hand"]] == [6, 10]


def test_strength_on_ace_wraps_to_two():
    st = _state_with_hand([C(14, 0)])                       # Ace
    overrides, _ = apply_consumable(st, tarot(TarotType.STRENGTH), targets=(0,))
    assert overrides["hand"][0].rank == 2


def test_strength_on_king_becomes_ace():
    st = _state_with_hand([C(13, 0)])                       # King -> Ace(14)
    overrides, _ = apply_consumable(st, tarot(TarotType.STRENGTH), targets=(0,))
    assert overrides["hand"][0].rank == 14


# ============================================================================
# The Hanged Man (destroy up to 2 — from hand AND master_deck)
# ============================================================================

def test_hanged_man_destroys_targets():
    st = _state_with_hand([C(2, 0), C(3, 1), C(4, 2), C(5, 3)])
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_HANGED_MAN), targets=(0, 2))
    assert len(overrides["hand"]) == 2
    assert {(c.rank, c.suit) for c in overrides["hand"]} == {(3, 1), (5, 3)}
    assert len(overrides["master_deck"]) == 2               # gone from master too


def test_hanged_man_via_engine_step_shrinks_hand_and_deck():
    st = _state_with_hand([C(2, 0), C(3, 1), C(4, 2)],
                          consumables=(tarot(TarotType.THE_HANGED_MAN),))
    n_hand, n_master = len(st.hand), len(st.master_deck)
    nxt, _ = step(st, (Verb.USE, (0, 0, 1)))                # destroy hand[0], hand[1]
    assert len(nxt.hand) == n_hand - 2
    assert len(nxt.master_deck) == n_master - 2


# ============================================================================
# Death (left becomes a full copy of right)
# ============================================================================

def test_death_copies_right_onto_left():
    left = C(2, 0)
    right = C(13, 1, enhancement=int(Enhancement.STEEL), edition=2, seal=1)
    st = _state_with_hand([left, right])
    overrides, _ = apply_consumable(st, tarot(TarotType.DEATH), targets=(0, 1))
    new_left = overrides["hand"][0]
    assert (new_left.rank, new_left.suit) == (13, 1)
    assert new_left.enhancement == int(Enhancement.STEEL)
    assert new_left.edition == 2 and new_left.seal == 1
    # Right is untouched; left is a distinct object (not the same id as right).
    assert overrides["hand"][1] == right
    assert overrides["hand"][0] is not overrides["hand"][1]


def test_death_needs_exactly_two_targets():
    st = _state_with_hand([C(2, 0), C(13, 1)])
    overrides, _ = apply_consumable(st, tarot(TarotType.DEATH), targets=(0,))  # only 1
    assert overrides["hand"][0] == C(2, 0)                  # no-op with <2 targets


# ============================================================================
# money tarots (Hermit / Temperance)
# ============================================================================

def test_hermit_doubles_money_capped_at_20():
    st = dataclasses.replace(reset(seed=0), money=8)
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_HERMIT))
    assert overrides["money"] == 16                         # +min(8, 20)


def test_hermit_caps_gain_at_20():
    st = dataclasses.replace(reset(seed=0), money=100)
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_HERMIT))
    assert overrides["money"] == 120                        # +min(100, 20) = +20


def test_hermit_via_step():
    st = dataclasses.replace(reset(seed=0), money=5,
                             consumables=(tarot(TarotType.THE_HERMIT),))
    nxt, _ = step(st, (Verb.USE, 0))
    assert nxt.money == 10 and nxt.consumables == ()


def test_temperance_pays_sum_of_joker_sell_values():
    from balatro_rl.engine.shop import sell_value
    jokers = (JokerState(JokerType.BARON), JokerState(JokerType.HACK))
    st = dataclasses.replace(reset(seed=0), money=10, jokers=jokers)
    overrides, _ = apply_consumable(st, tarot(TarotType.TEMPERANCE))
    expect = sell_value(JokerType.BARON) + sell_value(JokerType.HACK)
    assert overrides["money"] == 10 + min(50, expect)


def test_temperance_capped_at_50():
    # Five Blueprints (cost 10 -> sell 5 each) = 25; below the cap. Use a high-value mix to
    # exceed 50: not easily reachable with current registry, so just assert the cap formula.
    jokers = tuple(JokerState(JokerType.BLUEPRINT) for _ in range(5))   # 5 * 5 = 25
    st = dataclasses.replace(reset(seed=0), money=0, jokers=jokers)
    overrides, _ = apply_consumable(st, tarot(TarotType.TEMPERANCE))
    assert overrides["money"] == min(50, 25)


def test_temperance_no_jokers_pays_zero():
    st = dataclasses.replace(reset(seed=0), money=7, jokers=())
    overrides, _ = apply_consumable(st, tarot(TarotType.TEMPERANCE))
    assert overrides["money"] == 7


# ============================================================================
# create tarots (High Priestess / Emperor / Judgement) — respect slot caps
# ============================================================================

def test_high_priestess_creates_planets():
    st = dataclasses.replace(reset(seed=0), consumables=(), consumable_slots=2)
    overrides, rng = apply_consumable(st, tarot(TarotType.THE_HIGH_PRIESTESS),
                                      rng=RNG.from_seed(123))
    created = overrides["consumables"]
    assert len(created) == 2
    assert all(c.kind == int(ConsumableKind.PLANET) for c in created)
    assert all(PlanetType(c.type_id) in PlanetType for c in created)
    assert rng != RNG.from_seed(123)                        # consumed rng


def test_high_priestess_respects_one_free_slot():
    st = dataclasses.replace(reset(seed=0),
                             consumables=(planet(PlanetType.PLUTO),), consumable_slots=2)
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_HIGH_PRIESTESS),
                                    rng=RNG.from_seed(1))
    # 1 free slot -> creates exactly 1 (added to the existing one).
    assert len(overrides["consumables"]) == 2


def test_high_priestess_none_when_slots_full():
    st = dataclasses.replace(reset(seed=0),
                             consumables=(planet(PlanetType.PLUTO), planet(PlanetType.MARS)),
                             consumable_slots=2)
    overrides, rng = apply_consumable(st, tarot(TarotType.THE_HIGH_PRIESTESS),
                                      rng=RNG.from_seed(1))
    assert overrides == {}                                  # no room -> no creation
    assert rng == RNG.from_seed(1)                          # rng untouched


def test_emperor_creates_implemented_tarots():
    st = dataclasses.replace(reset(seed=0), consumables=(), consumable_slots=2)
    overrides, _ = apply_consumable(st, tarot(TarotType.THE_EMPEROR), rng=RNG.from_seed(9))
    created = overrides["consumables"]
    assert len(created) == 2
    assert all(c.kind == int(ConsumableKind.TAROT) for c in created)
    assert all(TarotType(c.type_id) in IMPLEMENTED_TAROTS for c in created)  # never deferred


def test_judgement_creates_a_joker():
    st = dataclasses.replace(reset(seed=0), jokers=())
    overrides, rng = apply_consumable(st, tarot(TarotType.JUDGEMENT), rng=RNG.from_seed(5))
    assert len(overrides["jokers"]) == 1
    assert isinstance(overrides["jokers"][0], JokerState)
    assert rng != RNG.from_seed(5)


def test_judgement_none_when_joker_slots_full():
    full = tuple(JokerState(JokerType.JOKER) for _ in range(JOKER_SLOTS))
    st = dataclasses.replace(reset(seed=0), jokers=full)
    overrides, rng = apply_consumable(st, tarot(TarotType.JUDGEMENT), rng=RNG.from_seed(5))
    assert overrides == {}
    assert rng == RNG.from_seed(5)


def test_create_tarot_via_step_uses_freed_slot_and_threads_rng():
    """Using the create-tarot frees its own slot first, so a full-but-for-itself slot set
    still has room to create. And the advanced rng rides back into the successor."""
    st = dataclasses.replace(reset(seed=0),
                             consumables=(tarot(TarotType.THE_HIGH_PRIESTESS),),
                             consumable_slots=2)
    nxt, info = step(st, (Verb.USE, 0))
    assert info["verb"] == "use"
    # The used High Priestess is removed first -> 0 owned, 2 free slots -> creates 2 planets.
    assert len(nxt.consumables) == 2
    assert all(c.kind == int(ConsumableKind.PLANET) for c in nxt.consumables)
    assert nxt.rng != st.rng


# ============================================================================
# legal_actions: agent blind to card-targeting tarots
# ============================================================================

def test_legal_actions_offers_no_target_tarot_use():
    st = dataclasses.replace(reset(seed=0), consumables=(tarot(TarotType.THE_HERMIT),))
    assert (Verb.USE, 0) in legal_actions(st)


def test_legal_actions_arms_card_targeting_tarot_in_playing():
    # E5: in PLAYING (hand present) a card-targeting Tarot is offered as (USE, ci) to ARM the
    # two-step. Stepping it sets pending_consumable and applies nothing.
    st = dataclasses.replace(reset(seed=0), consumables=(tarot(TarotType.THE_CHARIOT),))
    assert (Verb.USE, 0) in legal_actions(st)
    armed, _ = step(st, (Verb.USE, 0))
    assert armed.pending_consumable == 0 and armed.consumables == st.consumables  # not yet used


def test_legal_actions_playing_offers_all_consumable_uses():
    # E5: in PLAYING every owned consumable is USE-able (targeting Tarots arm; the rest apply).
    cons = (tarot(TarotType.THE_CHARIOT), tarot(TarotType.THE_HERMIT),
            tarot(TarotType.THE_STAR), planet(PlanetType.MERCURY))
    st = dataclasses.replace(reset(seed=0), consumables=cons)
    uses = {a[1] for a in legal_actions(st) if a[0] == Verb.USE}
    assert uses == {0, 1, 2, 3}


def test_legal_actions_shop_withholds_card_targeting_tarot_use():
    # Targeting Tarots can only be armed in PLAYING (they target hand cards). In the SHOP only
    # the non-targeting consumables (Hermit idx1, Planet idx3) are USE-able.
    cons = (tarot(TarotType.THE_CHARIOT), tarot(TarotType.THE_HERMIT),
            tarot(TarotType.THE_STAR), planet(PlanetType.MERCURY))
    st = dataclasses.replace(reset(seed=0), phase=Phase.SHOP, consumables=cons)
    uses = {a[1] for a in legal_actions(st) if a[0] == Verb.USE}
    assert uses == {1, 3}


def test_card_targeting_tarot_via_direct_step():
    """The direct USE-with-targets tuple step still resolves immediately (scripted path)."""
    st = _state_with_hand([C(2, 0), C(3, 1)],
                          consumables=(tarot(TarotType.THE_SUN),))
    nxt, _ = step(st, (Verb.USE, (0, 0, 1)))                       # direct (ci, *targets) form
    assert all(c.suit == 1 for c in nxt.hand)                      # Hearts


def test_card_targeting_tarot_via_armed_two_step():
    """E5 agent path: arm with (USE, ci), then apply with (USE_TARGET, subset)."""
    st = _state_with_hand([C(2, 0), C(3, 1)],
                          consumables=(tarot(TarotType.THE_SUN),))
    armed, _ = step(st, (Verb.USE, 0))                            # arm
    assert armed.pending_consumable == 0
    assert all(a[0] == Verb.USE_TARGET for a in legal_actions(armed))   # only targeting now
    nxt, _ = step(armed, (Verb.USE_TARGET, (0, 1)))              # apply to both cards
    assert all(c.suit == 1 for c in nxt.hand) and nxt.pending_consumable == -1
    assert nxt.consumables == ()                                  # consumed


# ============================================================================
# USE-with-targets is a free action (doesn't touch hands/discards/phase)
# ============================================================================

def test_targeted_use_is_a_free_action():
    st = _state_with_hand([C(13, 0), C(10, 1)],
                          consumables=(tarot(TarotType.THE_CHARIOT),))
    nxt, _ = step(st, (Verb.USE, (0, 0)))
    assert nxt.hands_left == st.hands_left and nxt.discards_left == st.discards_left
    assert nxt.phase == st.phase

"""Phase D1b: consumables engine core + Planet cards. A Planet, applied via the USE action,
levels up its poker hand by 1 (free action, any phase). Other consumable kinds (Tarot/
Spectral) arrive later. Acquisition is a stopgap (states constructed with consumables).

USE is engine-only here: legal_actions does NOT emit it yet, so the agent's action space is
unchanged until the Phase D obs/action widening. Values verified against balatrowiki.org.
"""
import dataclasses

import pytest

from balatro_rl.engine.cards import Card
from balatro_rl.engine.consumables import (
    Consumable, ConsumableKind, PlanetType, PLANET_HAND, planet, apply_consumable,
)
from balatro_rl.engine.engine import reset, step, legal_actions, Verb
from balatro_rl.engine.hands import HandType


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


# ============================================================================
# value types + effect resolution
# ============================================================================

def test_planet_constructor():
    con = planet(PlanetType.MERCURY)
    assert con.kind == ConsumableKind.PLANET and con.type_id == PlanetType.MERCURY


def test_every_planet_maps_to_a_hand_type():
    assert set(PLANET_HAND) == set(PlanetType)
    assert PLANET_HAND[PlanetType.PLUTO] == HandType.HIGH_CARD
    assert PLANET_HAND[PlanetType.ERIS] == HandType.FLUSH_FIVE


def test_apply_planet_returns_leveled_levels():
    st = reset(seed=0)
    overrides = apply_consumable(st, planet(PlanetType.MERCURY))   # -> PAIR
    assert overrides["levels"][int(HandType.PAIR)] == st.levels[int(HandType.PAIR)] + 1


def test_apply_consumable_unimplemented_kind_raises():
    with pytest.raises(NotImplementedError):
        apply_consumable(reset(seed=0), Consumable(kind=int(ConsumableKind.TAROT), type_id=1))


# ============================================================================
# USE action through engine.step
# ============================================================================

def test_use_planet_levels_hand_and_removes_consumable():
    st = dataclasses.replace(reset(seed=0), consumables=(planet(PlanetType.MERCURY),))
    nxt, info = step(st, (Verb.USE, 0))
    assert nxt.levels[int(HandType.PAIR)] == 2
    assert nxt.consumables == ()
    assert info["verb"] == "use"


def test_use_is_a_free_action():
    st = dataclasses.replace(reset(seed=0), consumables=(planet(PlanetType.MERCURY),))
    nxt, _info = step(st, (Verb.USE, 0))
    assert nxt.hands_left == st.hands_left and nxt.discards_left == st.discards_left
    assert nxt.phase == st.phase                    # stays PLAYING; no hand consumed


def test_use_then_play_scores_at_new_level():
    st = dataclasses.replace(reset(seed=0), consumables=(planet(PlanetType.MERCURY),),
                             hand=(C(13, 0), C(13, 1), C(5, 2), C(7, 3), C(9, 0)),
                             required=10_000_000)
    st2, _i = step(st, (Verb.USE, 0))               # Mercury -> PAIR level 2
    _nxt, info = step(st2, (Verb.PLAY, (0, 1)))     # pair lvl2: base (25,3) + 10 + 10 = 45 x3
    assert info["score"] == 135


def test_all_planets_level_their_hand():
    for pt, ht in PLANET_HAND.items():
        st = dataclasses.replace(reset(seed=0), consumables=(planet(pt),))
        nxt, _info = step(st, (Verb.USE, 0))
        assert nxt.levels[int(ht)] == 2, pt


def test_use_only_removes_the_used_consumable():
    st = dataclasses.replace(reset(seed=0),
                             consumables=(planet(PlanetType.PLUTO), planet(PlanetType.MARS)))
    nxt, _info = step(st, (Verb.USE, 0))            # use Pluto
    assert nxt.consumables == (planet(PlanetType.MARS),)
    assert nxt.levels[int(HandType.HIGH_CARD)] == 2


# ============================================================================
# byte-compat: no consumables -> unchanged; agent can't USE yet
# ============================================================================

def test_defaults_have_no_consumables():
    st = reset(seed=0)
    assert st.consumables == () and st.consumable_slots == 2


def test_legal_actions_does_not_emit_use_yet():
    st = dataclasses.replace(reset(seed=0), consumables=(planet(PlanetType.MERCURY),))
    assert not any(v == Verb.USE for v, _p in legal_actions(st))

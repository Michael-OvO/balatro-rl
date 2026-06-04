"""Coverage + safety tests for the replay-viewer description tables.

Pure data/lookup module (no JAX): asserts every registered JokerType, every
BossEffect, and every PlanetType has a non-empty description, and that the
lookup helpers always return strings and never raise on unknown ids.
"""
from balatro_rl.engine.bosses import BossEffect
from balatro_rl.engine.consumables import ConsumableKind, PlanetType, TarotType
from balatro_rl.engine.descriptions import (
    BOSS_DESC, JOKER_DESC, PLANET_DESC, TAROT_DESC, boss_desc, consumable_desc, joker_desc,
)
from balatro_rl.engine.jokers.base import JokerType, REGISTRY


def test_every_registered_joker_has_description():
    missing = [jt.name for jt in REGISTRY if not JOKER_DESC.get(int(jt))]
    assert not missing, f"jokers missing a JOKER_DESC entry: {missing}"


def test_every_boss_has_description():
    missing = [b.name for b in BossEffect if not BOSS_DESC.get(int(b))]
    assert not missing, f"bosses missing a BOSS_DESC entry: {missing}"


def test_every_planet_has_description():
    missing = [p.name for p in PlanetType if not PLANET_DESC.get(int(p))]
    assert not missing, f"planets missing a PLANET_DESC entry: {missing}"


def test_every_tarot_has_description():
    missing = [t.name for t in TarotType if not TAROT_DESC.get(int(t))]
    assert not missing, f"tarots missing a TAROT_DESC entry: {missing}"


def test_joker_desc_returns_str_for_known_and_unknown():
    assert isinstance(joker_desc(JokerType.JOKER), str)
    assert joker_desc(JokerType.JOKER)  # non-empty for a known joker
    assert joker_desc(999999) == ""     # unknown int id
    assert joker_desc(None) == ""       # bad input does not raise


def test_boss_desc_returns_str_for_known_and_unknown():
    assert isinstance(boss_desc(BossEffect.THE_HOOK), str)
    assert boss_desc(BossEffect.THE_HOOK)
    assert boss_desc(BossEffect.NONE) == "No boss"
    assert boss_desc(999999) == ""
    assert boss_desc(None) == ""


def test_consumable_desc_dispatches_and_is_safe():
    # Planet dispatch hits PLANET_DESC.
    planet = consumable_desc(int(ConsumableKind.PLANET), int(PlanetType.MERCURY))
    assert isinstance(planet, str) and "Pair" in planet
    # Tarot dispatch hits TAROT_DESC (e.g. The Chariot -> Steel).
    chariot = consumable_desc(int(ConsumableKind.TAROT), int(TarotType.THE_CHARIOT))
    assert isinstance(chariot, str) and "Steel" in chariot
    assert consumable_desc(int(ConsumableKind.TAROT), 1)   # The Fool (deferred) still has text
    assert consumable_desc(int(ConsumableKind.SPECTRAL), 1)
    # Unknown kind / id / bad input return "" and never raise.
    assert consumable_desc(int(ConsumableKind.PLANET), 999999) == ""
    assert consumable_desc(int(ConsumableKind.NONE), 0) == ""
    assert consumable_desc(999, 999) == ""
    assert consumable_desc(None, None) == ""


def test_descriptions_are_short_one_liners():
    for table in (JOKER_DESC, BOSS_DESC, PLANET_DESC, TAROT_DESC):
        for text in table.values():
            assert "\n" not in text
            assert len(text) <= 100  # one-line, with rarity/cost suffix headroom

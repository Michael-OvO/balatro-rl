"""Phase D recording additions: jokers carry effect descriptions, and the per-step dict
gains boss / consumables / score_trace for the enriched viewer."""
import dataclasses

from balatro_rl.engine.bosses import BossEffect
from balatro_rl.engine.consumables import planet, PlanetType, Consumable, ConsumableKind
from balatro_rl.engine.engine import reset
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.viz.replay_data import _joker_d, _consum_d, _boss_d


def test_joker_d_includes_effect_description():
    d = _joker_d(JokerState(type=JokerType.GREEDY))
    assert d["name"] == "GREEDY" and d["desc"]            # non-empty effect text


def test_consum_d_planet_name_and_desc():
    d = _consum_d(planet(PlanetType.MERCURY))
    assert d["kind"] == int(ConsumableKind.PLANET) and d["name"] == "Mercury" and d["desc"]


def test_consum_d_handles_non_planet_kind():
    d = _consum_d(Consumable(kind=int(ConsumableKind.TAROT), type_id=3))
    assert isinstance(d["name"], str) and isinstance(d["desc"], str)


def test_boss_d_empty_without_boss():
    assert _boss_d(reset(seed=0)) == {}


def test_boss_d_has_name_and_desc():
    d = _boss_d(dataclasses.replace(reset(seed=0), boss=int(BossEffect.THE_FLINT)))
    assert d["id"] == int(BossEffect.THE_FLINT) and d["name"] == "The Flint" and d["desc"]

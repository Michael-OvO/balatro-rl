"""Training-env ACQUISITION for the Phase D retrain: so the agent actually ENCOUNTERS the
new content (enhanced cards, consumables) during training and can learn to exploit it.

The engine can already represent/score it all; this just seeds it into episodes. It's a
curriculum stopgap for the real acquisition sources (shop / packs), mirroring the plan's
"pre-seeded decks". Deterministic per episode seed (a derived RNG, independent of the game
stream), and a no-op when both rates are 0 -> byte-identical to the un-exposed game.
"""
from __future__ import annotations

from ..engine.cards import Enhancement
from ..engine.consumables import PlanetType, planet
from ..engine.rng import RNG

# Enhancements seeded onto deck cards (the scoring-relevant set; NONE excluded).
_ENH = [Enhancement.BONUS, Enhancement.MULT, Enhancement.WILD, Enhancement.GLASS,
        Enhancement.STEEL, Enhancement.GOLD, Enhancement.LUCKY, Enhancement.STONE]
_PLANETS = list(PlanetType)
_DECK = 52


def make_exposure(seed: int, enhance_rate: float = 0.0, grant_planets: int = 0):
    """Deterministic per-episode acquisition -> (card_mods | None, consumables tuple).

    enhance_rate: probability EACH of the 52 deck cards gets a random enhancement.
    grant_planets: number of random Planet consumables to start with (capped to the slot cap).
    Both 0 -> (None, ()) so engine.reset builds a plain deck and the game is byte-identical."""
    if enhance_rate <= 0 and grant_planets <= 0:
        return None, ()
    rng = RNG.from_seed(int(seed) + 999983)   # derived stream; never touches the game rng
    card_mods = None
    if enhance_rate > 0:
        card_mods = {}
        for i in range(_DECK):
            r, rng = rng.random()
            if r < enhance_rate:
                j, rng = rng.randint(0, len(_ENH) - 1)
                card_mods[i] = {"enhancement": int(_ENH[j])}
    consumables = ()
    if grant_planets > 0:
        cs = []
        for _ in range(min(grant_planets, 2)):   # consumable_slots cap
            j, rng = rng.randint(0, len(_PLANETS) - 1)
            cs.append(planet(_PLANETS[j]))
        consumables = tuple(cs)
    return card_mods, consumables

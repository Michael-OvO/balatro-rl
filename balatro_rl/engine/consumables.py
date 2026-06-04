"""Consumables (Phase D/E): Tarot / Planet / Spectral cards held in consumable slots and
applied via the USE action. This module is the consumable VALUE TYPES + effect resolution;
the engine owns the slots (GameState.consumables) and the USE step.

D1b implemented Planet cards (level up a poker hand). E2 adds Tarot cards (enhance/transform/
destroy selected hand cards, money effects, or create more consumables/jokers). Spectrals
arrive later. Acquisition is via the shop (Phase E1+). Values verified against balatrowiki.org.

USE-with-targets: card-targeting Tarots modify SELECTED hand cards. apply_consumable takes a
`targets` tuple of hand indices and an `rng` (for the create-* Tarots), and returns
`(overrides_dict, rng)`. The engine applies the overrides via dataclasses.replace and threads
the rng back. Card mutations persist to the master deck by identity (the hand card objects ARE
the master_deck objects, like Vampire/Midas), so we replace by id(original).
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

from .cards import Card, Enhancement, RANK_MAX, RANK_MIN
from .hands import HandType


class ConsumableKind(IntEnum):
    NONE = 0
    PLANET = 1
    TAROT = 2
    SPECTRAL = 3


class PlanetType(IntEnum):
    PLUTO = 1
    MERCURY = 2
    URANUS = 3
    VENUS = 4
    SATURN = 5
    JUPITER = 6
    EARTH = 7
    MARS = 8
    NEPTUNE = 9
    PLANET_X = 10
    CERES = 11
    ERIS = 12


class TarotType(IntEnum):
    """The 22 Tarot cards (wiki: balatrowiki.org/w/Tarot_Cards). Numbered by the
    Major Arcana order. THE_FOOL and THE_WHEEL_OF_FORTUNE are DEFERRED (need
    run-history / joker-edition systems not built yet); the other 20 are implemented."""
    THE_FOOL = 1                 # DEFERRED: re-creates last Tarot/Planet used this run
    THE_MAGICIAN = 2             # up to 2 -> Lucky
    THE_HIGH_PRIESTESS = 3       # create up to 2 random Planets
    THE_EMPRESS = 4             # up to 2 -> Mult
    THE_EMPEROR = 5              # create up to 2 random Tarots
    THE_HIEROPHANT = 6           # up to 2 -> Bonus
    THE_LOVERS = 7               # 1 -> Wild
    THE_CHARIOT = 8              # 1 -> Steel
    JUSTICE = 9                  # 1 -> Glass
    THE_HERMIT = 10              # double money (max +$20)
    THE_WHEEL_OF_FORTUNE = 11    # DEFERRED: 1-in-4 add edition to random joker
    STRENGTH = 12                # up to 2 -> rank +1
    THE_HANGED_MAN = 13          # destroy up to 2
    DEATH = 14                   # 2 cards -> left becomes a copy of right
    TEMPERANCE = 15              # money += min(50, total joker sell value)
    THE_DEVIL = 16               # 1 -> Gold
    THE_TOWER = 17               # 1 -> Stone
    THE_STAR = 18                # up to 3 -> Diamonds
    THE_MOON = 19                # up to 3 -> Clubs
    THE_SUN = 20                 # up to 3 -> Hearts
    JUDGEMENT = 21               # create 1 random Joker
    THE_WORLD = 22               # up to 3 -> Spades


# Which poker hand each Planet levels up (balatrowiki.org/w/Planet_Cards).
PLANET_HAND: dict[PlanetType, HandType] = {
    PlanetType.PLUTO: HandType.HIGH_CARD,
    PlanetType.MERCURY: HandType.PAIR,
    PlanetType.URANUS: HandType.TWO_PAIR,
    PlanetType.VENUS: HandType.THREE_OF_A_KIND,
    PlanetType.SATURN: HandType.STRAIGHT,
    PlanetType.JUPITER: HandType.FLUSH,
    PlanetType.EARTH: HandType.FULL_HOUSE,
    PlanetType.MARS: HandType.FOUR_OF_A_KIND,
    PlanetType.NEPTUNE: HandType.STRAIGHT_FLUSH,
    PlanetType.PLANET_X: HandType.FIVE_OF_A_KIND,
    PlanetType.CERES: HandType.FLUSH_HOUSE,
    PlanetType.ERIS: HandType.FLUSH_FIVE,
}

# DEFERRED Tarots: their effects need systems not built yet (The Fool: last-consumable-used
# run history; The Wheel of Fortune: joker editions). generate_offers excludes these.
DEFERRED_TAROTS: frozenset = frozenset({TarotType.THE_FOOL, TarotType.THE_WHEEL_OF_FORTUNE})
IMPLEMENTED_TAROTS: tuple = tuple(t for t in TarotType if t not in DEFERRED_TAROTS)

# Card-targeting Tarots: those that modify SELECTED hand cards (vs. money/create effects).
# `consumable_needs_target` returns True for these, and legal_actions withholds their USE
# (the agent stays blind to targeting until the Phase E5 obs/action widening).
_ENHANCE_TAROT: dict[TarotType, Enhancement] = {
    TarotType.THE_MAGICIAN: Enhancement.LUCKY,
    TarotType.THE_EMPRESS: Enhancement.MULT,
    TarotType.THE_HIEROPHANT: Enhancement.BONUS,
    TarotType.THE_LOVERS: Enhancement.WILD,
    TarotType.THE_CHARIOT: Enhancement.STEEL,
    TarotType.JUSTICE: Enhancement.GLASS,
    TarotType.THE_DEVIL: Enhancement.GOLD,
    TarotType.THE_TOWER: Enhancement.STONE,
}
_SUIT_TAROT: dict[TarotType, int] = {
    TarotType.THE_STAR: 3,    # Diamonds
    TarotType.THE_MOON: 2,    # Clubs
    TarotType.THE_SUN: 1,     # Hearts
    TarotType.THE_WORLD: 0,   # Spades
}
# Max cards each targeting Tarot affects (extra selected targets are ignored).
_TAROT_MAX_TARGETS: dict[TarotType, int] = {
    **{t: 2 for t in (TarotType.THE_MAGICIAN, TarotType.THE_EMPRESS,
                      TarotType.THE_HIEROPHANT, TarotType.STRENGTH,
                      TarotType.THE_HANGED_MAN)},
    **{t: 1 for t in (TarotType.THE_LOVERS, TarotType.THE_CHARIOT,
                      TarotType.JUSTICE, TarotType.THE_DEVIL, TarotType.THE_TOWER)},
    TarotType.DEATH: 2,
    **{t: 3 for t in _SUIT_TAROT},
}
CARD_TARGETING_TAROTS: frozenset = frozenset(_TAROT_MAX_TARGETS)


@dataclasses.dataclass(frozen=True, slots=True)
class Consumable:
    """An owned consumable: `kind` (ConsumableKind) + `type_id` (the specific card within
    that kind, e.g. a PlanetType or TarotType value). POD so it rides GameState cleanly."""
    kind: int
    type_id: int


def planet(ptype: PlanetType) -> Consumable:
    """Convenience constructor for a Planet consumable."""
    return Consumable(kind=int(ConsumableKind.PLANET), type_id=int(ptype))


def tarot(ttype: TarotType) -> Consumable:
    """Convenience constructor for a Tarot consumable."""
    return Consumable(kind=int(ConsumableKind.TAROT), type_id=int(ttype))


def consumable_needs_target(con: Consumable) -> bool:
    """True iff USEing this consumable requires selecting target HAND cards (the card-
    targeting Tarots). The engine uses this to withhold their USE from legal_actions
    (the agent is blind to targeting until E5); Planets and no-target Tarots return False."""
    return (con.kind == int(ConsumableKind.TAROT)
            and con.type_id in CARD_TARGETING_TAROTS)


def max_targets(con: Consumable) -> int:
    """How many hand cards a card-targeting Tarot selects (its USE_TARGET subset size cap);
    0 for non-targeting consumables. Used by the E5 pending two-step to bound the offered
    target subsets to the Tarot's reach (e.g. The Magician 2, The Lovers 1, the suit Tarots 3)."""
    if not consumable_needs_target(con):
        return 0
    return _TAROT_MAX_TARGETS[TarotType(con.type_id)]


def _bump_rank(rank: int) -> int:
    """Strength's rank +1 with Ace wrap: Ace(14) -> 2, King(13) -> Ace(14), else +1."""
    if rank >= RANK_MAX:        # Ace wraps to 2
        return RANK_MIN
    return rank + 1


def _apply_card_tarot(state, ttype: TarotType, targets) -> dict:
    """Resolve a card-targeting Tarot into {"hand", "master_deck"} overrides.

    Builds the modified cards, replacing them in `hand` by index AND in `master_deck`
    by id(original) (the hand card objects ARE the master_deck objects; see
    engine.reset/_advance_blind). Respects "up to N" by ignoring extra targets. The
    Hanged Man DESTROYS the targets (removed from both hand and master_deck)."""
    hand = list(state.hand)
    max_n = _TAROT_MAX_TARGETS[ttype]
    # Keep only valid, distinct, in-range targets, capped at the Tarot's max.
    seen: set = set()
    sel: list = []
    for i in targets:
        if 0 <= i < len(hand) and i not in seen:
            seen.add(i)
            sel.append(i)
        if len(sel) >= max_n:
            break

    if ttype == TarotType.THE_HANGED_MAN:
        destroy_ids = {id(hand[i]) for i in sel}
        new_hand = tuple(c for c in hand if id(c) not in destroy_ids)
        new_master = tuple(c for c in state.master_deck if id(c) not in destroy_ids)
        return {"hand": new_hand, "master_deck": new_master}

    # Build a map of original-id -> replacement Card for the selected targets.
    replace: dict = {}
    if ttype == TarotType.DEATH:
        # Exactly 2 -> [left, right]: left card becomes a full copy of the right card.
        if len(sel) == 2:
            left, right = sel
            r = hand[right]
            replace[id(hand[left])] = dataclasses.replace(r)  # fresh copy of right
    elif ttype in _ENHANCE_TAROT:
        enh = int(_ENHANCE_TAROT[ttype])
        for i in sel:
            replace[id(hand[i])] = dataclasses.replace(hand[i], enhancement=enh)
    elif ttype in _SUIT_TAROT:
        suit = _SUIT_TAROT[ttype]
        for i in sel:
            replace[id(hand[i])] = dataclasses.replace(hand[i], suit=suit)
    elif ttype == TarotType.STRENGTH:
        for i in sel:
            replace[id(hand[i])] = dataclasses.replace(hand[i], rank=_bump_rank(hand[i].rank))

    new_hand = tuple(replace.get(id(c), c) for c in hand)
    new_master = tuple(replace.get(id(c), c) for c in state.master_deck)
    return {"hand": new_hand, "master_deck": new_master}


def _apply_money_tarot(state, ttype: TarotType) -> dict:
    """The Hermit (double money, max +$20) and Temperance (joker sell value, max +$50)."""
    if ttype == TarotType.THE_HERMIT:
        return {"money": state.money + min(state.money, 20)}
    # Temperance: import sell_value lazily to avoid an engine<->shop import cycle.
    from .shop import sell_value
    total = sum(sell_value(js.type, js.sell_bonus) for js in state.jokers)
    return {"money": state.money + min(50, total)}


def _apply_create_tarot(state, ttype: TarotType, rng):
    """The High Priestess / The Emperor / Judgement: create random consumables or a joker,
    respecting slot caps. Consumes rng for each random pick. Returns (overrides, rng)."""
    if ttype == TarotType.JUDGEMENT:
        from .jokers.base import JokerState, REGISTRY, Rarity
        from .jokers import library as _library  # noqa: F401  populate REGISTRY
        from .engine import JOKER_SLOTS
        if len(state.jokers) >= JOKER_SLOTS:
            return {}, rng
        # Pool: any non-Legendary registered joker (deterministic registry order).
        pool = [t for t in REGISTRY if REGISTRY[t].rarity != Rarity.LEGENDARY]
        idx, rng = rng.randint(0, len(pool) - 1)
        return {"jokers": state.jokers + (JokerState(type=pool[idx]),)}, rng

    # The High Priestess -> Planets; The Emperor -> Tarots. Create up to 2, capped by slots.
    free = state.consumable_slots - len(state.consumables)
    if free <= 0:
        return {}, rng
    n = min(2, free)
    created: list = []
    if ttype == TarotType.THE_HIGH_PRIESTESS:
        choices = list(PlanetType)
        make = planet
    else:  # THE_EMPEROR
        choices = list(IMPLEMENTED_TAROTS)
        make = tarot
    for _ in range(n):
        idx, rng = rng.randint(0, len(choices) - 1)
        created.append(make(choices[idx]))
    return {"consumables": state.consumables + tuple(created)}, rng


def apply_consumable(state, con: Consumable, targets=(), rng=None) -> tuple[dict, object]:
    """Resolve a consumable's effect into GameState field overrides (applied by the engine's
    USE step, which also removes the card). Returns `(overrides_dict, rng)`.

    - PLANET: raises its hand type's level by 1 -> {"levels": ...} (rng passed through).
    - TAROT (card-targeting): modifies SELECTED hand cards (`targets` = hand indices) ->
      {"hand": ..., "master_deck": ...}. Persists mutations to master_deck by identity.
    - TAROT (Hermit/Temperance): {"money": ...}.
    - TAROT (High Priestess/Emperor/Judgement): {"consumables": ...} / {"jokers": ...},
      respecting slot caps and consuming rng for random picks.

    Raises for not-yet-implemented kinds (Spectral) and the two deferred Tarots so an
    unknown USE fails loudly."""
    if con.kind == ConsumableKind.PLANET:
        ht = int(PLANET_HAND[PlanetType(con.type_id)])
        levels = list(state.levels)
        levels[ht] += 1
        return {"levels": tuple(levels)}, rng
    if con.kind == ConsumableKind.TAROT:
        ttype = TarotType(con.type_id)
        if ttype in DEFERRED_TAROTS:
            raise NotImplementedError(f"Tarot {ttype.name} is deferred (needs systems not built)")
        if ttype in CARD_TARGETING_TAROTS:
            return _apply_card_tarot(state, ttype, targets), rng
        if ttype in (TarotType.THE_HERMIT, TarotType.TEMPERANCE):
            return _apply_money_tarot(state, ttype), rng
        if ttype in (TarotType.THE_HIGH_PRIESTESS, TarotType.THE_EMPEROR, TarotType.JUDGEMENT):
            return _apply_create_tarot(state, ttype, rng)
        raise NotImplementedError(f"Tarot {ttype.name} not implemented")
    raise NotImplementedError(f"consumable kind {con.kind} not implemented yet")

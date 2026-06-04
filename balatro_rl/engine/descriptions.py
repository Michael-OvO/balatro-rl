"""Human-readable effect descriptions for jokers, consumables, and bosses.

A pure data + lookup module for the replay viewer: it maps every registered
JokerType, every BossEffect, and every PlanetType to a short (~one line) effect
string so a viewer never has to reference the game rules. Joker effects are
transcribed faithfully from the `# wiki:` comments in jokers/library.py; boss
effects from bosses.py; Planet levels from consumables.py.

Imports only JAX-free engine value types (JokerType, BossEffect, PlanetType,
ConsumableKind), so importing this module is cheap and does not pull in JAX.
"""
from __future__ import annotations

from .jokers import library as _library  # noqa: F401  populates REGISTRY via @register
from .bosses import BossEffect
from .consumables import ConsumableKind, PlanetType, TarotType
from .jokers.base import JokerType, REGISTRY, Rarity

_RARITY_NAME: dict[int, str] = {
    int(Rarity.COMMON): "Common",
    int(Rarity.UNCOMMON): "Uncommon",
    int(Rarity.RARE): "Rare",
    int(Rarity.LEGENDARY): "Legendary",
}


# Short effect text per JokerType, transcribed from the `# wiki:` comments in
# jokers/library.py (the text after the em-dash). One line each, <= ~80 chars.
JOKER_DESC: dict[int, str] = {
    int(JokerType.JOKER): "+4 Mult",
    int(JokerType.CAVENDISH): "X3 Mult; 1 in 1000 self-destroy at end of round",
    int(JokerType.GREEDY): "+3 Mult per scored Diamond",
    int(JokerType.SCARY_FACE): "+30 Chips per scored face card",
    int(JokerType.PHOTOGRAPH): "X2 Mult on first scoring face card (re-applies on retrigger)",
    int(JokerType.BARON): "Each King held in hand gives X1.5 Mult",
    int(JokerType.HACK): "Retrigger each played 2, 3, 4, 5",
    int(JokerType.SPLASH): "Every played card scores",
    int(JokerType.PAREIDOLIA): "All cards are considered face cards",
    int(JokerType.RIDE_THE_BUS): "+1 Mult per consecutive hand with no scored face card",
    int(JokerType.BLUEPRINT): "Copies the Joker to its right",
    int(JokerType.GOLDEN_JOKER): "Earn +$4 at end of round",
    int(JokerType.EGG): "Gains +$3 sell value at end of round",
    int(JokerType.LUSTY): "+3 Mult per scored Heart",
    int(JokerType.WRATHFUL): "+3 Mult per scored Spade",
    int(JokerType.GLUTTONOUS): "+3 Mult per scored Club",
    int(JokerType.JOLLY): "+8 Mult if hand contains a Pair",
    int(JokerType.ZANY): "+12 Mult if hand contains Three of a Kind",
    int(JokerType.MAD): "+10 Mult if hand contains Two Pair",
    int(JokerType.CRAZY): "+12 Mult if hand contains a Straight",
    int(JokerType.DROLL): "+10 Mult if hand contains a Flush",
    int(JokerType.SLY): "+50 Chips if hand contains a Pair",
    int(JokerType.WILY): "+100 Chips if hand contains Three of a Kind",
    int(JokerType.CLEVER): "+80 Chips if hand contains Two Pair",
    int(JokerType.DEVIOUS): "+100 Chips if hand contains a Straight",
    int(JokerType.CRAFTY): "+80 Chips if hand contains a Flush",
    int(JokerType.HALF): "+20 Mult if played hand has 3 or fewer cards",
    int(JokerType.FIBONACCI): "Each scored A, 2, 3, 5, 8 gives +8 Mult",
    int(JokerType.EVEN_STEVEN): "Even-rank scored cards (2,4,6,8,10) give +4 Mult",
    int(JokerType.ODD_TODD): "Odd-rank scored cards (3,5,7,9,A) give +31 Chips",
    int(JokerType.SCHOLAR): "Scored Aces give +20 Chips and +4 Mult",
    int(JokerType.WALKIE_TALKIE): "Each scored 10 or 4 gives +10 Chips and +4 Mult",
    int(JokerType.SMILEY_FACE): "Scored face cards give +5 Mult",
    int(JokerType.SOCK_AND_BUSKIN): "Retrigger all played face cards",
    int(JokerType.GROS_MICHEL): "+15 Mult; 1 in 6 self-destroy at end of round",
    int(JokerType.RUNNER): "Gains +15 Chips per played hand that contains a Straight",
    int(JokerType.ICE_CREAM): "+100 Chips, -5 Chips per hand played",
    int(JokerType.ABSTRACT_JOKER): "+3 Mult per owned Joker",
    int(JokerType.JOKER_STENCIL): "X1 Mult per empty Joker slot (own slot counts as empty)",
    int(JokerType.BULL): "+2 Chips per $1 held",
    int(JokerType.BANNER): "+30 Chips per remaining discard",
    int(JokerType.MYSTIC_SUMMIT): "+15 Mult when 0 discards remaining",
    int(JokerType.BLUE_JOKER): "+2 Chips per remaining card in deck",
    int(JokerType.SQUARE_JOKER): "Gains +4 Chips if played hand has exactly 4 cards",
    int(JokerType.SPARE_TROUSERS): "Gains +2 Mult if played hand contains Two Pair",
    int(JokerType.WEE_JOKER): "Gains +8 Chips per scored 2",
    int(JokerType.POPCORN): "+20 Mult, loses -4 Mult per round played",
    int(JokerType.THE_DUO): "X2 Mult if hand contains a Pair",
    int(JokerType.THE_TRIO): "X3 Mult if hand contains Three of a Kind",
    int(JokerType.THE_FAMILY): "X4 Mult if hand contains Four of a Kind",
    int(JokerType.THE_ORDER): "X3 Mult if hand contains a Straight",
    int(JokerType.THE_TRIBE): "X2 Mult if hand contains a Flush",
    int(JokerType.ONYX_AGATE): "+7 Mult per scored Club",
    int(JokerType.ARROWHEAD): "+50 Chips per scored Spade",
    int(JokerType.SEEING_DOUBLE): "X2 Mult if scoring cards include a Club and another suit",
    int(JokerType.FLOWER_POT): "X3 Mult if scoring cards include all four suits",
    int(JokerType.BLACKBOARD): "X3 Mult if every held card is a Spade or Club",
    int(JokerType.TO_THE_MOON): "Extra $1 interest per $5 held at end of round",
    int(JokerType.DELAYED_GRATIFICATION): "$2 per remaining discard if no discards used this round",
    int(JokerType.FACELESS_JOKER): "Earn $5 if 3+ face cards discarded at once",
    int(JokerType.GREEN_JOKER): "+1 Mult per hand played, -1 Mult per discard",
    int(JokerType.RAMEN): "X2 Mult, -X0.01 per card discarded; eaten at 100 cards",
    int(JokerType.MISPRINT): "+0 to +23 Mult (random), changes every hand",
    int(JokerType.BLOODSTONE): "1 in 2 chance per scored Heart to give X1.5 Mult",
    int(JokerType.ANCIENT_JOKER): "X1.5 Mult per scored card of a suit (re-rolled each round)",
    int(JokerType.THE_IDOL): "X2 Mult per scored card of a rank+suit (re-rolled each round)",
    int(JokerType.MAIL_IN_REBATE): "$5 per discarded card of a rank (re-rolled each round)",
    int(JokerType.SUPERNOVA): "Adds times this hand type has been played this run to Mult",
    int(JokerType.CARD_SHARP): "X3 Mult if this hand type was already played this round",
    int(JokerType.OBELISK): "Gains X0.2 Mult per consecutive non-most-played hand",
    int(JokerType.STEEL_JOKER): "X Mult: +0.2 per Steel card in full deck",
    int(JokerType.STONE_JOKER): "+25 Chips per Stone card in full deck",
    int(JokerType.GOLDEN_TICKET): "Played Gold-enhancement card earns $4",
    int(JokerType.ROUGH_GEM): "Played Diamond earns $1",
    int(JokerType.BUSINESS_CARD): "Played face card has 1 in 2 chance to earn $2",
    int(JokerType.RESERVED_PARKING): "Each held face card has 1 in 2 chance to earn $1",
    int(JokerType.GLASS_JOKER): "Gains X0.75 Mult per Glass card destroyed",
    int(JokerType.LUCKY_CAT): "Gains X0.25 Mult per Lucky card trigger",
    int(JokerType.VAMPIRE): "X0.1 Mult per scored Enhanced card; removes the Enhancement",
    int(JokerType.MIDAS_MASK): "All scored face cards become Gold cards",
}


def _augment_with_meta() -> None:
    """Append `(Rarity, $cost)` to each joker description from its REGISTRY effect."""
    for jt, eff in REGISTRY.items():
        key = int(jt)
        base = JOKER_DESC.get(key)
        if not base:
            continue
        rarity = _RARITY_NAME.get(int(eff.rarity)) if eff.rarity is not None else None
        cost = eff.cost
        if rarity is not None and cost is not None:
            JOKER_DESC[key] = f"{base} ({rarity}, ${cost})"


_augment_with_meta()


# Boss effects, transcribed from bosses.py (29 entries incl. NONE).
BOSS_DESC: dict[int, str] = {
    int(BossEffect.NONE): "No boss",
    int(BossEffect.THE_HOOK): "Discards 2 random held cards after each play",
    int(BossEffect.THE_OX): "Sets money to $0 when you play your most-played hand",
    int(BossEffect.THE_HOUSE): "First hand is drawn face down",
    int(BossEffect.THE_WALL): "4x blind size (extra-large score requirement)",
    int(BossEffect.THE_WHEEL): "1 in 7 cards drawn face down",
    int(BossEffect.THE_ARM): "Decreases level of played poker hand by 1",
    int(BossEffect.THE_CLUB): "All Club cards are debuffed",
    int(BossEffect.THE_FISH): "Cards drawn face down after each hand played",
    int(BossEffect.THE_PSYCHIC): "Must play exactly 5 cards",
    int(BossEffect.THE_GOAD): "All Spade cards are debuffed",
    int(BossEffect.THE_WATER): "Start with 0 discards",
    int(BossEffect.THE_WINDOW): "All Diamond cards are debuffed",
    int(BossEffect.THE_MANACLE): "-1 hand size",
    int(BossEffect.THE_EYE): "No repeat hand types this round",
    int(BossEffect.THE_MOUTH): "Only one hand type can be played this round",
    int(BossEffect.THE_PLANT): "All face cards are debuffed",
    int(BossEffect.THE_SERPENT): "Always draws exactly 3 cards after each play/discard",
    int(BossEffect.THE_PILLAR): "Cards played last round are debuffed",
    int(BossEffect.THE_NEEDLE): "Only 1 hand",
    int(BossEffect.THE_HEAD): "All Heart cards are debuffed",
    int(BossEffect.THE_TOOTH): "Lose $1 per card played",
    int(BossEffect.THE_FLINT): "Halved base Chips and Mult of played hand",
    int(BossEffect.THE_MARK): "All face cards are drawn face down",
    int(BossEffect.AMBER_ACORN): "Flips and shuffles all Jokers face down",
    int(BossEffect.VERDANT_LEAF): "All cards debuffed until 1 Joker is sold",
    int(BossEffect.VIOLET_VESSEL): "6x blind size (very large score requirement)",
    int(BossEffect.CRIMSON_HEART): "One random Joker disabled each hand",
    int(BossEffect.CERULEAN_BELL): "Forces one card to always be selected",
}


# Planet cards: each levels up one poker hand by 1 (consumables.PLANET_HAND).
PLANET_DESC: dict[int, str] = {
    int(PlanetType.PLUTO): "+1 level to High Card",
    int(PlanetType.MERCURY): "+1 level to Pair",
    int(PlanetType.URANUS): "+1 level to Two Pair",
    int(PlanetType.VENUS): "+1 level to Three of a Kind",
    int(PlanetType.SATURN): "+1 level to Straight",
    int(PlanetType.JUPITER): "+1 level to Flush",
    int(PlanetType.EARTH): "+1 level to Full House",
    int(PlanetType.MARS): "+1 level to Four of a Kind",
    int(PlanetType.NEPTUNE): "+1 level to Straight Flush",
    int(PlanetType.PLANET_X): "+1 level to Five of a Kind",
    int(PlanetType.CERES): "+1 level to Flush House",
    int(PlanetType.ERIS): "+1 level to Flush Five",
}


# Tarot cards (wiki: balatrowiki.org/w/Tarot_Cards). One line each. The Fool and The
# Wheel of Fortune are DEFERRED in the engine (need run-history / joker-edition systems),
# so their text notes that.
TAROT_DESC: dict[int, str] = {
    int(TarotType.THE_FOOL): "Creates the last Tarot/Planet used this run (deferred)",
    int(TarotType.THE_MAGICIAN): "Enhances up to 2 selected cards to Lucky",
    int(TarotType.THE_HIGH_PRIESTESS): "Creates up to 2 random Planet cards",
    int(TarotType.THE_EMPRESS): "Enhances up to 2 selected cards to Mult",
    int(TarotType.THE_EMPEROR): "Creates up to 2 random Tarot cards",
    int(TarotType.THE_HIEROPHANT): "Enhances up to 2 selected cards to Bonus",
    int(TarotType.THE_LOVERS): "Enhances 1 selected card to Wild",
    int(TarotType.THE_CHARIOT): "Enhances 1 selected card to Steel",
    int(TarotType.JUSTICE): "Enhances 1 selected card to Glass",
    int(TarotType.THE_HERMIT): "Doubles money (max +$20)",
    int(TarotType.THE_WHEEL_OF_FORTUNE): "1 in 4 to add an edition to a random Joker (deferred)",
    int(TarotType.STRENGTH): "Increases rank of up to 2 selected cards by 1",
    int(TarotType.THE_HANGED_MAN): "Destroys up to 2 selected cards",
    int(TarotType.DEATH): "Converts the left selected card into the right",
    int(TarotType.TEMPERANCE): "Gives total sell value of all Jokers (max $50)",
    int(TarotType.THE_DEVIL): "Enhances 1 selected card to Gold",
    int(TarotType.THE_TOWER): "Enhances 1 selected card to Stone",
    int(TarotType.THE_STAR): "Converts up to 3 selected cards to Diamonds",
    int(TarotType.THE_MOON): "Converts up to 3 selected cards to Clubs",
    int(TarotType.THE_SUN): "Converts up to 3 selected cards to Hearts",
    int(TarotType.JUDGEMENT): "Creates a random Joker",
    int(TarotType.THE_WORLD): "Converts up to 3 selected cards to Spades",
}


def joker_desc(joker_type) -> str:
    """Effect description for a JokerType (or its int id). "" if unknown."""
    try:
        return JOKER_DESC.get(int(joker_type), "")
    except (TypeError, ValueError):
        return ""


def boss_desc(boss) -> str:
    """Effect description for a BossEffect (or its int id). "" if unknown."""
    try:
        return BOSS_DESC.get(int(boss), "")
    except (TypeError, ValueError):
        return ""


def consumable_desc(kind: int, type_id: int) -> str:
    """Effect description for a consumable, dispatched on its ConsumableKind.

    PLANET looks up PLANET_DESC, TAROT looks up TAROT_DESC; SPECTRAL returns a sensible
    placeholder (not yet implemented in the engine). Returns "" on an unknown kind/id
    rather than raising.
    """
    try:
        kind_i = int(kind)
        type_i = int(type_id)
    except (TypeError, ValueError):
        return ""
    if kind_i == int(ConsumableKind.PLANET):
        return PLANET_DESC.get(type_i, "")
    if kind_i == int(ConsumableKind.TAROT):
        return TAROT_DESC.get(type_i, "Tarot card")
    if kind_i == int(ConsumableKind.SPECTRAL):
        return "Spectral card"
    return ""


# Voucher effect text (summaries of the E4 wiki-verified effects in engine/vouchers.py), keyed
# by VoucherType name so it survives id renumbering. Used by the replay viewer.
_VOUCHER_DESC: dict[str, str] = {
    "OVERSTOCK": "+1 card slot in the shop (3 -> 4 offers).",
    "OVERSTOCK_PLUS": "+1 more card slot in the shop (4 -> 5 offers).",
    "CRYSTAL_BALL": "+1 consumable slot.",
    "GRABBER": "+1 hand per round.",
    "NACHO_TONG": "+1 more hand per round.",
    "WASTEFUL": "+1 discard per round.",
    "RECYCLOMANCY": "+1 more discard per round.",
    "PAINT_BRUSH": "+1 hand size.",
    "PALETTE": "+1 more hand size.",
    "ANTIMATTER": "+1 joker slot.",
    "SEED_MONEY": "Raise the interest cap to $10 (1 extra interest tier).",
    "MONEY_TREE": "Raise the interest cap to $20 (2 extra interest tiers).",
    "REROLL_SURPLUS": "Shop rerolls cost $2 less.",
    "REROLL_GLUT": "Shop rerolls cost $2 less again.",
    "TAROT_MERCHANT": "Tarot cards appear more often in the shop.",
    "TAROT_TYCOON": "Tarot cards appear much more often in the shop.",
    "PLANET_MERCHANT": "Planet cards appear more often in the shop.",
    "PLANET_TYCOON": "Planet cards appear much more often in the shop.",
    "BLANK": "Does nothing (unlocks the next voucher tier).",
}

_PACK_DESC: dict[str, str] = {
    "ARCANA": "Arcana Pack — choose Tarot card(s) from those shown.",
    "CELESTIAL": "Celestial Pack — choose Planet card(s) from those shown.",
    "BUFFOON": "Buffoon Pack — choose Joker(s) from those shown.",
    "STANDARD": "Standard Pack — choose playing card(s) from those shown.",
    "SPECTRAL": "Spectral Pack — choose Spectral card(s) from those shown.",
}

_PACK_SIZE_NAME: dict[int, str] = {1: "", 2: "Jumbo ", 3: "Mega "}


def voucher_desc(vtype) -> str:
    """Effect text for a VoucherType (by name; "" if unknown)."""
    from .vouchers import VoucherType
    try:
        return _VOUCHER_DESC.get(VoucherType(int(vtype)).name, "")
    except (TypeError, ValueError):
        return ""


def pack_desc(kind, size=None) -> str:
    """Effect text for a booster pack (kind + optional size prefix)."""
    from .packs import PackKind
    try:
        base = _PACK_DESC.get(PackKind(int(kind)).name, "Booster pack")
    except (TypeError, ValueError):
        return ""
    prefix = _PACK_SIZE_NAME.get(int(size), "") if size is not None else ""
    return prefix + base

"""Render a GameState as compact, readable text for the LLM agent.

Reuses engine/descriptions.py for effect text. Pure function: serialize_state(state)
-> str. Also exposes name helpers (joker/consumable/shop/pack/voucher) shared with
actions_text.py so the menu and the observation label entities identically (DRY).
"""
from __future__ import annotations

from ..engine import descriptions
from ..engine.bosses import BossEffect
from ..engine.cards import Edition, Enhancement, Seal, card_str
from ..engine.consumables import ConsumableKind, PlanetType, TarotType
from ..engine.jokers.base import JokerType
from ..engine.shop import SHOP_TO_CONSUMABLE_KIND, ShopKind
from ..engine.state import Phase
from ..engine.vouchers import VoucherType

_BLIND_NAME = {0: "Small", 1: "Big", 2: "Boss"}


def _pretty(name: str) -> str:
    return name.replace("_", " ").title()


# --- shared name helpers (also imported by actions_text.py) ---

def joker_name(js) -> str:
    return _pretty(JokerType(int(js.type)).name)


def _consumable_name_by_kind(kind: int, type_id: int) -> str:
    k = int(kind)
    if k == int(ConsumableKind.PLANET):
        return _pretty(PlanetType(int(type_id)).name)
    if k == int(ConsumableKind.TAROT):
        return _pretty(TarotType(int(type_id)).name)
    return "Spectral"


def consumable_name(con) -> str:
    return _consumable_name_by_kind(con.kind, con.type_id)


def shop_offer_name(offer) -> str:
    if int(offer.kind) == int(ShopKind.JOKER):
        return _pretty(JokerType(int(offer.type_id)).name)
    return _consumable_name_by_kind(SHOP_TO_CONSUMABLE_KIND[offer.kind], offer.type_id)


def pack_name(pack) -> str:
    return descriptions.pack_desc(pack.kind, pack.size)


def voucher_name(vid) -> str:
    return _pretty(VoucherType(int(vid)).name)


# --- blocks ---

def _card_mods(c) -> str:
    parts = [Enhancement(c.enhancement).name, Edition(c.edition).name, Seal(c.seal).name]
    extra = " ".join(_pretty(p) for p in parts if p != "NONE")
    return f" ({extra})" if extra else ""


def _header(state) -> str:
    boss = ""
    if state.boss:
        boss = f" | BOSS: {_pretty(BossEffect(state.boss).name)} ({descriptions.boss_desc(state.boss)})"
    blind = _BLIND_NAME.get(state.blind_index, str(state.blind_index))
    return (
        f"Phase: {_pretty(Phase(state.phase).name)} | Ante {state.ante}, {blind} blind{boss}\n"
        f"Score: {state.round_score}/{state.required} | Hands left: {state.hands_left} | "
        f"Discards left: {state.discards_left} | Money: ${state.money}"
    )


def _hand_block(state) -> str:
    lines = ["Hand:"]
    for i, c in enumerate(state.hand):
        lines.append(f"  [{i}] {card_str(c)}{_card_mods(c)}")
    return "\n".join(lines)


def serialize_state(state) -> str:
    blocks = [_header(state), _hand_block(state)]
    return "\n".join(b for b in blocks if b)

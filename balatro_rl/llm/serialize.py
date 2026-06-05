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


def _jokers_block(state) -> str:
    lines = ["Jokers (left to right):"]
    for i, js in enumerate(state.jokers):
        ctr = f" x{js.counter:g}" if js.counter else ""
        desc = descriptions.joker_desc(js.type)
        lines.append(f"  [{i}] {joker_name(js)}{ctr} - {desc}")
    return "\n".join(lines)


def _consumables_block(state) -> str:
    lines = [f"Consumables ({len(state.consumables)}/{state.consumable_slots}):"]
    for i, con in enumerate(state.consumables):
        desc = descriptions.consumable_desc(con.kind, con.type_id)
        lines.append(f"  [{i}] {consumable_name(con)} - {desc}")
    return "\n".join(lines)


def _shop_block(state) -> str:
    lines = ["Shop offers:"]
    for i, offer in enumerate(state.shop_offers):
        if int(offer.kind) == int(ShopKind.JOKER):
            desc = descriptions.joker_desc(offer.type_id)
        else:
            desc = descriptions.consumable_desc(SHOP_TO_CONSUMABLE_KIND[offer.kind], offer.type_id)
        lines.append(f"  [{i}] {shop_offer_name(offer)} (${offer.cost}) - {desc}")
    if state.pack_offers:
        lines.append("Packs:")
        for i, pack in enumerate(state.pack_offers):
            lines.append(f"  [{i}] {pack_name(pack)} (${pack.cost})")
    if state.voucher_offer:
        lines.append(f"Voucher: {voucher_name(state.voucher_offer)} - "
                     f"{descriptions.voucher_desc(state.voucher_offer)}")
    return "\n".join(lines)


def _pack_open_block(state) -> str:
    lines = [f"Opened pack - pick {state.pack_picks}:"]
    for i, item in enumerate(state.pack_open):
        lines.append(f"  [{i}] {_pack_item_name(item)}")
    return "\n".join(lines)


def _pack_item_name(item) -> str:
    from ..engine.packs import PackItemKind
    if int(item.kind) == int(PackItemKind.JOKER):
        return f"{joker_name(item.payload)} - {descriptions.joker_desc(item.payload.type)}"
    con = item.payload
    return f"{consumable_name(con)} - {descriptions.consumable_desc(con.kind, con.type_id)}"


def _vouchers_block(state) -> str:
    names = ", ".join(voucher_name(v) for v in state.vouchers)
    return f"Owned vouchers: {names}"


def serialize_state(state) -> str:
    blocks = [_header(state), _hand_block(state)]
    if state.jokers:
        blocks.append(_jokers_block(state))
    if state.consumables:
        blocks.append(_consumables_block(state))
    if state.phase == Phase.SHOP:
        blocks.append(_shop_block(state))
    if state.phase == Phase.OPEN_PACK:
        blocks.append(_pack_open_block(state))
    if state.vouchers:
        blocks.append(_vouchers_block(state))
    return "\n".join(b for b in blocks if b)

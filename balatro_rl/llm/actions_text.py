"""Legal-action menu + parser: the LLM-facing analog of envs/actions.py masking.

Discrete actions (shop / pack / voucher / use / sell / reroll / reorder / leave /
skip / pick) become a numbered menu -> the model returns {"choice": n}. Card-subset
actions (PLAY / DISCARD / USE_TARGET) are emitted as {"action": "play"|"discard"|
"target", "cards": [hand indices]}, validated against the legal set. Both forms are
checked against legal_mask so an illegal action can never be returned.
"""
from __future__ import annotations

import dataclasses

from ..engine.engine import Verb, legal_actions
from ..envs.actions import NUM_ACTIONS, encode_action, legal_mask
from .serialize import (consumable_name, joker_name, pack_name, shop_offer_name,
                        voucher_name)

_SUBSET_VERBS = {Verb.PLAY, Verb.DISCARD, Verb.USE_TARGET}


@dataclasses.dataclass(frozen=True)
class MenuOption:
    index: int
    action_id: int
    label: str


@dataclasses.dataclass(frozen=True)
class Menu:
    options: tuple[MenuOption, ...]
    can_play: bool
    can_discard: bool
    can_target: bool


def _label(verb, arg, state) -> str:
    if verb == Verb.BUY:
        o = state.shop_offers[arg]
        return f"Buy {shop_offer_name(o)} (${o.cost})"
    if verb == Verb.SELL:
        return f"Sell joker [{arg}] {joker_name(state.jokers[arg])}"
    if verb == Verb.REROLL:
        return "Reroll the shop"
    if verb == Verb.REORDER:
        return f"Move joker {arg[0]} -> position {arg[1]}"
    if verb == Verb.LEAVE_SHOP:
        return "Leave the shop"
    if verb == Verb.USE:
        return f"Use consumable [{arg}] {consumable_name(state.consumables[arg])}"
    if verb == Verb.OPEN:
        return f"Open pack {pack_name(state.pack_offers[arg])}"
    if verb == Verb.PICK:
        return f"Take pack item [{arg}]"
    if verb == Verb.SKIP_PACK:
        return "Skip the pack"
    if verb == Verb.BUY_VOUCHER:
        return f"Buy voucher {voucher_name(state.voucher_offer)}"
    return f"{verb.name} {arg}"


def build_menu(state) -> Menu:
    options: list[MenuOption] = []
    can_play = can_discard = can_target = False
    for verb, arg in legal_actions(state):
        if verb in _SUBSET_VERBS:
            can_play = can_play or verb == Verb.PLAY
            can_discard = can_discard or verb == Verb.DISCARD
            can_target = can_target or verb == Verb.USE_TARGET
            continue
        try:
            aid = encode_action(verb, arg)
        except (KeyError, ValueError):
            continue                            # slot beyond a flat-id cap -> not offered
        if aid >= NUM_ACTIONS:
            continue
        options.append(MenuOption(index=len(options), action_id=aid, label=_label(verb, arg, state)))
    return Menu(options=tuple(options), can_play=can_play, can_discard=can_discard,
                can_target=can_target)


def render_menu(menu: Menu) -> str:
    lines = ["Legal actions:"]
    for o in menu.options:
        lines.append(f"  [{o.index}] {o.label}")
    if menu.options:
        lines.append('To take one, reply with JSON: {"choice": <index>}')
    card_verbs = []
    if menu.can_play:
        card_verbs.append("play")
    if menu.can_discard:
        card_verbs.append("discard")
    if menu.can_target:
        card_verbs.append("target")
    if card_verbs:
        verbs = " / ".join(card_verbs)
        lines.append(f'To {verbs} cards, reply with JSON: '
                     f'{{"action": "{card_verbs[0]}", "cards": [hand indices]}}')
    return "\n".join(lines)

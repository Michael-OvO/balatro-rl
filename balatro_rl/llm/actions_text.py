"""Legal-action menu + parser: the LLM-facing analog of envs/actions.py masking.

Discrete actions (shop / pack / voucher / use / sell / reroll / reorder / leave /
skip / pick) become a numbered menu -> the model returns {"choice": n}. Card-subset
actions (PLAY / DISCARD / USE_TARGET) are emitted as {"action": "play"|"discard"|
"target", "cards": [hand indices]}, validated against the legal set. Both forms are
checked against legal_mask so an illegal action can never be returned.
"""
from __future__ import annotations

import dataclasses
import json

from ..engine.engine import Verb, legal_actions
from ..envs.actions import MAX_SELECT, NUM_ACTIONS, encode_action, legal_mask
from .serialize import (consumable_name, joker_name, pack_name, serialize_state,
                        shop_offer_name, voucher_name)

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
        verbs = " / ".join(f"{v} cards" for v in card_verbs)
        lines.append(f'To {verbs}, reply with JSON: '
                     f'{{"action": "{card_verbs[0]}", "cards": [1 to {MAX_SELECT} hand indices]}} '
                     f'(a played/discarded hand is at most {MAX_SELECT} cards).')
    return "\n".join(lines)


def observation_text(state, menu: "Menu | None" = None) -> str:
    """The canonical text observation = serialized state + the rendered legal-action menu.
    Single source of truth shared by LLMAgent (M1 eval) and BalatroTextEnv (M2 training) so the
    two CANNOT drift into different prompt formats. Pass a prebuilt menu to avoid recomputing it."""
    if menu is None:
        menu = build_menu(state)
    return serialize_state(state) + "\n\n" + render_menu(menu)


@dataclasses.dataclass(frozen=True)
class ParseResult:
    action_id: int | None = None
    error: str | None = None


def _extract_json(reply: str):
    """Find the model's action object in a free-text reply. Scans for every balanced
    {...} span and returns the LAST one that parses to a dict. Robust to stray braces
    in the model's pre-JSON reasoning (the system prompt invites it to "think briefly"
    first), which a naive first-'{'..last-'}' span would mis-capture into invalid JSON."""
    spans: list[str] = []
    depth = start = 0
    started = False
    for i, ch in enumerate(reply):
        if ch == "{":
            if not started:
                start, started = i, True
            depth += 1
        elif ch == "}" and started:
            depth -= 1
            if depth == 0:
                spans.append(reply[start:i + 1])
                started = False
    for span in reversed(spans):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


_CARD_VERB = {"play": Verb.PLAY, "discard": Verb.DISCARD, "target": Verb.USE_TARGET}


def parse_action(reply: str, state, menu: "Menu | None" = None, mask=None) -> ParseResult:
    """Parse an LLM reply into a validated flat action id. `menu` and `mask` may be passed
    in to avoid recomputing build_menu(state)/legal_mask(state) when the caller already has
    them (act() does); when omitted they are derived from `state` (back-compatible)."""
    obj = _extract_json(reply)
    if obj is None:
        return ParseResult(error="no JSON object found in reply")
    if mask is None:
        mask = legal_mask(state)
    if "choice" in obj:
        try:
            idx = int(obj["choice"])
        except (TypeError, ValueError):
            return ParseResult(error=f"choice is not an int: {obj['choice']!r}")
        options = (menu if menu is not None else build_menu(state)).options
        if not 0 <= idx < len(options):
            return ParseResult(error=f"choice {idx} out of range 0..{len(options) - 1}")
        aid = options[idx].action_id
        if not mask[aid]:
            return ParseResult(error=f"choice {idx} maps to an illegal action")
        return ParseResult(action_id=aid)
    action = str(obj.get("action", "")).lower()
    if action in _CARD_VERB:
        cards = obj.get("cards")
        if not isinstance(cards, list) or not cards:
            return ParseResult(error="'cards' must be a non-empty list of hand indices")
        try:
            subset = tuple(sorted({int(c) for c in cards}))
        except (TypeError, ValueError):
            return ParseResult(error=f"'cards' has non-integer entries: {cards!r}")
        try:
            aid = encode_action(_CARD_VERB[action], subset)
        except (KeyError, ValueError):
            return ParseResult(error=f"{action} {list(subset)} is not an encodable subset")
        if not mask[aid]:
            return ParseResult(error=f"{action} {list(subset)} is not legal right now")
        return ParseResult(action_id=aid)
    return ParseResult(error=f"unrecognized action object: {obj!r}")

# Agentic-RL M1 — Text-Env Boundary + Frozen Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the trainer-agnostic LLM-agent boundary (state→text serializer, legal-action menu + parser, multi-turn context manager, `LLMAgent`, frozen-model policy client, baseline eval entrypoint) so a frozen LLM can play full Balatro games on the existing eval/replay harness and produce a measured Ante-8 win rate vs the Random/Greedy/PPO baselines.

**Architecture:** A new `balatro_rl/llm/` package consumes the engine and `envs/` unchanged. `LLMAgent` implements the same `act(state, mask) -> action_id` interface as `RandomAgent`/`GreedyAgent`, so it drops into the existing `runner.run_episode` + eval harness with zero changes. The frozen baseline runs with no trainer dependency. (The verl/GRPO trainer and the gym-like `BalatroTextEnv` are M2 — a separate plan.)

**Tech Stack:** Python 3.11+, numpy, pytest. New optional dependency `openai>=1.0` (the `llm` extra) for talking to a vLLM OpenAI-compatible endpoint. Everything except the policy client is pure-Python and dependency-free, so unit tests need no network and no GPU.

**Spec:** `docs/specs/2026-06-04-agentic-rl-design.md` (covers M1 §3–§4, §6–§9). M2 (GRPO) and Endless are deferred to their own specs/plans.

---

## File structure

| File | Responsibility |
|---|---|
| `balatro_rl/llm/__init__.py` | Package marker; export `LLMAgent`. |
| `balatro_rl/llm/serialize.py` | `serialize_state(state) -> str` + shared name helpers (joker/consumable/shop/pack/voucher names). Pure function. |
| `balatro_rl/llm/actions_text.py` | `build_menu(state) -> Menu`, `render_menu(menu) -> str`, `parse_action(reply, state) -> ParseResult`. The action layer. |
| `balatro_rl/llm/context.py` | `ConversationContext`: rolling-window multi-turn message builder, bounded length. |
| `balatro_rl/llm/policy_client.py` | `Policy` protocol + `FrozenEndpointPolicy` (OpenAI-compatible client). |
| `balatro_rl/llm/agent.py` | `LLMAgent.act(state, mask) -> action_id` (multi-turn; uses serialize + menu + context + policy). |
| `balatro_rl/llm/baseline.py` | `run_baseline(...)` + CLI: play N seeds, report win rate / ante depth vs baselines, dump replays. |
| `tests/llm/...` | Unit + integration tests mirroring the package. |

Test fixtures construct states with `engine.reset(seed)` and `dataclasses.replace(...)` plus the real structs (`JokerState`, `Consumable`, `ShopItem`, `Pack`, `PackItem`). Tests assert structural properties (substrings, counts, legality) rather than brittle exact strings.

---

## Task 1: Package scaffold + `Policy` protocol + `llm` extra

**Files:**
- Create: `balatro_rl/llm/__init__.py`
- Create: `balatro_rl/llm/policy_client.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_policy_client.py`
- Modify: `pyproject.toml` (add `llm` optional-dependency group)

- [ ] **Step 1: Add the `llm` optional dependency**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a line after the `dev` entry:

```toml
llm = ["openai>=1.0"]
```

- [ ] **Step 2: Write the failing test**

Create `tests/llm/__init__.py` (empty), then `tests/llm/test_policy_client.py`:

```python
from balatro_rl.llm.policy_client import Policy


class _Echo:
    def generate(self, messages):
        return messages[-1]["content"]


def test_policy_protocol_is_satisfied_by_duck_typing():
    p: Policy = _Echo()
    out = p.generate([{"role": "user", "content": "hi"}])
    assert out == "hi"


def test_policy_is_runtime_checkable():
    assert isinstance(_Echo(), Policy)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_policy_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'balatro_rl.llm'`.

- [ ] **Step 4: Write minimal implementation**

Create `balatro_rl/llm/__init__.py`:

```python
"""Agentic-RL track: the trainer-agnostic LLM-agent boundary over the engine."""
from __future__ import annotations
```

Create `balatro_rl/llm/policy_client.py`:

```python
"""Policy backends for the LLM agent.

A Policy maps a chat-style message list to the assistant's text reply. The same
interface serves the frozen baseline (this file's FrozenEndpointPolicy, talking to
a vLLM OpenAI-compatible endpoint) and, later, the verl training rollout worker.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    def generate(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        Returns the assistant's text reply."""
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_policy_client.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml balatro_rl/llm/__init__.py balatro_rl/llm/policy_client.py tests/llm/__init__.py tests/llm/test_policy_client.py
git commit -m "E6 M1: llm package scaffold + Policy protocol + llm extra"
```

---

## Task 2: `serialize.py` — header, hand, and name helpers

**Files:**
- Create: `balatro_rl/llm/serialize.py`
- Create: `tests/llm/test_serialize.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_serialize.py`:

```python
import dataclasses

from balatro_rl.engine import engine
from balatro_rl.engine.cards import Card, Enhancement
from balatro_rl.llm.serialize import serialize_state


def test_header_has_ante_score_and_resources():
    state = engine.reset(0)
    text = serialize_state(state)
    assert "Ante 1" in text
    assert "Small blind" in text                 # blind_index 0
    assert f"{state.round_score}/{state.required}" in text
    assert "Hands left:" in text and "Discards left:" in text
    assert f"${state.money}" in text


def test_hand_block_lists_every_card_with_an_index():
    state = engine.reset(0)
    text = serialize_state(state)
    assert "Hand:" in text
    for i in range(len(state.hand)):
        assert f"[{i}]" in text


def test_enhanced_card_shows_its_modifier():
    state = engine.reset(0)
    hand = list(state.hand)
    hand[0] = Card(rank=hand[0].rank, suit=hand[0].suit, enhancement=int(Enhancement.STEEL))
    state = dataclasses.replace(state, hand=tuple(hand))
    text = serialize_state(state)
    assert "Steel" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_serialize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'balatro_rl.llm.serialize'`.

- [ ] **Step 3: Write minimal implementation**

Create `balatro_rl/llm/serialize.py`:

```python
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
        f"Phase: {Phase(state.phase).name} | Ante {state.ante}, {blind} blind{boss}\n"
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_serialize.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/llm/serialize.py tests/llm/test_serialize.py
git commit -m "E6 M1: serialize_state header/hand + shared name helpers"
```

---

## Task 3: `serialize.py` — jokers, consumables, shop, pack, vouchers

**Files:**
- Modify: `balatro_rl/llm/serialize.py`
- Modify: `tests/llm/test_serialize.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/llm/test_serialize.py`:

```python
from balatro_rl.engine.consumables import Consumable, ConsumableKind, PlanetType
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.engine.shop import ShopItem, ShopKind
from balatro_rl.engine.state import Phase


def test_jokers_block_names_and_describes_each_joker():
    state = engine.reset(0)
    state = dataclasses.replace(state, jokers=(JokerState(type=JokerType.BARON),))
    text = serialize_state(state)
    assert "Baron" in text
    assert "King" in text                     # from descriptions.joker_desc(BARON)


def test_consumables_block_lists_owned_consumables():
    state = engine.reset(0)
    con = Consumable(kind=int(ConsumableKind.PLANET), type_id=int(PlanetType.PLUTO))
    state = dataclasses.replace(state, consumables=(con,))
    text = serialize_state(state)
    assert "Pluto" in text


def test_shop_block_lists_offers_with_cost_when_in_shop():
    state = engine.reset(0)
    offer = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.JOKER), cost=2)
    state = dataclasses.replace(state, phase=Phase.SHOP, shop_offers=(offer,))
    text = serialize_state(state)
    assert "Shop" in text
    assert "Joker" in text
    assert "$2" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_serialize.py -v`
Expected: FAIL — the new tests fail (jokers/consumables/shop blocks not rendered yet).

- [ ] **Step 3: Write minimal implementation**

In `balatro_rl/llm/serialize.py`, add these block functions before `serialize_state`:

```python
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
```

Then replace the body of `serialize_state` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_serialize.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/llm/serialize.py tests/llm/test_serialize.py
git commit -m "E6 M1: serialize jokers/consumables/shop/pack/vouchers blocks"
```

---

## Task 4: `actions_text.py` — `build_menu` + `render_menu`

**Files:**
- Create: `balatro_rl/llm/actions_text.py`
- Create: `tests/llm/test_actions_text.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_actions_text.py`:

```python
import dataclasses

from balatro_rl.engine import engine
from balatro_rl.engine.state import Phase
from balatro_rl.engine.shop import ShopItem, ShopKind
from balatro_rl.engine.jokers.base import JokerType
from balatro_rl.llm.actions_text import build_menu, render_menu


def test_playing_state_offers_play_and_discard_not_discrete_subsets():
    state = engine.reset(0)                       # PLAYING, full hand
    menu = build_menu(state)
    assert menu.can_play is True
    assert menu.can_discard is True
    # PLAY/DISCARD are emitted as card-index calls, never enumerated as discrete options.
    assert all("play" not in o.label.lower() for o in menu.options)


def test_shop_state_lists_buy_and_leave_as_numbered_options():
    state = engine.reset(0)
    offer = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.JOKER), cost=2)
    state = dataclasses.replace(state, phase=Phase.SHOP, money=10, shop_offers=(offer,))
    menu = build_menu(state)
    labels = [o.label for o in menu.options]
    assert any("Buy" in l and "Joker" in l for l in labels)
    assert any("Leave" in l for l in labels)
    # options are contiguously indexed from 0
    assert [o.index for o in menu.options] == list(range(len(menu.options)))


def test_render_menu_includes_indices_and_card_instructions():
    state = engine.reset(0)
    text = render_menu(build_menu(state))
    assert "play" in text.lower() and "cards" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_actions_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'balatro_rl.llm.actions_text'`.

- [ ] **Step 3: Write minimal implementation**

Create `balatro_rl/llm/actions_text.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_actions_text.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/llm/actions_text.py tests/llm/test_actions_text.py
git commit -m "E6 M1: build_menu + render_menu (discrete menu + card-call instructions)"
```

---

## Task 5: `actions_text.py` — `parse_action`

**Files:**
- Modify: `balatro_rl/llm/actions_text.py`
- Modify: `tests/llm/test_actions_text.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/llm/test_actions_text.py`:

```python
from balatro_rl.envs.actions import decode, legal_mask
from balatro_rl.llm.actions_text import parse_action


def test_parse_menu_choice_returns_legal_action_id():
    state = engine.reset(0)
    offer = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.JOKER), cost=2)
    state = dataclasses.replace(state, phase=Phase.SHOP, money=10, shop_offers=(offer,))
    menu = build_menu(state)
    leave = next(o for o in menu.options if "Leave" in o.label)
    res = parse_action(f'{{"choice": {leave.index}}}', state)
    assert res.error is None
    assert res.action_id == leave.action_id
    assert legal_mask(state)[res.action_id]


def test_parse_play_cards_returns_legal_play_id():
    state = engine.reset(0)                              # PLAYING, hands_left > 0
    res = parse_action('{"action": "play", "cards": [0]}', state)
    assert res.error is None
    verb, arg = decode(res.action_id)
    assert verb.name == "PLAY" and arg == (0,)


def test_parse_rejects_illegal_choice_index():
    state = engine.reset(0)
    res = parse_action('{"choice": 999}', state)
    assert res.error is not None and res.action_id is None


def test_parse_rejects_unparseable_reply():
    state = engine.reset(0)
    res = parse_action("I think I will play the kings.", state)
    assert res.error is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_actions_text.py::test_parse_menu_choice_returns_legal_action_id -v`
Expected: FAIL with `ImportError: cannot import name 'parse_action'`.

- [ ] **Step 3: Write minimal implementation**

In `balatro_rl/llm/actions_text.py`, add `import json` at the top, and append:

```python
@dataclasses.dataclass(frozen=True)
class ParseResult:
    action_id: int | None = None
    error: str | None = None


def _extract_json(reply: str):
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


_CARD_VERB = {"play": Verb.PLAY, "discard": Verb.DISCARD, "target": Verb.USE_TARGET}


def parse_action(reply: str, state) -> ParseResult:
    obj = _extract_json(reply)
    if obj is None:
        return ParseResult(error="no JSON object found in reply")
    mask = legal_mask(state)
    if "choice" in obj:
        try:
            idx = int(obj["choice"])
        except (TypeError, ValueError):
            return ParseResult(error=f"choice is not an int: {obj['choice']!r}")
        options = build_menu(state).options
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_actions_text.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/llm/actions_text.py tests/llm/test_actions_text.py
git commit -m "E6 M1: parse_action (menu choice + card-call, legality-validated)"
```

---

## Task 6: `context.py` — bounded multi-turn `ConversationContext`

**Files:**
- Create: `balatro_rl/llm/context.py`
- Create: `tests/llm/test_context.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_context.py`:

```python
from balatro_rl.llm.context import ConversationContext


def test_first_render_has_system_then_user():
    ctx = ConversationContext(system_prompt="RULES", window_turns=3)
    msgs = ctx.render("OBS-0")
    assert msgs[0] == {"role": "system", "content": "RULES"}
    assert msgs[-1]["role"] == "user" and "OBS-0" in msgs[-1]["content"]


def test_window_bounds_number_of_turns():
    ctx = ConversationContext(system_prompt="RULES", window_turns=2)
    for t in range(5):
        ctx.render(f"OBS-{t}")
        ctx.update(assistant_reply=f"REPLY-{t}", observation="")
    msgs = ctx.render("OBS-5")
    # system + at most window_turns*(user,assistant) history + current user.
    assert len(msgs) <= 1 + 2 * 2 + 1
    # oldest turns are dropped; only recent replies survive verbatim.
    blob = "\n".join(m["content"] for m in msgs)
    assert "REPLY-0" not in blob
    assert "REPLY-4" in blob


def test_dropped_turns_are_summarized_not_silently_lost():
    ctx = ConversationContext(system_prompt="RULES", window_turns=1)
    for t in range(3):
        ctx.render(f"OBS-{t}")
        ctx.update(assistant_reply=f"REPLY-{t}", observation="")
    msgs = ctx.render("OBS-3")
    blob = "\n".join(m["content"] for m in msgs)
    assert "earlier turns" in blob.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'balatro_rl.llm.context'`.

- [ ] **Step 3: Write minimal implementation**

Create `balatro_rl/llm/context.py`:

```python
"""Bounded multi-turn conversation context for the LLM agent.

Keeps a static system prompt, a sliding window of the most recent (observation,
assistant-reply) turns, and a one-line deterministic note for the turns dropped out
of the window so context length stays bounded across a ~300-turn game. A model-written
plan slot and a richer summary are deferred (YAGNI) behind this same interface.
"""
from __future__ import annotations


class ConversationContext:
    def __init__(self, system_prompt: str, window_turns: int = 12):
        self._system = system_prompt
        self._window = window_turns
        self._turns: list[tuple[str, str]] = []   # (observation, assistant_reply)
        self._dropped = 0

    def render(self, observation: str) -> list[dict]:
        messages = [{"role": "system", "content": self._system}]
        if self._dropped:
            messages.append({"role": "system",
                             "content": f"(Summary: {self._dropped} earlier turns elided.)"})
        for obs, reply in self._turns[-self._window:]:
            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": observation})
        self._pending_obs = observation
        return messages

    def update(self, assistant_reply: str, observation: str = "") -> None:
        obs = getattr(self, "_pending_obs", observation)
        self._turns.append((obs, assistant_reply))
        if len(self._turns) > self._window:
            self._dropped = len(self._turns) - self._window
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_context.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/llm/context.py tests/llm/test_context.py
git commit -m "E6 M1: bounded multi-turn ConversationContext"
```

---

## Task 7: `agent.py` — `LLMAgent.act`

**Files:**
- Create: `balatro_rl/llm/agent.py`
- Create: `tests/llm/test_agent.py`
- Modify: `balatro_rl/llm/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_agent.py`:

```python
from balatro_rl.engine import engine
from balatro_rl.envs.actions import decode, legal_mask
from balatro_rl.llm.agent import LLMAgent


class _AlwaysPlayFirstCard:
    """A stand-in policy: always returns a legal play of the first hand card."""
    def __init__(self):
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return '{"action": "play", "cards": [0]}'


def test_act_returns_a_legal_action_id():
    policy = _AlwaysPlayFirstCard()
    agent = LLMAgent(policy=policy)
    state = engine.reset(0)
    aid = agent.act(state, legal_mask(state))
    assert legal_mask(state)[aid]
    verb, _ = decode(aid)
    assert verb.name == "PLAY"


def test_act_retries_then_falls_back_on_bad_replies():
    class _Garbage:
        def generate(self, messages):
            return "no json here"
    agent = LLMAgent(policy=_Garbage(), max_retries=2)
    state = engine.reset(0)
    aid = agent.act(state, legal_mask(state))   # must still return SOMETHING legal
    assert legal_mask(state)[aid]


def test_act_passes_serialized_state_to_the_policy():
    policy = _AlwaysPlayFirstCard()
    agent = LLMAgent(policy=policy)
    state = engine.reset(0)
    agent.act(state, legal_mask(state))
    last_user_msg = policy.calls[-1][-1]["content"]
    assert "Ante 1" in last_user_msg and "Legal actions" in last_user_msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'balatro_rl.llm.agent'`.

- [ ] **Step 3: Write minimal implementation**

Create `balatro_rl/llm/agent.py`:

```python
"""LLMAgent: an LLM policy with the same act(state, mask) -> action_id interface as
the baseline agents, so it drops into runner.run_episode + the eval harness unchanged.

Per act(): serialize the state, build + render the legal menu, ask the Policy, parse
the chosen action (validated legal), and feed the turn back into the multi-turn context.
On an unparseable / illegal reply it retries with the error, then falls back to a safe
legal action so a game never deadlocks during a baseline run.
"""
from __future__ import annotations

import numpy as np

from .actions_text import build_menu, parse_action, render_menu
from .context import ConversationContext
from .policy_client import Policy
from .serialize import serialize_state

SYSTEM_PROMPT = (
    "You are an expert Balatro player. Each turn you see the game state and a list of "
    "legal actions. Think briefly, then reply with exactly one JSON object choosing your "
    'action: {"choice": <index>} for a listed action, or {"action": "play"|"discard"|'
    '"target", "cards": [hand indices]} to play/discard/target cards. Your goal is to '
    "clear blinds and win Ante 8."
)


class LLMAgent:
    def __init__(self, policy: Policy, window_turns: int = 12, max_retries: int = 2,
                 system_prompt: str = SYSTEM_PROMPT):
        self._policy = policy
        self._ctx = ConversationContext(system_prompt=system_prompt, window_turns=window_turns)
        self._max_retries = max_retries

    def reset(self) -> None:
        self._ctx = ConversationContext(system_prompt=self._ctx._system,
                                        window_turns=self._ctx._window)

    def act(self, state, mask) -> int:
        observation = serialize_state(state) + "\n\n" + render_menu(build_menu(state))
        messages = self._ctx.render(observation)
        reply, chosen = "", None
        for attempt in range(self._max_retries + 1):
            reply = self._policy.generate(messages)
            res = parse_action(reply, state)
            if res.error is None:
                chosen = res.action_id
                break
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That action was invalid ({res.error}). "
                                             "Reply with one valid JSON action."},
            ]
        if chosen is None:
            chosen = int(np.flatnonzero(mask)[0])    # safe legal fallback
        self._ctx.update(assistant_reply=reply, observation="")
        return chosen
```

Then update `balatro_rl/llm/__init__.py` to export it:

```python
"""Agentic-RL track: the trainer-agnostic LLM-agent boundary over the engine."""
from __future__ import annotations

from .agent import LLMAgent

__all__ = ["LLMAgent"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_agent.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/llm/agent.py balatro_rl/llm/__init__.py tests/llm/test_agent.py
git commit -m "E6 M1: LLMAgent.act (serialize -> menu -> policy -> parse, retry+fallback)"
```

---

## Task 8: `policy_client.py` — `FrozenEndpointPolicy`

**Files:**
- Modify: `balatro_rl/llm/policy_client.py`
- Modify: `tests/llm/test_policy_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/llm/test_policy_client.py`:

```python
from balatro_rl.llm.policy_client import FrozenEndpointPolicy


class _FakeChat:
    def __init__(self, text):
        self._text = text
        self.seen = None

    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})

    def create(self, **kwargs):
        self.seen = kwargs
        return type("R", (), {"choices": [self._Msg(self._text)]})


class _FakeClient:
    def __init__(self, text):
        self.chat = type("C", (), {"completions": _FakeChat(text)})()


def test_frozen_policy_returns_reply_text_and_passes_messages():
    client = _FakeClient('{"choice": 0}')
    policy = FrozenEndpointPolicy(model="test-model", client=client, temperature=0.7)
    msgs = [{"role": "user", "content": "go"}]
    out = policy.generate(msgs)
    assert out == '{"choice": 0}'
    assert client.chat.completions.seen["model"] == "test-model"
    assert client.chat.completions.seen["messages"] == msgs
    assert client.chat.completions.seen["temperature"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_policy_client.py::test_frozen_policy_returns_reply_text_and_passes_messages -v`
Expected: FAIL with `ImportError: cannot import name 'FrozenEndpointPolicy'`.

- [ ] **Step 3: Write minimal implementation**

Append to `balatro_rl/llm/policy_client.py`:

```python
class FrozenEndpointPolicy:
    """Talks to an OpenAI-compatible chat endpoint (e.g. a vLLM server). Inject a
    pre-built `client` (tests pass a fake); otherwise an openai.OpenAI client is built
    from base_url/api_key. Used for the M1 frozen baseline and for eval."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str = "EMPTY",
                 temperature: float = 0.7, max_tokens: int = 512, client=None):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI            # optional dep: pip install -e '.[llm]'
            self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_policy_client.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/llm/policy_client.py tests/llm/test_policy_client.py
git commit -m "E6 M1: FrozenEndpointPolicy (OpenAI-compatible, injectable client)"
```

---

## Task 9: Integration — `LLMAgent` plays a full game via `run_episode`

**Files:**
- Create: `tests/llm/test_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_integration.py`:

```python
import re

from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.runner import Trajectory, run_episode, replay
from balatro_rl.llm.agent import LLMAgent


class ScriptedStubPolicy:
    """A no-LLM policy for tests: reads the rendered menu in the last user message and
    always returns a legal choice -- prefer a single-card play, else discard, else the
    first listed discrete option. Exercises the full serialize->menu->parse->step path."""

    def generate(self, messages: list[dict]) -> str:
        text = messages[-1]["content"]
        if 'play' in text and '"action"' in text and "play cards" in text.lower():
            return '{"action": "play", "cards": [0]}'
        if "discard cards" in text.lower():
            return '{"action": "discard", "cards": [0]}'
        m = re.search(r"\[(\d+)\]", text)         # first listed discrete option index
        idx = int(m.group(1)) if m else 0
        return f'{{"choice": {idx}}}'


def test_llm_agent_plays_a_full_game_and_produces_a_valid_trajectory():
    env = BalatroEnv(reward_name="shaped")
    agent = LLMAgent(policy=ScriptedStubPolicy())
    traj = run_episode(env, agent, seed=0)
    assert isinstance(traj, Trajectory)
    assert len(traj.actions) > 0
    # Engine determinism: the recorded (seed, actions) reconstructs to a terminal state.
    final = replay(traj)
    assert final.done or traj.truncated
```

NOTE: the stub keys off the menu wording from `render_menu` — the play/discard line reads
`To play / discard cards, reply with JSON:` (see Task 4), so `"play cards"`/`"discard cards"`
are matched case-insensitively. If `render_menu` wording changes, update this stub.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_integration.py -v`
Expected: FAIL — initially because the stub's `play cards` / `discard cards` match must align with the rendered menu text.

- [ ] **Step 3: Make the menu wording match the stub (minimal change)**

In `balatro_rl/llm/actions_text.py`, in `render_menu`, change the card-action line so the rendered text contains `play cards` / `discard cards` explicitly:

```python
    if card_verbs:
        verbs = " / ".join(f"{v} cards" for v in card_verbs)
        lines.append(f'To {verbs}, reply with JSON: '
                     f'{{"action": "{card_verbs[0]}", "cards": [hand indices]}}')
```

Re-run the Task 4 test to confirm no regression:
Run: `python -m pytest tests/llm/test_actions_text.py -v`
Expected: PASS (the `render_menu` test asserts `"play"` and `"cards"` substrings, still present).

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `python -m pytest tests/llm/test_integration.py -v`
Expected: PASS (1 passed) — the stubbed LLM agent plays a full game to a terminal/truncated state.

- [ ] **Step 5: Commit**

```bash
git add tests/llm/test_integration.py balatro_rl/llm/actions_text.py
git commit -m "E6 M1: integration - LLMAgent plays a full game via run_episode"
```

---

## Task 10: `baseline.py` — frozen-baseline eval entrypoint

**Files:**
- Create: `balatro_rl/llm/baseline.py`
- Create: `tests/llm/test_baseline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/llm/test_baseline.py`:

```python
from balatro_rl.llm.baseline import run_baseline, BaselineReport
from tests.llm.test_integration import ScriptedStubPolicy


def test_run_baseline_aggregates_win_rate_and_ante_depth():
    report = run_baseline(ScriptedStubPolicy(), seeds=[0, 1, 2], reward_name="shaped")
    assert isinstance(report, BaselineReport)
    assert report.games == 3
    assert 0.0 <= report.win_rate <= 1.0
    assert report.mean_final_ante >= 1.0
    assert len(report.trajectories) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'balatro_rl.llm.baseline'`.

- [ ] **Step 3: Write minimal implementation**

Create `balatro_rl/llm/baseline.py`:

```python
"""Frozen-baseline eval: run the LLM agent over a set of seeds on the existing env +
runner, and report Ante-8 win rate and ante-depth -- the M1 go/no-go gate. No training.

CLI:  python -m balatro_rl.llm.baseline --model <name> --base-url <url> --seeds 0-31
"""
from __future__ import annotations

import argparse
import dataclasses

from ..envs.balatro_env import BalatroEnv
from ..envs.runner import Trajectory, run_episode
from .agent import LLMAgent
from .policy_client import FrozenEndpointPolicy


@dataclasses.dataclass
class BaselineReport:
    games: int
    win_rate: float
    mean_final_ante: float
    trajectories: list[Trajectory]


def run_baseline(policy, seeds, reward_name: str = "shaped",
                 window_turns: int = 12) -> BaselineReport:
    trajectories: list[Trajectory] = []
    for seed in seeds:
        env = BalatroEnv(reward_name=reward_name)
        agent = LLMAgent(policy=policy, window_turns=window_turns)
        trajectories.append(run_episode(env, agent, seed=seed))
    wins = sum(1 for t in trajectories if t.won)
    n = len(trajectories)
    return BaselineReport(
        games=n,
        win_rate=wins / n if n else 0.0,
        mean_final_ante=sum(t.final_ante for t in trajectories) / n if n else 0.0,
        trajectories=trajectories,
    )


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen LLM baseline for Balatro (M1 gate).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--seeds", default="0-31")
    ap.add_argument("--reward-name", default="shaped")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    policy = FrozenEndpointPolicy(model=args.model, base_url=args.base_url,
                                  api_key=args.api_key, temperature=args.temperature)
    report = run_baseline(policy, seeds=_parse_seeds(args.seeds), reward_name=args.reward_name)
    print(f"games={report.games} win_rate={report.win_rate:.3f} "
          f"mean_final_ante={report.mean_final_ante:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_baseline.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full llm suite**

Run: `python -m pytest tests/llm -v`
Expected: PASS (all llm tests green).

- [ ] **Step 6: Commit**

```bash
git add balatro_rl/llm/baseline.py tests/llm/test_baseline.py
git commit -m "E6 M1: frozen-baseline eval entrypoint (run_baseline + CLI)"
```

---

## Manual smoke test (optional, needs a running endpoint)

Not part of CI (requires a GPU + a served model). After M1 lands, to get the real baseline number:

```bash
pip install -e '.[llm]'
# In another shell, serve a model (example): vllm serve <model> --port 8000
python -m balatro_rl.llm.baseline --model <model> --base-url http://localhost:8000/v1 --seeds 0-31
# prints e.g.: games=32 win_rate=0.063 mean_final_ante=3.41
```

This number, plus a few CoT replays, is the **M1 gate**: it tells us whether a frozen model has headroom worth training (M2).

---

## Self-review

**Spec coverage:**
- Spec §3 components — `serialize.py` (T2–T3), `actions_text.py` (T4–T5), `context.py` (T6), `policy_client.py` (T1, T8), `agent.py` (T7). ✅ `text_env.py`/`reward_adapter.py`/`verl_env.py`/`train.py` are explicitly M2 (deferred), consistent with the spec milestones.
- Spec §4 frozen-baseline data flow — `baseline.py` + integration test (T9–T10). ✅
- Spec §6 eval/observability — `run_baseline` reuses `run_episode` and reports win rate + ante depth; CoT is preserved in the message history (replay-viewer CoT attachment is a thin follow-on, noted in spec §6 as additive). ✅
- Spec §7 error handling — `parse_action` returns structured errors; `LLMAgent.act` retries then falls back to a safe legal action (T5, T7). ✅
- Spec §8 testing — unit (serialize/menu/parse/context/policy), stub-policy integration full game, golden corpus untouched (engine unchanged). ✅
- Spec §5 tunability — M1 reuses `make_reward(name)` via `BalatroEnv(reward_name=...)`; no reward/credit/algorithm framework built (correct per YAGNI). ✅

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test step shows full test code and the exact run command + expected result. ✅

**Type consistency:** `Menu`/`MenuOption`/`ParseResult` dataclasses defined in T4–T5 and used consistently in T7/T9; `Policy` (T1) implemented by `FrozenEndpointPolicy` (T8) and the test stubs; `serialize_state`, `build_menu`, `render_menu`, `parse_action`, `LLMAgent.act`, `run_baseline`/`BaselineReport` names are consistent across tasks. Name helpers (`joker_name`, `consumable_name`, `shop_offer_name`, `pack_name`, `voucher_name`) defined once in `serialize.py` (T2) and imported by `actions_text.py` (T4) — DRY. ✅

**Note for the implementer:** T9 couples the test stub to `render_menu`'s wording; T9 Step 3 aligns them deliberately. If you change `render_menu`, update `ScriptedStubPolicy` in `tests/llm/test_integration.py` (it is also imported by T10's test).

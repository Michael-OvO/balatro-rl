# Economy + Shop — Implementation Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.
>
> **ACCURACY RULE:** all economy/shop numbers come from `docs/reference/economy-shop.md` (verified against balatrowiki.org). Before implementing, an implementer/reviewer may re-`WebFetch` the cited page to confirm.

**Goal:** Build Balatro's run-economy loop — earn money on clearing a blind (blind reward + interest + leftover hands), then a **shop phase** (buy/sell/reroll/reorder jokers) before the next blind. This is the keystone that makes the game *playably* acquire jokers and gives the future RL agent its highest-skill decisions.

**Architecture:** Clearing a blind now routes **clear → win-check → cash-out → SHOP phase → (shop actions) → leave → next blind**, instead of advancing directly. New pure helpers (`economy.py` for payouts, `shop.py` for offer generation/pricing/reroll/sell), a `Phase.SHOP`, shop fields on `GameState`, new shop `Verb`s, and an `on_round_end` joker lifecycle hook (enabling round-end economy jokers + Cavendish's self-destroy). Engine stays pure/deterministic.

**Tech Stack:** Python ≥3.11, `dataclasses`, `pytest`. Builds on Plans 1–2 (`balatro_rl/engine/`, jokers package). Suit ints ♠0 ♥1 ♣2 ♦3.

**Scope (this plan):** earning + interest + leftover-hand money; shop phase with joker offers (rarity-weighted from the registry, fixed per-joker prices), buy/sell/reroll/reorder/leave; `on_round_end` lifecycle; 3 economy jokers (Golden Joker, Egg, Cavendish self-destroy). **Deferred:** vouchers, booster packs, tags, consumables in shop, edition surcharges, stake modifiers, the other economy jokers — later plans.

**Conventions:** repo `/Users/michael/Documents/GitHub/balatro-rl`; run `python3 -m pytest`; commit per task (no co-author trailers); feature branch off `master`. The `tests/engine/conftest.py` registry-isolation fixture from Plan 2 is in effect.

---

### Task 1: Rarity + per-joker cost; `sell_bonus` on JokerState

**Files:**
- Modify: `balatro_rl/engine/jokers/base.py`
- Modify: `balatro_rl/engine/jokers/library.py`
- Test: `tests/engine/jokers/test_rarity_cost.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_rarity_cost.py
from balatro_rl.engine.jokers.base import Rarity, JokerType, JokerState, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401


def test_rarity_enum():
    assert [r.name for r in Rarity] == ["COMMON", "UNCOMMON", "RARE", "LEGENDARY"]


def test_joker_state_has_sell_bonus_default():
    assert JokerState(type=JokerType.JOKER).sell_bonus == 0


def test_existing_jokers_declare_rarity_and_cost():
    # wiki: docs/reference/jokers.md
    expected = {
        JokerType.JOKER: (Rarity.COMMON, 2),
        JokerType.GREEDY: (Rarity.COMMON, 5),
        JokerType.SCARY_FACE: (Rarity.COMMON, 4),
        JokerType.PHOTOGRAPH: (Rarity.COMMON, 5),
        JokerType.CAVENDISH: (Rarity.COMMON, 4),
        JokerType.SPLASH: (Rarity.COMMON, 3),
        JokerType.RIDE_THE_BUS: (Rarity.COMMON, 6),
        JokerType.HACK: (Rarity.UNCOMMON, 6),
        JokerType.PAREIDOLIA: (Rarity.UNCOMMON, 5),
        JokerType.BARON: (Rarity.RARE, 8),
        JokerType.BLUEPRINT: (Rarity.RARE, 10),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_rarity_cost.py -v`
Expected: FAIL — `ImportError: cannot import name 'Rarity'`

- [ ] **Step 3: Write minimal implementation**

In `balatro_rl/engine/jokers/base.py`, add the `Rarity` enum (near `JokerType`):

```python
class Rarity(IntEnum):
    COMMON = 0
    UNCOMMON = 1
    RARE = 2
    LEGENDARY = 3
```

Add `sell_bonus` to `JokerState` (after `counter`):

```python
    sell_bonus: int = 0   # extra sell value beyond floor(cost/2), e.g. from Egg
```

Add `rarity` and `cost` class attributes to `JokerEffect` (alongside `copyable`):

```python
    rarity: "Rarity" = None      # set by each joker; Rarity enum
    cost: int = 4                # base shop buy price ($); set by each joker
```

In `balatro_rl/engine/jokers/library.py`, import `Rarity` and add `rarity`/`cost` to each registered class. Add these two class attributes to each existing joker:

```python
# _Joker:        rarity = Rarity.COMMON;   cost = 2
# _Cavendish:    rarity = Rarity.COMMON;   cost = 4
# _Greedy:       rarity = Rarity.COMMON;   cost = 5
# _ScaryFace:    rarity = Rarity.COMMON;   cost = 4
# _Photograph:   rarity = Rarity.COMMON;   cost = 5
# _Baron:        rarity = Rarity.RARE;     cost = 8
# _Hack:         rarity = Rarity.UNCOMMON; cost = 6
# _Splash:       rarity = Rarity.COMMON;   cost = 3
# _Pareidolia:   rarity = Rarity.UNCOMMON; cost = 5
# _RideTheBus:   rarity = Rarity.COMMON;   cost = 6
# _Blueprint:    rarity = Rarity.RARE;     cost = 10
```

Update the import line in `library.py` to include `Rarity`:
```python
from .base import Effect, JokerEffect, JokerState, JokerType, Rarity, RuleFlags, register
```
Then add the two class attributes to each joker class, e.g.:
```python
@register(JokerType.JOKER)
class _Joker(JokerEffect):  # wiki: /w/Joker  — +4 Mult
    rarity = Rarity.COMMON
    cost = 2
    def independent(self, ctx, js):
        return Effect(mult=4)
```
(Apply the corresponding `rarity`/`cost` to all 11 classes per the table above.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_rarity_cost.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/base.py balatro_rl/engine/jokers/library.py tests/engine/jokers/test_rarity_cost.py
git commit -m "feat(jokers): add Rarity enum, per-joker cost, and JokerState.sell_bonus"
```

---

### Task 2: Economy module (blind reward + interest)

**Files:**
- Create: `balatro_rl/engine/economy.py`
- Test: `tests/engine/test_economy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_economy.py
from balatro_rl.engine.economy import blind_reward, interest, BLIND_REWARD


def test_blind_rewards():  # wiki: docs/reference/economy-shop.md §2
    assert blind_reward(0) == 3   # Small
    assert blind_reward(1) == 4   # Big
    assert blind_reward(2) == 5   # Boss
    assert BLIND_REWARD == (3, 4, 5)


def test_interest_rate_and_cap():  # +$1 per $5, cap $5 at $25
    assert interest(0) == 0
    assert interest(4) == 0
    assert interest(5) == 1
    assert interest(24) == 4
    assert interest(25) == 5
    assert interest(30) == 5
    assert interest(100) == 5


def test_interest_custom_cap():
    assert interest(100, cap=10) == 10   # Seed Money
    assert interest(40, cap=10) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_economy.py -v`
Expected: FAIL — `ModuleNotFoundError: ...economy`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/economy.py
"""Run economy: blind rewards and interest. Verified values in
docs/reference/economy-shop.md (balatrowiki.org). Stake/deck modifiers and
voucher cap-raises are later plans (default cap = 5).
"""
from __future__ import annotations

BLIND_REWARD: tuple[int, int, int] = (3, 4, 5)  # Small, Big, Boss
INTEREST_PER = 5      # +$1 per $5 held
INTEREST_CAP = 5      # default cap ($5, reached at $25)
MONEY_PER_UNUSED_HAND = 1  # standard decks


def blind_reward(blind_index: int) -> int:
    return BLIND_REWARD[blind_index]


def interest(money: int, cap: int = INTEREST_CAP) -> int:
    if money <= 0:
        return 0
    return min(money // INTEREST_PER, cap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/test_economy.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/economy.py tests/engine/test_economy.py
git commit -m "feat(engine): economy module (blind reward + interest)"
```

---

### Task 3: `on_round_end` lifecycle hook

**Files:**
- Modify: `balatro_rl/engine/jokers/base.py`
- Test: `tests/engine/jokers/test_round_end_hook.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_round_end_hook.py
from balatro_rl.engine.jokers.base import JokerEffect, JokerType, JokerState
from balatro_rl.engine.rng import RNG


def test_default_on_round_end_is_noop():
    eff = JokerEffect()
    js = JokerState(type=JokerType.JOKER)
    rng = RNG.from_seed(1)
    js2, money_delta, destroy, rng2 = eff.on_round_end(None, js, rng)
    assert js2 is js
    assert money_delta == 0
    assert destroy is False
    assert rng2 is rng
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_round_end_hook.py -v`
Expected: FAIL — `JokerEffect` has no `on_round_end`

- [ ] **Step 3: Write minimal implementation** (append a method to `JokerEffect` in `base.py`)

```python
    def on_round_end(self, state, js: "JokerState", rng):
        """End-of-round (cash-out) lifecycle. Returns
        (updated JokerState, money_delta:int, destroy:bool, rng).
        rng is threaded for probabilistic effects (e.g. self-destroy)."""
        return js, 0, False, rng
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_round_end_hook.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/base.py tests/engine/jokers/test_round_end_hook.py
git commit -m "feat(jokers): on_round_end lifecycle hook (money/destroy at cash-out)"
```

---

### Task 4: Shop module (generation, pricing, reroll, sell)

**Files:**
- Create: `balatro_rl/engine/shop.py`
- Test: `tests/engine/test_shop.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_shop.py
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.jokers.base import JokerType, JokerState, Rarity, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401
from balatro_rl.engine.shop import generate_offers, joker_cost, reroll_cost, sell_value


def test_reroll_cost_scaling():  # base 5, +1 each, per economy-shop.md §6
    assert reroll_cost(0) == 5
    assert reroll_cost(1) == 6
    assert reroll_cost(3) == 8


def test_sell_value_floor_min_1():
    assert sell_value(JokerType.JOKER) == 1      # cost 2 -> floor(2/2)=1
    assert sell_value(JokerType.BARON) == 4      # cost 8 -> 4
    assert sell_value(JokerType.HACK) == 3       # cost 6 -> 3
    assert sell_value(JokerType.JOKER, sell_bonus=3) == 4   # +Egg bonus


def test_joker_cost_reads_registry():
    assert joker_cost(JokerType.BLUEPRINT) == 10
    assert joker_cost(JokerType.JOKER) == 2


def test_generate_offers_deterministic_and_valid():
    offers_a, _ = generate_offers(RNG.from_seed(42), 2)
    offers_b, _ = generate_offers(RNG.from_seed(42), 2)
    assert offers_a == offers_b          # deterministic per seed
    assert len(offers_a) == 2
    for js in offers_a:
        assert isinstance(js, JokerState)
        assert REGISTRY[js.type].rarity != Rarity.LEGENDARY   # never legendary in shop


def test_generate_offers_advances_rng():
    rng = RNG.from_seed(7)
    _, rng2 = generate_offers(rng, 2)
    assert rng2 != rng
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_shop.py -v`
Expected: FAIL — `ModuleNotFoundError: ...shop`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/shop.py
"""Shop: offer generation, pricing, reroll, sell. Verified values in
docs/reference/economy-shop.md. Tier-3 scope offers only Jokers in card slots
(Tarot/Planet/Spectral and packs/vouchers are later plans); per-joker prices are
the deterministic base costs from the registry (edition surcharges deferred).
"""
from __future__ import annotations

from .jokers import library as _library  # noqa: F401  (ensures REGISTRY is populated)
from .jokers.base import REGISTRY, JokerState, JokerType, Rarity

CARD_SLOTS = 2
REROLL_BASE = 5

# Joker-rarity distribution once a Joker rolls (Common 70 / Uncommon 25 / Rare 5).
_RARITY_THRESHOLDS = ((Rarity.RARE, 0.05), (Rarity.UNCOMMON, 0.30))  # else COMMON


def joker_cost(jtype: JokerType) -> int:
    return REGISTRY[jtype].cost


def reroll_cost(rerolls_done: int, base: int = REROLL_BASE) -> int:
    return max(0, base + rerolls_done)


def sell_value(jtype: JokerType, sell_bonus: int = 0) -> int:
    return max(1, REGISTRY[jtype].cost // 2) + sell_bonus


def _roll_rarity(rng):
    r, rng = rng.random()
    if r < 0.05:
        return Rarity.RARE, rng
    if r < 0.30:
        return Rarity.UNCOMMON, rng
    return Rarity.COMMON, rng


def _pool(rarity: Rarity) -> list[JokerType]:
    # Deterministic order (registry insertion order); never Legendary in shop.
    return [t for t in REGISTRY if REGISTRY[t].rarity == rarity and rarity != Rarity.LEGENDARY]


def generate_offers(rng, n: int = CARD_SLOTS):
    """Generate n shop joker offers, rarity-weighted, from the registry. Returns
    (tuple[JokerState], rng). Falls back to Common if a rarity pool is empty."""
    offers = []
    for _ in range(n):
        rarity, rng = _roll_rarity(rng)
        pool = _pool(rarity) or _pool(Rarity.COMMON)
        idx, rng = rng.randint(0, len(pool) - 1)
        offers.append(JokerState(type=pool[idx]))
    return tuple(offers), rng
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/test_shop.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/shop.py tests/engine/test_shop.py
git commit -m "feat(engine): shop module (offer generation, pricing, reroll, sell)"
```

---

### Task 5: GameState shop fields + `Phase.SHOP`

**Files:**
- Modify: `balatro_rl/engine/state.py`
- Test: `tests/engine/test_state_shop.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_state_shop.py
import dataclasses
from balatro_rl.engine.state import GameState, Phase
from balatro_rl.engine.engine import reset


def test_phase_has_shop():
    assert Phase.SHOP.name == "SHOP"


def test_reset_shop_fields_default_empty():
    s = reset(seed=1)
    assert s.shop_offers == ()
    assert s.rerolls_done == 0
    assert s.phase == Phase.PLAYING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_state_shop.py -v`
Expected: FAIL — `Phase` has no `SHOP` / `GameState` has no `shop_offers`

- [ ] **Step 3: Write minimal implementation**

In `balatro_rl/engine/state.py`, add `SHOP` to `Phase`:

```python
class Phase(IntEnum):
    PLAYING = 0
    WON = 1
    LOST = 2
    SHOP = 3
```

Add shop fields at the END of `GameState` (after `jokers`, to satisfy default-field ordering):

```python
    shop_offers: tuple = ()   # tuple[JokerState, ...] offered in the shop
    rerolls_done: int = 0      # rerolls used in the current shop (for reroll cost)
```

In `balatro_rl/engine/engine.py` `reset(...)`, add to the `GameState(...)` constructor (they have defaults, so explicit is optional but keep clarity):
```python
        shop_offers=(), rerolls_done=0,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/test_state_shop.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/state.py balatro_rl/engine/engine.py tests/engine/test_state_shop.py
git commit -m "feat(engine): add Phase.SHOP and shop fields to GameState"
```

---

### Task 6: Shop verbs + cash-out + shop flow in `step`

**Files:**
- Modify: `balatro_rl/engine/engine.py`
- Modify: `tests/engine/test_engine.py` (update the clear-advances test for the new flow)
- Test: `tests/engine/test_engine_shop.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_engine_shop.py
import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, legal_actions, step
from balatro_rl.engine.state import Phase
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def _clearable(seed=1, **over):
    """A state one play away from clearing, with a known big hand."""
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    return dataclasses.replace(s, hand=hand, required=10, **over)


def test_clearing_a_blind_enters_shop_and_pays_out():
    s = _clearable(money=10, hands_left=3)   # Small blind, $10 held, 3 hands left
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))  # four-of-a-kind kings, clears
    assert s2.phase == Phase.SHOP
    # cash-out = reward(small $3) + interest(10 -> $2) + leftover hands (2 left after play) $2 = $7
    assert s2.money == 10 + 3 + 2 + 2
    assert len(s2.shop_offers) == 2
    assert s2.blind_index == 0   # NOT advanced yet (advance happens on leave)


def test_shop_buy_adds_joker_and_spends():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))  # enter shop
    assert s2.phase == Phase.SHOP
    offer0 = s2.shop_offers[0]
    cost = __import__("balatro_rl.engine.shop", fromlist=["joker_cost"]).joker_cost(offer0.type)
    s3, info = step(s2, (Verb.BUY, 0))
    assert info["verb"] == "buy"
    assert s3.money == s2.money - cost
    assert s3.jokers[-1].type == offer0.type
    assert len(s3.shop_offers) == 1


def test_shop_sell_returns_money_and_frees_slot():
    s = _clearable(money=100, hands_left=1, jokers=(JokerState(JokerType.BARON),))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    s3, info = step(s2, (Verb.SELL, 0))   # sell Baron (cost 8 -> sell 4)
    assert info["verb"] == "sell" and info["value"] == 4
    assert s3.money == s2.money + 4
    assert s3.jokers == ()


def test_shop_reroll_costs_and_replaces_offers():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    before = s2.money
    s3, info = step(s2, (Verb.REROLL, 0))
    assert info["verb"] == "reroll" and info["cost"] == 5
    assert s3.money == before - 5
    assert s3.rerolls_done == 1
    s4, info2 = step(s3, (Verb.REROLL, 0))
    assert info2["cost"] == 6   # +1 each reroll


def test_shop_reorder_jokers():
    s = _clearable(money=10, hands_left=1,
                   jokers=(JokerState(JokerType.JOKER), JokerState(JokerType.BARON)))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    s3, info = step(s2, (Verb.REORDER, (0, 1)))   # move slot0 -> slot1
    assert info["verb"] == "reorder"
    assert [j.type for j in s3.jokers] == [JokerType.BARON, JokerType.JOKER]


def test_leave_shop_advances_to_next_blind():
    s = _clearable(money=10, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))   # cleared Small -> shop
    s3, info = step(s2, (Verb.LEAVE_SHOP, 0))
    assert s3.phase == Phase.PLAYING
    assert s3.blind_index == 1       # advanced to Big
    assert s3.round_score == 0 and s3.hands_left == 4 and len(s3.hand) == 8
    assert s3.shop_offers == () and s3.rerolls_done == 0


def test_clearing_ante8_boss_wins_no_shop():
    s = _clearable(ante=8, blind_index=2, required=10, hands_left=1)
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert s2.done and s2.won and s2.phase == Phase.WON


def test_legal_actions_in_shop():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    verbs = {a[0] for a in legal_actions(s2)}
    assert Verb.LEAVE_SHOP in verbs
    assert Verb.BUY in verbs     # affordable offers exist
    assert Verb.REROLL in verbs
```

Also UPDATE the Plan-1 test `tests/engine/test_engine.py::test_clearing_a_blind_advances_and_resets_counters` — the old behavior (clear → immediate advance) is replaced by (clear → SHOP). Replace that test body with:

```python
def test_clearing_a_blind_enters_shop():
    import dataclasses
    from balatro_rl.engine.state import Phase
    s = reset(seed=1)
    big_hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
                Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=big_hand, required=10)
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert info.get("cleared") is True
    assert s2.phase == Phase.SHOP        # now enters the shop instead of advancing
    assert s2.blind_index == 0           # advance is deferred to LEAVE_SHOP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_engine_shop.py -v`
Expected: FAIL — `Verb` has no `BUY` / shop flow not implemented

- [ ] **Step 3: Write minimal implementation** (edit `balatro_rl/engine/engine.py`)

Add imports near the top:

```python
from .economy import blind_reward, interest, MONEY_PER_UNUSED_HAND
from .shop import generate_offers, joker_cost, reroll_cost, sell_value, CARD_SLOTS
```

Extend the `Verb` enum:

```python
class Verb(IntEnum):
    PLAY = 0
    DISCARD = 1
    BUY = 2
    SELL = 3
    REROLL = 4
    REORDER = 5
    LEAVE_SHOP = 6
```

Add a constant near the others:

```python
JOKER_SLOTS = 5
```

Replace `_advance_blind` with a version that advances from the (already-cashed-out) state and returns to PLAYING:

```python
def _advance_blind(state: GameState):
    if state.blind_index < 2:
        new_ante, new_blind = state.ante, state.blind_index + 1
    else:
        new_ante, new_blind = state.ante + 1, 0
    deck, rng = state.rng.shuffle(standard_deck())
    hand, deck = _draw([], deck, state.hand_size)
    nxt = dataclasses.replace(
        state, ante=new_ante, blind_index=new_blind, deck=tuple(deck), hand=tuple(hand),
        round_score=0, required=required_score(new_ante, new_blind),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND, rng=rng,
        phase=Phase.PLAYING, shop_offers=(), rerolls_done=0)
    return nxt, {"verb": "leave_shop", "result": "next_blind",
                 "ante": new_ante, "blind": new_blind}


def _cash_out(state: GameState):
    """Apply blind reward + interest + leftover-hand money + joker on_round_end."""
    delta = (blind_reward(state.blind_index)
             + interest(state.money)
             + state.hands_left * MONEY_PER_UNUSED_HAND)
    money = state.money + delta
    rng = state.rng
    kept = []
    for js in state.jokers:
        js2, mdelta, destroy, rng = REGISTRY[js.type].on_round_end(state, js, rng)
        money += mdelta
        if not destroy:
            kept.append(js2)
    return money, tuple(kept), rng


def _enter_cashout_or_win(state: GameState, info: dict):
    # Win immediately if the Ante-8 Boss was just cleared (no shop).
    if state.ante >= 8 and state.blind_index == 2:
        won = dataclasses.replace(state, done=True, won=True, phase=Phase.WON)
        return won, {**info, "cleared": True, "result": "won"}
    money, jokers, rng = _cash_out(state)
    offers, rng = generate_offers(rng, CARD_SLOTS)
    shop = dataclasses.replace(state, money=money, jokers=jokers, rng=rng,
                               phase=Phase.SHOP, shop_offers=offers, rerolls_done=0)
    return shop, {**info, "cleared": True, "result": "shop", "earned": money - state.money}
```

In `step`, route SHOP-phase actions and change the PLAY clear path. At the very top of `step` (after `assert not state.done`):

```python
    if state.phase == Phase.SHOP:
        return _shop_step(state, action)
```

In the PLAY branch, replace the clear path. Where it currently reads:

```python
    if round_score >= state.required:
        carried = dataclasses.replace(state, jokers=new_jokers)
        return _advance_blind(carried, round_score, info)
```

change it to:

```python
    if round_score >= state.required:
        carried = dataclasses.replace(state, jokers=new_jokers, round_score=round_score,
                                      hands_left=hands_left)  # decremented count feeds cash-out
        return _enter_cashout_or_win(carried, info)
```

Add the shop step handler:

```python
def _shop_step(state: GameState, action):
    verb = action[0]
    if verb == Verb.LEAVE_SHOP:
        return _advance_blind(state)
    if verb == Verb.BUY:
        i = action[1]
        offer = state.shop_offers[i]
        cost = joker_cost(offer.type)
        assert state.money >= cost, "cannot afford"
        assert len(state.jokers) < JOKER_SLOTS, "no joker slot"
        offers = tuple(o for k, o in enumerate(state.shop_offers) if k != i)
        nxt = dataclasses.replace(state, money=state.money - cost,
                                  jokers=state.jokers + (offer,), shop_offers=offers)
        return nxt, {"verb": "buy", "joker": int(offer.type), "cost": cost}
    if verb == Verb.SELL:
        i = action[1]
        js = state.jokers[i]
        value = sell_value(js.type, js.sell_bonus)
        jokers = tuple(j for k, j in enumerate(state.jokers) if k != i)
        nxt = dataclasses.replace(state, money=state.money + value, jokers=jokers)
        return nxt, {"verb": "sell", "value": value}
    if verb == Verb.REROLL:
        cost = reroll_cost(state.rerolls_done)
        assert state.money >= cost, "cannot afford reroll"
        offers, rng = generate_offers(state.rng, CARD_SLOTS)
        nxt = dataclasses.replace(state, money=state.money - cost, shop_offers=offers,
                                  rerolls_done=state.rerolls_done + 1, rng=rng)
        return nxt, {"verb": "reroll", "cost": cost}
    if verb == Verb.REORDER:
        i, j = action[1]
        jk = list(state.jokers)
        item = jk.pop(i)
        jk.insert(j, item)
        return dataclasses.replace(state, jokers=tuple(jk)), {"verb": "reorder"}
    raise ValueError(f"illegal shop action: {verb}")
```

Replace `legal_actions` to handle the SHOP phase. At the top of `legal_actions(state)`, after the `if state.done: return []` guard, add:

```python
    if state.phase == Phase.SHOP:
        actions = [(Verb.LEAVE_SHOP, 0)]
        for i, offer in enumerate(state.shop_offers):
            if state.money >= joker_cost(offer.type) and len(state.jokers) < JOKER_SLOTS:
                actions.append((Verb.BUY, i))
        for i in range(len(state.jokers)):
            actions.append((Verb.SELL, i))
        if state.money >= reroll_cost(state.rerolls_done):
            actions.append((Verb.REROLL, 0))
        n = len(state.jokers)
        for i in range(n):
            for j in range(n):
                if i != j:
                    actions.append((Verb.REORDER, (i, j)))
        return actions
```

(The existing play/discard enumeration stays as the PLAYING-phase branch below it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/test_engine_shop.py tests/engine/test_engine.py -v`
Expected: PASS (new shop tests + updated/again-green Plan-1 engine tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/engine.py tests/engine/test_engine_shop.py tests/engine/test_engine.py
git commit -m "feat(engine): cash-out + shop phase (buy/sell/reroll/reorder/leave) in step"
```

---

### Task 7: Economy jokers — Golden Joker, Egg, Cavendish self-destroy

**Files:**
- Modify: `balatro_rl/engine/jokers/base.py` (add JokerType members)
- Modify: `balatro_rl/engine/jokers/library.py`
- Test: `tests/engine/jokers/test_economy_jokers.py`

**Wiki values (verify):** Golden Joker +$4 at end of round (`/w/Golden_Joker`, Common $6); Egg gains +$3 sell value at end of round (`/w/Egg`, Common $4); Cavendish 1-in-1000 self-destroy at end of round (`/w/Cavendish`).

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_economy_jokers.py
import dataclasses
from balatro_rl.engine.jokers.base import JokerType, JokerState, REGISTRY
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.shop import sell_value
import balatro_rl.engine.jokers.library  # noqa: F401


def test_golden_joker_pays_4_at_round_end():  # wiki: /w/Golden_Joker
    eff = REGISTRY[JokerType.GOLDEN_JOKER]
    js = JokerState(type=JokerType.GOLDEN_JOKER)
    js2, money_delta, destroy, _ = eff.on_round_end(None, js, RNG.from_seed(1))
    assert money_delta == 4 and destroy is False and js2 is js


def test_egg_gains_3_sell_value_each_round():  # wiki: /w/Egg
    eff = REGISTRY[JokerType.EGG]
    js = JokerState(type=JokerType.EGG)
    js2, money_delta, destroy, _ = eff.on_round_end(None, js, RNG.from_seed(1))
    assert money_delta == 0 and destroy is False
    assert js2.sell_bonus == 3
    # sell value reflects the bonus (Egg cost 4 -> floor=2, +3 = 5)
    assert sell_value(JokerType.EGG, js2.sell_bonus) == 5


def test_cavendish_usually_survives_round_end():  # wiki: /w/Cavendish  (1 in 1000)
    eff = REGISTRY[JokerType.CAVENDISH]
    js = JokerState(type=JokerType.CAVENDISH)
    # seed 1's first random() is >= 0.001, so it survives and rng advances.
    js2, money_delta, destroy, rng2 = eff.on_round_end(None, js, RNG.from_seed(1))
    assert destroy is False and money_delta == 0
    assert rng2 != RNG.from_seed(1)   # rng consumed by the roll


def test_cavendish_destroys_on_low_roll():
    eff = REGISTRY[JokerType.CAVENDISH]
    js = JokerState(type=JokerType.CAVENDISH)
    # Find a seed whose first random() < 0.001 by scanning; assert destroy fires.
    from balatro_rl.engine.rng import RNG as _RNG
    seed = next(s for s in range(100000) if _RNG.from_seed(s).random()[0] < 0.001)
    _, _, destroy, _ = eff.on_round_end(None, js, _RNG.from_seed(seed))
    assert destroy is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_economy_jokers.py -v`
Expected: FAIL — `JokerType` has no `GOLDEN_JOKER`

- [ ] **Step 3: Write minimal implementation**

In `balatro_rl/engine/jokers/base.py`, add to `JokerType`:

```python
    GOLDEN_JOKER = 90
    EGG = 46
```

In `balatro_rl/engine/jokers/library.py`, append:

```python
@register(JokerType.GOLDEN_JOKER)
class _GoldenJoker(JokerEffect):  # wiki: /w/Golden_Joker  — +$4 at end of round
    rarity = Rarity.COMMON
    cost = 6
    def on_round_end(self, state, js, rng):
        return js, 4, False, rng


@register(JokerType.EGG)
class _Egg(JokerEffect):  # wiki: /w/Egg  — gains +$3 sell value at end of round
    rarity = Rarity.COMMON
    cost = 4
    def on_round_end(self, state, js, rng):
        return dataclasses.replace(js, sell_bonus=js.sell_bonus + 3), 0, False, rng
```

And extend the existing `_Cavendish` class to add its self-destroy (keep its `independent` ×3):

```python
@register(JokerType.CAVENDISH)
class _Cavendish(JokerEffect):  # wiki: /w/Cavendish  — X3 Mult; 1 in 1000 self-destroy at end of round
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(xmult=3.0)
    def on_round_end(self, state, js, rng):
        roll, rng = rng.random()
        return js, 0, roll < 0.001, rng
```

(`dataclasses` is already imported in `library.py` from Task 8 of Plan 2's Ride the Bus; if not, add `import dataclasses` at the top.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_economy_jokers.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/base.py balatro_rl/engine/jokers/library.py tests/engine/jokers/test_economy_jokers.py
git commit -m "feat(jokers): Golden Joker, Egg, Cavendish self-destroy (on_round_end)"
```

---

### Task 8: Full-loop integration check

**Files:**
- Test: `tests/engine/test_economy_loop.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_economy_loop.py
import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, step
from balatro_rl.engine.state import Phase
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def _clearable(seed=1, **over):
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    return dataclasses.replace(s, hand=hand, required=10, **over)


def test_play_clear_shop_buy_leave_next_blind():
    s = _clearable(money=100, hands_left=2)
    s, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))     # clear Small -> shop
    assert s.phase == Phase.SHOP
    money_in_shop = s.money
    s, _ = step(s, (Verb.BUY, 0))                    # buy first offer
    assert len(s.jokers) == 1 and s.money < money_in_shop
    s, _ = step(s, (Verb.LEAVE_SHOP, 0))             # leave -> Big blind
    assert s.phase == Phase.PLAYING and s.blind_index == 1
    assert len(s.jokers) == 1                        # joker persists across the shop


def test_golden_joker_pays_out_at_cashout():
    s = _clearable(money=0, hands_left=1,
                   jokers=(JokerState(JokerType.GOLDEN_JOKER),))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))       # clear Small; 1 hand -> 0 left after play
    # money = 0 + reward(3) + interest(0) + hands(0 left) + golden(4) = 7
    assert s2.money == 7


def test_deterministic_full_run_is_reproducible():
    # Same seed + same scripted actions -> identical money/phase trajectory.
    def run():
        s = _clearable(seed=99, money=50, hands_left=1)
        s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
        s, _ = step(s, (Verb.REROLL, 0))
        s, _ = step(s, (Verb.LEAVE_SHOP, 0))
        return s
    a, b = run(), run()
    assert a.money == b.money and a.blind_index == b.blind_index
    assert a.shop_offers == b.shop_offers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_economy_loop.py -v`
Expected: initially may FAIL if any wiring is off; otherwise PASS after Tasks 1–7.

- [ ] **Step 3: Implementation**

No new code — this validates the assembled loop. If it fails, fix the responsible task.

- [ ] **Step 4: Run the FULL suite**

Run: `python3 -m pytest -v`
Expected: ALL tests pass (Plans 1–3). Also `python3 -m balatro_rl.engine 7` still terminates (random play now also exercises the shop when it clears a blind).

- [ ] **Step 5: Commit**

```bash
git add tests/engine/test_economy_loop.py
git commit -m "test(engine): full play->cash-out->shop->next-blind loop integration"
```

---

## Self-Review

**1. Spec coverage (economy-shop reference §1–§6 core):**
- Blind reward + interest + leftover hands → Tasks 2, 6 ✓
- Cash-out at clear (then shop) → Task 6 ✓
- Shop offer generation (rarity-weighted) + prices → Tasks 1, 4 ✓
- Buy / sell / reroll / reorder / leave + legal actions → Task 6 ✓
- `on_round_end` lifecycle + economy jokers (Golden, Egg, Cavendish destroy) → Tasks 3, 7 ✓
- Win-at-Ante-8-boss bypasses shop → Task 6 ✓
- **Deferred (correctly):** vouchers, packs, tags, consumables/tarots in shop, edition surcharges, stake/deck money modifiers, Green-deck discard money, the other economy jokers — later plans (flagged in scope).

**2. Placeholder scan:** none — every step has complete code + exact expected values from the verified reference.

**3. Type consistency:** `Verb` extended (BUY/SELL/REROLL/REORDER/LEAVE_SHOP) and used identically in `step`/`_shop_step`/`legal_actions`; `on_round_end(state, js, rng) -> (js, money_delta, destroy, rng)` signature consistent across base/library/`_cash_out`; `generate_offers(rng, n) -> (offers, rng)`, `joker_cost`, `reroll_cost`, `sell_value` signatures consistent across `shop.py`/`engine.py`; `GameState` shop fields (`shop_offers`, `rerolls_done`) and `Phase.SHOP` consistent across `state.py`/`engine.py`/tests.

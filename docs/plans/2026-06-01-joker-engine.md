# Joker Engine + Proof Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.
>
> **ACCURACY RULE (non-negotiable):** every joker's numbers in this plan are taken from balatrowiki.org. Before implementing each joker, `WebFetch` its wiki page (`https://balatrowiki.org/w/<Name>`) and confirm the exact value/wording. Reviewers must do the same. Citations are inline as `# wiki: <url>`.

**Goal:** Build the joker engine — a registry of per-joker hooks folded into the scoring pipeline — and validate it end-to-end with an 11-joker **proof set** that exercises every mechanism (independent, on-scored, on-held, retrigger, scaling, rule-modifiers, copy) and their hard interactions.

**Architecture:** Each joker is a small `JokerEffect` in a registry, implementing only the hooks it needs (`independent`, `on_score`, `on_held`, `retrigger`, `rules`, `on_play`). The scoring pipeline gathers `RuleFlags` → `evaluate(cards, rules)` → folds hooks over played cards (×retriggers), held cards, then independent jokers in slot order, applying additive-before-multiplicative. Copy jokers (Blueprint) resolve to a neighbor's *copyable* hooks. State stays frozen/pure; per-joker scaling lives in `JokerState.counter`.

**Tech Stack:** Python ≥3.11, `dataclasses`, `pytest`. Builds on the Plan-1 engine (`balatro_rl/engine/`).

**Campaign context (Plan 2 of the joker campaign):** Plan 2 = engine + proof set (this). Plan 3 = bulk base-scoring jokers (~100). Plans 4–7 = shop/economy, consumables, enhancements/seals, packs/tags/bosses + their dependent jokers → all 150. The proof set deliberately includes the *hard* mechanisms so architectural risk is retired now.

**Proof set (11):** Joker, Cavendish, Greedy Joker, Scary Face, Photograph, Baron, Hack, Ride the Bus, Splash, Pareidolia, Blueprint.

**Conventions:** repo `/Users/michael/Documents/GitHub/balatro-rl`; run `python3 -m pytest`; commit per task (no co-author trailers). Work on a feature branch off `master`.

---

### Task 1: Joker value types (Effect, RuleFlags, JokerType, JokerState)

**Files:**
- Create: `balatro_rl/engine/jokers/__init__.py`
- Create: `balatro_rl/engine/jokers/base.py`
- Test: `tests/engine/jokers/__init__.py`, `tests/engine/jokers/test_base_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_base_types.py
from balatro_rl.engine.jokers.base import (
    Effect, NO_EFFECT, RuleFlags, NO_RULES, JokerType, JokerState,
)


def test_effect_defaults_are_identity():
    assert (NO_EFFECT.chips, NO_EFFECT.mult, NO_EFFECT.xmult) == (0, 0.0, 1.0)


def test_effect_is_frozen():
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        Effect().chips = 5


def test_rule_flags_default_off():
    assert NO_RULES == RuleFlags(splash=False, all_face=False)


def test_joker_type_has_proof_set():
    names = {jt.name for jt in JokerType}
    assert {"JOKER", "CAVENDISH", "GREEDY", "SCARY_FACE", "PHOTOGRAPH", "BARON",
            "HACK", "RIDE_THE_BUS", "SPLASH", "PAREIDOLIA", "BLUEPRINT"} <= names


def test_joker_state_defaults():
    js = JokerState(type=JokerType.JOKER)
    assert js.edition == 0 and js.counter == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_base_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.jokers'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/jokers/__init__.py
```

```python
# balatro_rl/engine/jokers/base.py
"""Joker engine core: value types, the hook protocol, the registry, and the
copy/rule resolution helpers. Each joker is a small JokerEffect registered by
JokerType; the scoring pipeline (scoring.py) folds their hooks.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum


class JokerType(IntEnum):
    JOKER = 1
    CAVENDISH = 61
    GREEDY = 2
    SCARY_FACE = 33
    PHOTOGRAPH = 78
    BARON = 72
    HACK = 36
    RIDE_THE_BUS = 44
    SPLASH = 52
    PAREIDOLIA = 37
    BLUEPRINT = 123


@dataclasses.dataclass(frozen=True, slots=True)
class Effect:
    """A scoring contribution. Applied as: chips += chips; mult += mult; mult *= xmult."""
    chips: int = 0
    mult: float = 0.0
    xmult: float = 1.0


NO_EFFECT = Effect()


@dataclasses.dataclass(frozen=True, slots=True)
class RuleFlags:
    splash: bool = False     # every played card scores (Splash)
    all_face: bool = False   # all cards count as face cards (Pareidolia)

    def merge(self, other: "RuleFlags") -> "RuleFlags":
        return RuleFlags(splash=self.splash or other.splash,
                         all_face=self.all_face or other.all_face)


NO_RULES = RuleFlags()


@dataclasses.dataclass(frozen=True, slots=True)
class JokerState:
    """Per-instance joker state. `counter` holds scaling value (e.g. Ride the Bus mult)."""
    type: JokerType
    edition: int = 0      # 0 = base (editions are a later plan)
    counter: float = 0.0
```

Also create empty `tests/engine/jokers/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_base_types.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/ tests/engine/jokers/
git commit -m "feat(jokers): base value types (Effect, RuleFlags, JokerType, JokerState)"
```

---

### Task 2: Hook protocol, registry, ScoreContext, copy & rule resolution

**Files:**
- Modify: `balatro_rl/engine/jokers/base.py`
- Create: `tests/engine/conftest.py` (registry isolation fixture)
- Test: `tests/engine/jokers/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_registry.py
from balatro_rl.engine.jokers.base import (
    JokerEffect, JokerType, JokerState, Effect, RuleFlags,
    register, REGISTRY, aggregate_rules, resolve_providers, ScoreContext,
)


# Minimal stubs so the registry/resolve tests are self-contained when this file
# runs alone (the Plan-2 joker library is not imported here). The conftest
# fixture restores REGISTRY between tests, so these module-level stubs persist.
@register(JokerType.JOKER)
class _StubJoker(JokerEffect):
    def independent(self, ctx, js):
        return Effect(mult=4)


@register(JokerType.BLUEPRINT)
class _StubBlueprint(JokerEffect):
    pass


def test_default_hooks_are_noops():
    eff = JokerEffect()
    js = JokerState(type=JokerType.JOKER)
    assert eff.independent(None, js) == Effect()
    assert eff.on_score(None, None, 0, js) == Effect()
    assert eff.on_held(None, None, js) == Effect()
    assert eff.retrigger(None, None, js) == 0
    assert eff.rules() == RuleFlags()
    assert eff.copyable is True


def test_register_populates_registry():
    @register(JokerType.JOKER)
    class _J(JokerEffect):
        def independent(self, ctx, js):
            return Effect(mult=4)
    assert isinstance(REGISTRY[JokerType.JOKER], _J)
    assert REGISTRY[JokerType.JOKER].independent(None, JokerState(JokerType.JOKER)).mult == 4


def test_aggregate_rules_ors_flags():
    class _Splash(JokerEffect):
        def rules(self):
            return RuleFlags(splash=True)
    REGISTRY[JokerType.SPLASH] = _Splash()
    flags = aggregate_rules((JokerState(JokerType.SPLASH), JokerState(JokerType.JOKER)))
    assert flags.splash is True and flags.all_face is False


def test_resolve_providers_passes_through_non_copy():
    provs = resolve_providers((JokerState(JokerType.JOKER),))
    assert len(provs) == 1
    eff, js = provs[0]
    assert js.type == JokerType.JOKER


def test_blueprint_resolves_to_right_neighbor():
    # Blueprint (slot 0) left of Joker (slot 1) -> copies Joker's effect.
    class _BP(JokerEffect):
        pass
    REGISTRY[JokerType.BLUEPRINT] = _BP()
    jokers = (JokerState(JokerType.BLUEPRINT), JokerState(JokerType.JOKER))
    provs = resolve_providers(jokers)
    # Two providers: Blueprint-as-Joker, and Joker itself.
    assert provs[0][0] is REGISTRY[JokerType.JOKER]   # blueprint copies Joker's effect
    assert provs[1][0] is REGISTRY[JokerType.JOKER]


def test_blueprint_at_rightmost_contributes_nothing():
    jokers = (JokerState(JokerType.JOKER), JokerState(JokerType.BLUEPRINT))
    provs = resolve_providers(jokers)
    assert len(provs) == 1  # blueprint has no right neighbor -> dropped
    assert provs[0][1].type == JokerType.JOKER


def test_score_context_is_mutable():
    ctx = ScoreContext(chips=10, mult=2.0)
    ctx.chips += 5
    ctx.mult *= 3
    assert ctx.chips == 15 and ctx.mult == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'JokerEffect'`

- [ ] **Step 3: Write minimal implementation** (append to `balatro_rl/engine/jokers/base.py`)

```python
# --- append to balatro_rl/engine/jokers/base.py ---


@dataclasses.dataclass(slots=True)
class ScoreContext:
    """Mutable scratch used only during one hand's scoring (never stored in state)."""
    chips: int = 0
    mult: float = 0.0
    played: list = dataclasses.field(default_factory=list)
    scoring_idx: list = dataclasses.field(default_factory=list)
    held: list = dataclasses.field(default_factory=list)
    hand_type: object = None
    rules: RuleFlags = NO_RULES
    first_face_idx: int | None = None


class JokerEffect:
    """Base joker behaviour. Subclasses override only the hooks they need.

    `copyable` declares whether Blueprint/Brainstorm may copy this joker's
    scoring hooks (passive/rule/economy jokers set it False — see wiki).
    """
    copyable: bool = True

    def independent(self, ctx, js: "JokerState") -> Effect:
        return NO_EFFECT

    def on_score(self, ctx, card, index: int, js: "JokerState") -> Effect:
        return NO_EFFECT

    def on_held(self, ctx, card, js: "JokerState") -> Effect:
        return NO_EFFECT

    def retrigger(self, ctx, card, js: "JokerState") -> int:
        return 0

    def rules(self) -> RuleFlags:
        return NO_RULES

    def on_play(self, state, played, scoring_idx, rules, js: "JokerState") -> "JokerState":
        """Lifecycle after a hand is played; return updated JokerState (scaling)."""
        return js


REGISTRY: dict[JokerType, JokerEffect] = {}


def register(joker_type: JokerType):
    def deco(cls):
        REGISTRY[joker_type] = cls()
        return cls
    return deco


def aggregate_rules(jokers: tuple) -> RuleFlags:
    flags = NO_RULES
    for js in jokers:
        flags = flags.merge(REGISTRY[js.type].rules())
    return flags


def _blueprint_target(jokers: tuple, i: int) -> int | None:
    """Index of the joker Blueprint at slot i ultimately copies (walks right past
    chained Blueprints; None if it runs off the end)."""
    j = i + 1
    seen = set()
    while j < len(jokers) and jokers[j].type == JokerType.BLUEPRINT:
        if j in seen:
            return None
        seen.add(j)
        j += 1
    return j if j < len(jokers) else None


def resolve_providers(jokers: tuple) -> list:
    """Return [(JokerEffect, JokerState)] in slot order, with Blueprint resolved to
    its target's *copyable* effect (using the target's state)."""
    out = []
    for i, js in enumerate(jokers):
        if js.type == JokerType.BLUEPRINT:
            tgt = _blueprint_target(jokers, i)
            if tgt is None:
                continue
            teff = REGISTRY[jokers[tgt].type]
            if teff.copyable:
                out.append((teff, jokers[tgt]))
        else:
            out.append((REGISTRY[js.type], js))
    return out
```

Also create the registry-isolation fixture so tests that register stub jokers don't leak into each other (the global `REGISTRY` is mutable; without this, test order changes results):

```python
# tests/engine/conftest.py
import pytest

from balatro_rl.engine.jokers.base import REGISTRY


@pytest.fixture(autouse=True)
def _isolate_joker_registry():
    """Snapshot REGISTRY before each test and restore it after, so per-test
    registrations (stubs/fakes) never contaminate other tests."""
    saved = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_registry.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/base.py tests/engine/jokers/test_registry.py
git commit -m "feat(jokers): hook protocol, registry, ScoreContext, copy/rule resolution"
```

---

### Task 3: `evaluate()` accepts RuleFlags; add `is_face()`

**Files:**
- Modify: `balatro_rl/engine/hands.py`
- Test: `tests/engine/test_hands_rules.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_hands_rules.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import evaluate, is_face, HandType
from balatro_rl.engine.jokers.base import RuleFlags


def C(rank, suit):
    return Card(rank=rank, suit=suit)


def test_evaluate_default_rules_unchanged():
    # Backward compatible: no rules arg behaves like Plan 1.
    ht, idx = evaluate([C(13, 0), C(13, 1), C(3, 3), C(7, 3), C(9, 1)])
    assert ht == HandType.PAIR
    assert sorted(idx) == [0, 1]


def test_splash_makes_all_cards_score():
    rules = RuleFlags(splash=True)
    ht, idx = evaluate([C(13, 0), C(13, 1), C(3, 3), C(7, 3), C(9, 1)], rules)
    assert ht == HandType.PAIR                 # hand type unchanged
    assert sorted(idx) == [0, 1, 2, 3, 4]      # but every card scores


def test_is_face_normal():
    assert is_face(C(13, 0), RuleFlags()) is True    # King
    assert is_face(C(12, 0), RuleFlags()) is True    # Queen
    assert is_face(C(11, 0), RuleFlags()) is True     # Jack
    assert is_face(C(10, 0), RuleFlags()) is False
    assert is_face(C(2, 0), RuleFlags()) is False


def test_is_face_with_pareidolia():
    assert is_face(C(2, 0), RuleFlags(all_face=True)) is True
    assert is_face(C(10, 0), RuleFlags(all_face=True)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_hands_rules.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_face'` (and `evaluate` rejects 2nd arg)

- [ ] **Step 3: Write minimal implementation** (edit `balatro_rl/engine/hands.py`)

Add the import at the top and a module-level default:

```python
from .jokers.base import RuleFlags, NO_RULES
```

Add the `is_face` helper (after `_is_straight`):

```python
def is_face(card: Card, rules: RuleFlags = NO_RULES) -> bool:
    """King/Queen/Jack, or any card when Pareidolia (all_face) is active."""
    return rules.all_face or card.rank in (11, 12, 13)
```

Change the `evaluate` signature and the final `scoring_idx` selection. Replace the signature line:

```python
def evaluate(cards: list[Card], rules: RuleFlags = NO_RULES) -> tuple[HandType, tuple[int, ...]]:
```

Then, immediately before `return` of each branch is unchanged EXCEPT: wrap the whole body so that when `rules.splash` is set, every played index scores. Do this by computing the normal result, then overriding the indices. Concretely, rename the existing function body to compute `hand_type, idx`, then at the very end return splash-adjusted indices. The simplest correct edit: keep all existing branch logic but capture its result and apply splash at the end. Replace the existing chain of `return HandType.X, ...` statements by assigning to a local and returning once:

```python
def evaluate(cards: list[Card], rules: RuleFlags = NO_RULES) -> tuple[HandType, tuple[int, ...]]:
    """Best (HandType, scoring-card indices) for 1..5 played cards.

    With rules.splash, every played card scores (indices = all), though the hand
    type is still the best poker hand.
    """
    n = len(cards)
    if n == 0:
        raise ValueError("evaluate() requires at least one card")
    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    is_flush = n == 5 and len(set(suits)) == 1
    is_straight = n == 5 and _is_straight(ranks)
    all_idx = tuple(range(n))

    def idx_with_count(k: int) -> tuple[int, ...]:
        targets = {r for r, c in rank_counts.items() if c == k}
        return tuple(i for i, r in enumerate(ranks) if r in targets)

    if is_flush and counts == [5]:
        hand_type, idx = HandType.FLUSH_FIVE, all_idx
    elif is_flush and counts == [3, 2]:
        hand_type, idx = HandType.FLUSH_HOUSE, all_idx
    elif counts == [5]:
        hand_type, idx = HandType.FIVE_OF_A_KIND, all_idx
    elif is_flush and is_straight:
        hand_type, idx = HandType.STRAIGHT_FLUSH, all_idx
    elif counts and counts[0] == 4:
        hand_type, idx = HandType.FOUR_OF_A_KIND, idx_with_count(4)
    elif counts == [3, 2]:
        hand_type, idx = HandType.FULL_HOUSE, all_idx
    elif is_flush:
        hand_type, idx = HandType.FLUSH, all_idx
    elif is_straight:
        hand_type, idx = HandType.STRAIGHT, all_idx
    elif counts and counts[0] == 3:
        hand_type, idx = HandType.THREE_OF_A_KIND, idx_with_count(3)
    elif counts[:2] == [2, 2]:
        hand_type, idx = HandType.TWO_PAIR, idx_with_count(2)
    elif counts and counts[0] == 2:
        hand_type, idx = HandType.PAIR, idx_with_count(2)
    else:
        hi = max(range(n), key=lambda i: ranks[i])
        hand_type, idx = HandType.HIGH_CARD, (hi,)

    if rules.splash:
        idx = all_idx
    return hand_type, idx
```

(Delete the old branch-based body so the function is defined exactly once.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/test_hands_rules.py tests/engine/test_hands.py -v`
Expected: PASS (new file + all Plan-1 hand tests still green)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/hands.py tests/engine/test_hands_rules.py
git commit -m "feat(engine): evaluate() honors RuleFlags (Splash); add is_face()"
```

---

### Task 4: Scoring pipeline folds joker hooks

**Files:**
- Modify: `balatro_rl/engine/scoring.py`
- Test: `tests/engine/test_scoring_jokers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_scoring_jokers.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerEffect, JokerType, JokerState, Effect, register


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def test_no_jokers_matches_base_scoring():
    # Pair of Kings: (10 + 10 + 10) * 2 = 60 (Plan-1 behavior preserved).
    res = score_play([C(13), C(13), C(3), C(7), C(9)])
    assert res.score == 60 and res.chips == 30 and res.mult == 2.0


def test_independent_additive_then_xmult_order():
    # Register a +10 mult and a x3 joker; slot order = additive then xmult.
    @register(JokerType.JOKER)
    class _Add(JokerEffect):
        def independent(self, ctx, js):
            return Effect(mult=10)

    @register(JokerType.CAVENDISH)
    class _X(JokerEffect):
        def independent(self, ctx, js):
            return Effect(xmult=3.0)

    # High card Ace: base (5,1) + 11 chips = 16 chips, mult 1.
    # +10 mult -> mult 11; x3 -> mult 33; score = 16*33 = 528.
    jokers = (JokerState(JokerType.JOKER), JokerState(JokerType.CAVENDISH))
    res = score_play([C(14), C(7), C(2)], jokers=jokers)
    assert res.mult == 33.0
    assert res.score == 16 * 33
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_scoring_jokers.py -v`
Expected: FAIL — `score_play()` got an unexpected keyword argument `jokers`

- [ ] **Step 3: Write minimal implementation** (rewrite `balatro_rl/engine/scoring.py`)

```python
# balatro_rl/engine/scoring.py
"""Scoring pipeline. Folds joker hooks over a played hand.

Order (per wiki https://balatrowiki.org/w/Scoring):
  1. each scoring card L->R, repeated (1 + retriggers): card chips + on_score hooks
  2. each held card: on_held hooks
  3. independent jokers in slot order
Additive (+chips/+mult) applies before ×mult within the fold (slot order matters).
With no jokers the result is identical to the Plan-1 base scoring.
"""
from __future__ import annotations

import dataclasses

from .cards import Card, rank_chip_value
from .hands import HAND_BASE, HandType, evaluate, is_face
from .jokers.base import (
    NO_RULES, ScoreContext, aggregate_rules, resolve_providers,
)


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreResult:
    score: int
    hand_type: HandType
    chips: int
    mult: float
    scoring_idx: tuple[int, ...]


def _apply(ctx: ScoreContext, eff) -> None:
    ctx.chips += eff.chips
    ctx.mult += eff.mult
    ctx.mult *= eff.xmult


def score_play(played, jokers: tuple = (), held: tuple = ()) -> ScoreResult:
    played = list(played)
    rules = aggregate_rules(jokers) if jokers else NO_RULES
    hand_type, scoring_idx = evaluate(played, rules)
    base_chips, base_mult = HAND_BASE[hand_type]

    ctx = ScoreContext(chips=base_chips, mult=float(base_mult), played=played,
                       scoring_idx=list(scoring_idx), held=list(held),
                       hand_type=hand_type, rules=rules)
    ctx.first_face_idx = next((i for i in scoring_idx if is_face(played[i], rules)), None)
    providers = resolve_providers(jokers)

    # 1) played scoring cards, left to right, with retriggers
    for i in scoring_idx:
        card = played[i]
        retriggers = sum(eff.retrigger(ctx, card, js) for eff, js in providers)
        for _ in range(1 + retriggers):
            ctx.chips += rank_chip_value(card.rank)
            for eff, js in providers:
                _apply(ctx, eff.on_score(ctx, card, i, js))

    # 2) held-in-hand cards
    for card in held:
        for eff, js in providers:
            _apply(ctx, eff.on_held(ctx, card, js))

    # 3) independent jokers, slot order
    for eff, js in providers:
        _apply(ctx, eff.independent(ctx, js))

    return ScoreResult(score=int(ctx.chips * ctx.mult), hand_type=hand_type,
                       chips=ctx.chips, mult=ctx.mult, scoring_idx=tuple(scoring_idx))
```

> Note: `score_play([cards])` with no jokers yields `providers=[]`, so it reduces to base chips×mult — Plan-1 tests stay green. The proof-set jokers (Tasks 6–10) `register()` real implementations, replacing the test stubs above.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/test_scoring_jokers.py tests/engine/test_scoring.py -v`
Expected: PASS (new + Plan-1 scoring tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/scoring.py tests/engine/test_scoring_jokers.py
git commit -m "feat(engine): scoring pipeline folds joker hooks (retrigger, held, independent)"
```

---

### Task 5: The proof-set joker library — independent & on-scored

**Files:**
- Create: `balatro_rl/engine/jokers/library.py`
- Test: `tests/engine/jokers/test_library_scoring.py`

**Wiki values (verify each before coding):** Joker +4 Mult (`/w/Joker`); Cavendish ×3 (`/w/Cavendish`); Greedy +3 Mult per ♦ scored (`/w/Greedy_Joker`); Scary Face +30 Chips per face scored (`/w/Scary_Face`); Photograph ×2 on first scoring face card, re-applies per retrigger (`/w/Photograph`). Suit ints: ♠0 ♥1 ♣2 ♦3.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_library_scoring.py
import importlib
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401  (registers jokers)


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_joker_plus_4_mult():
    # High-card Ace: 16 chips, base mult 1, +4 -> mult 5, score 80.  # wiki: /w/Joker
    res = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER),))
    assert res.mult == 5.0 and res.score == 80


def test_cavendish_x3():
    # High-card Ace: 16 chips, mult 1 -> x3 = 3, score 48.  # wiki: /w/Cavendish
    res = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.CAVENDISH),))
    assert res.mult == 3.0 and res.score == 48


def test_greedy_per_diamond():
    # Pair of Kings (♦=suit3): two kings are ♦. base (10,2)+20 chips=30.
    # Greedy +3 per scored ♦; both kings score -> +6 mult -> mult 8 -> 240.  # wiki: /w/Greedy_Joker
    res = score_play([C(13, 3), C(13, 3), C(3, 0), C(7, 0), C(9, 1)],
                     jokers=(J(JokerType.GREEDY),))
    assert res.mult == 8.0 and res.score == 30 * 8


def test_scary_face_per_face():
    # Pair of Kings: both kings are face -> +30 chips each = +60. chips 30+60=90.  # wiki: /w/Scary_Face
    res = score_play([C(13), C(13), C(3), C(7), C(9)], jokers=(J(JokerType.SCARY_FACE),))
    assert res.chips == 90 and res.score == 90 * 2


def test_photograph_first_face_only():
    # Two kings score; Photograph x2 applies to the FIRST scoring face card only.
    # base (10,2)+20 chips=30 chips, mult 2 -> x2 once = 4 -> 120.  # wiki: /w/Photograph
    res = score_play([C(13), C(13), C(3), C(7), C(9)], jokers=(J(JokerType.PHOTOGRAPH),))
    assert res.mult == 4.0 and res.score == 30 * 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_library_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: ...jokers.library`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/jokers/library.py
"""The proof-set joker implementations. Each value is wiki-cited; verify against
balatrowiki.org before changing. This module registers jokers on import.
"""
from __future__ import annotations

import dataclasses

from ..hands import is_face
from .base import Effect, JokerEffect, JokerState, JokerType, RuleFlags, register


@register(JokerType.JOKER)
class _Joker(JokerEffect):  # wiki: /w/Joker  — +4 Mult
    def independent(self, ctx, js):
        return Effect(mult=4)


@register(JokerType.CAVENDISH)
class _Cavendish(JokerEffect):  # wiki: /w/Cavendish  — X3 Mult
    def independent(self, ctx, js):
        return Effect(xmult=3.0)


@register(JokerType.GREEDY)
class _Greedy(JokerEffect):  # wiki: /w/Greedy_Joker  — +3 Mult per scored Diamond
    def on_score(self, ctx, card, index, js):
        return Effect(mult=3) if card.suit == 3 else Effect()


@register(JokerType.SCARY_FACE)
class _ScaryFace(JokerEffect):  # wiki: /w/Scary_Face  — +30 Chips per scored face card
    def on_score(self, ctx, card, index, js):
        return Effect(chips=30) if is_face(card, ctx.rules) else Effect()


@register(JokerType.PHOTOGRAPH)
class _Photograph(JokerEffect):  # wiki: /w/Photograph  — X2 on first scoring face card (re-applies on retrigger)
    def on_score(self, ctx, card, index, js):
        return Effect(xmult=2.0) if index == ctx.first_face_idx else Effect()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_library_scoring.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/library.py tests/engine/jokers/test_library_scoring.py
git commit -m "feat(jokers): Joker, Cavendish, Greedy, Scary Face, Photograph"
```

---

### Task 6: On-held (Baron) and retrigger (Hack), with interactions

**Files:**
- Modify: `balatro_rl/engine/jokers/library.py`
- Test: `tests/engine/jokers/test_library_held_retrigger.py`

**Wiki values:** Baron ×1.5 per King held, stacks ×1.5ⁿ, held phase (`/w/Baron`); Hack retriggers each played 2/3/4/5, re-firing chips + on-scored joker effects (`/w/Hack`).

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_library_held_retrigger.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_baron_one_king_held():
    # Play a low pair; hold one King. base pair (5,4)? -> use a pair of 3s:
    # Pair of 3s: base (10,2) + (3+3)=6 -> 16 chips, mult 2. Baron x1.5 -> mult 3 -> 48.  # wiki: /w/Baron
    res = score_play([C(3), C(3), C(7), C(9), C(2)], jokers=(J(JokerType.BARON),),
                     held=(C(13),))
    assert res.mult == 3.0 and res.score == 16 * 3


def test_baron_two_kings_held_exponential():
    # Two Kings held -> x1.5^2 = x2.25.  mult 2 -> 4.5 -> int(16*4.5)=72.  # wiki: /w/Baron
    res = score_play([C(3), C(3), C(7), C(9), C(2)], jokers=(J(JokerType.BARON),),
                     held=(C(13), C(13)))
    assert res.mult == 4.5 and res.score == int(16 * 4.5)


def test_hack_retriggers_low_cards_chips():
    # Pair of 3s scored twice each via Hack. Base pair (10,2). Without Hack chips=10+3+3=16.
    # Hack: each scoring 3 triggers twice -> chips 10 + (3+3)*2 = 22, mult 2 -> 44.  # wiki: /w/Hack
    res = score_play([C(3), C(3), C(7), C(9), C(2)], jokers=(J(JokerType.HACK),))
    assert res.chips == 22 and res.score == 44


def test_hack_retrigger_refires_greedy():
    # 3 of Diamonds (suit 3) scored; Hack retriggers it; Greedy +3 mult each trigger.
    # Pair of 3♦: base (10,2). chips = 10 + (3+3)*2 = 22 (Hack). Greedy fires on each
    # scored ♦ each trigger: 2 cards x 2 triggers x +3 = +12 mult -> mult 14 -> 22*14=308.
    res = score_play([C(3, 3), C(3, 3), C(7, 0), C(9, 0), C(2, 0)],
                     jokers=(J(JokerType.HACK), J(JokerType.GREEDY)))
    assert res.chips == 22 and res.mult == 14.0 and res.score == 22 * 14
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_library_held_retrigger.py -v`
Expected: FAIL — Baron/Hack not yet registered (KeyError in REGISTRY)

- [ ] **Step 3: Write minimal implementation** (append to `library.py`)

```python
# --- append to balatro_rl/engine/jokers/library.py ---


@register(JokerType.BARON)
class _Baron(JokerEffect):  # wiki: /w/Baron  — each King held gives X1.5 Mult
    def on_held(self, ctx, card, js):
        return Effect(xmult=1.5) if card.rank == 13 else Effect()


@register(JokerType.HACK)
class _Hack(JokerEffect):  # wiki: /w/Hack  — retrigger each played 2,3,4,5
    def retrigger(self, ctx, card, js):
        return 1 if card.rank in (2, 3, 4, 5) else 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_library_held_retrigger.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/library.py tests/engine/jokers/test_library_held_retrigger.py
git commit -m "feat(jokers): Baron (on-held xmult) and Hack (retrigger), with interactions"
```

---

### Task 7: Rule-modifiers (Splash, Pareidolia) with interactions

**Files:**
- Modify: `balatro_rl/engine/jokers/library.py`
- Test: `tests/engine/jokers/test_library_rules.py`

**Wiki values:** Splash — every played card scores (`/w/Splash`); Pareidolia — all cards count as face (`/w/Pareidolia`). Both are passive `rules()` and `copyable = False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_library_rules.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_splash_makes_nonscoring_diamonds_trigger_greedy():
    # Pair of 3s + three ♦ kickers. Without Splash only the 3s score.
    # With Splash all 5 score; Greedy +3 per ♦. Three ♦ kickers + (3s are ♠) -> +9 mult.
    cards = [C(3, 0), C(3, 0), C(7, 3), C(9, 3), C(2, 3)]  # three diamonds among kickers
    res = score_play(cards, jokers=(J(JokerType.SPLASH), J(JokerType.GREEDY)))
    # chips: base 10 + (3+3+7+9+2)=34 -> 34.  mult: 2 + 3*3 = 11.  score 34*11=374.
    assert res.chips == 34 and res.mult == 11.0 and res.score == 374


def test_pareidolia_makes_scary_face_hit_all_cards():
    # Pair of 3s; Pareidolia -> all scored cards are "face" -> Scary Face +30 each.
    # Only the two 3s score (no Splash). chips = 10 + (3+3) + 30*2 = 76. mult 2 -> 152.
    res = score_play([C(3), C(3), C(7), C(9), C(2)],
                     jokers=(J(JokerType.PAREIDOLIA), J(JokerType.SCARY_FACE)))
    assert res.chips == 76 and res.score == 152


def test_pareidolia_photograph_hits_first_card_any_rank():
    # Pareidolia: first scoring card (a 3) is "face" -> Photograph x2 once.
    res = score_play([C(3), C(3), C(7), C(9), C(2)],
                     jokers=(J(JokerType.PAREIDOLIA), J(JokerType.PHOTOGRAPH)))
    # chips = 10 + 6 = 16, mult 2 -> x2 = 4 -> 64.
    assert res.mult == 4.0 and res.score == 64


def test_rule_jokers_not_copyable():
    assert REGISTRY[JokerType.SPLASH].copyable is False
    assert REGISTRY[JokerType.PAREIDOLIA].copyable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_library_rules.py -v`
Expected: FAIL — Splash/Pareidolia not registered

- [ ] **Step 3: Write minimal implementation** (append to `library.py`)

```python
# --- append to balatro_rl/engine/jokers/library.py ---


@register(JokerType.SPLASH)
class _Splash(JokerEffect):  # wiki: /w/Splash  — every played card scores
    copyable = False
    def rules(self):
        return RuleFlags(splash=True)


@register(JokerType.PAREIDOLIA)
class _Pareidolia(JokerEffect):  # wiki: /w/Pareidolia  — all cards are face cards
    copyable = False
    def rules(self):
        return RuleFlags(all_face=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_library_rules.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/library.py tests/engine/jokers/test_library_rules.py
git commit -m "feat(jokers): Splash and Pareidolia rule-modifiers (+ interactions)"
```

---

### Task 8: Scaling (Ride the Bus) via the on_play lifecycle

**Files:**
- Modify: `balatro_rl/engine/jokers/library.py`
- Test: `tests/engine/jokers/test_library_scaling.py`

**Wiki values:** Ride the Bus — +1 Mult per consecutive hand played with no scoring face card; resets to 0 on any scoring face card; increments once per hand; suppressed by Pareidolia (`/w/Ride_the_Bus`).

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_library_scaling.py
import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.hands import evaluate, is_face
from balatro_rl.engine.jokers.base import (
    JokerType, JokerState, REGISTRY, aggregate_rules,
)
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def _play_update(js, played):
    """Mimic engine.step's lifecycle call: compute scoring + rules, then on_play."""
    rules = aggregate_rules((js,))
    _, scoring_idx = evaluate(list(played), rules)
    eff = REGISTRY[js.type]
    return eff.on_play(None, list(played), list(scoring_idx), rules, js)


def test_ride_the_bus_increments_on_faceless_hand():
    js = JokerState(JokerType.RIDE_THE_BUS, counter=0.0)
    js = _play_update(js, [C(2), C(2), C(7), C(9), C(3)])   # no face cards
    assert js.counter == 1.0
    js = _play_update(js, [C(5), C(5), C(7), C(9), C(3)])   # still faceless
    assert js.counter == 2.0


def test_ride_the_bus_resets_on_scoring_face():
    js = JokerState(JokerType.RIDE_THE_BUS, counter=5.0)
    js = _play_update(js, [C(13), C(13), C(7), C(9), C(3)])  # kings score -> reset
    assert js.counter == 0.0


def test_ride_the_bus_applies_counter_as_mult():
    # counter 3 -> +3 mult. Pair of 3s: 16 chips, mult 2+3=5 -> 80.
    js = JokerState(JokerType.RIDE_THE_BUS, counter=3.0)
    res = score_play([C(3), C(3), C(7), C(9), C(2)], jokers=(js,))
    assert res.mult == 5.0 and res.score == 80


def test_pareidolia_suppresses_ride_the_bus():
    # With Pareidolia all cards are face -> a faceless-looking hand still "has face" -> reset.
    rtb = JokerState(JokerType.RIDE_THE_BUS, counter=4.0)
    pare = JokerState(JokerType.PAREIDOLIA)
    rules = aggregate_rules((rtb, pare))
    played = [C(2), C(2), C(7), C(9), C(3)]
    _, scoring_idx = evaluate(played, rules)
    rtb2 = REGISTRY[JokerType.RIDE_THE_BUS].on_play(None, played, list(scoring_idx), rules, rtb)
    assert rtb2.counter == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_library_scaling.py -v`
Expected: FAIL — Ride the Bus not registered / no on_play behavior

- [ ] **Step 3: Write minimal implementation** (append to `library.py`)

```python
# --- append to balatro_rl/engine/jokers/library.py ---


@register(JokerType.RIDE_THE_BUS)
class _RideTheBus(JokerEffect):  # wiki: /w/Ride_the_Bus
    def independent(self, ctx, js):
        return Effect(mult=js.counter)

    def on_play(self, state, played, scoring_idx, rules, js):
        scored_face = any(is_face(played[i], rules) for i in scoring_idx)
        new_counter = 0.0 if scored_face else js.counter + 1.0
        return dataclasses.replace(js, counter=new_counter)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_library_scaling.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/library.py tests/engine/jokers/test_library_scaling.py
git commit -m "feat(jokers): Ride the Bus scaling via on_play lifecycle"
```

---

### Task 9: Copy (Blueprint)

**Files:**
- Modify: `balatro_rl/engine/jokers/library.py`
- Test: `tests/engine/jokers/test_library_blueprint.py`

**Wiki values:** Blueprint copies the ability of the joker to its right; both trigger; cannot copy passive/rule jokers; copies ability not edition (`/w/Blueprint`).

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_library_blueprint.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_blueprint_copies_joker_to_right():
    # Blueprint (slot0) left of Joker(+4). Both give +4 -> +8 mult.
    # High-card Ace: 16 chips, mult 1+8=9 -> 144.  # wiki: /w/Blueprint
    jokers = (J(JokerType.BLUEPRINT), J(JokerType.JOKER))
    res = score_play([C(14), C(7), C(2)], jokers=jokers)
    assert res.mult == 9.0 and res.score == 16 * 9


def test_blueprint_copies_greedy():
    # Blueprint + Greedy: each scored ♦ gives +3 twice. Pair of K♦: two ♦ -> +12 mult.
    jokers = (J(JokerType.BLUEPRINT), J(JokerType.GREEDY))
    res = score_play([C(13, 3), C(13, 3), C(3, 0), C(7, 0), C(9, 0)], jokers=jokers)
    # base (10,2)+20 chips=30; mult 2 + (2 cards x 2 jokers x 3) = 2+12 = 14 -> 420.
    assert res.mult == 14.0 and res.score == 420


def test_blueprint_cannot_copy_pareidolia():
    # Blueprint right-neighbor is Pareidolia (copyable=False) -> Blueprint does nothing,
    # but real Pareidolia still applies its rule. Scary Face hits all via real Pareidolia only.
    jokers = (J(JokerType.BLUEPRINT), J(JokerType.PAREIDOLIA), J(JokerType.SCARY_FACE))
    res = score_play([C(3), C(3), C(7), C(9), C(2)], jokers=jokers)
    # Pareidolia active once -> both 3s "face" -> Scary +30*2. chips 10+6+60=76, mult 2 -> 152.
    assert res.chips == 76 and res.score == 152


def test_blueprint_rightmost_does_nothing():
    # Blueprint with no right neighbor contributes nothing. High-card Ace -> 16.
    res = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER), J(JokerType.BLUEPRINT)))
    assert res.mult == 5.0 and res.score == 80   # only the real Joker's +4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_library_blueprint.py -v`
Expected: FAIL — Blueprint not registered

- [ ] **Step 3: Write minimal implementation** (append to `library.py`)

```python
# --- append to balatro_rl/engine/jokers/library.py ---


@register(JokerType.BLUEPRINT)
class _Blueprint(JokerEffect):  # wiki: /w/Blueprint  — copy resolution handled in base.resolve_providers
    pass
```

> Blueprint has no hooks of its own; `resolve_providers` (Task 2) substitutes the right neighbour's copyable effect using the neighbour's state.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/jokers/test_library_blueprint.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/jokers/library.py tests/engine/jokers/test_library_blueprint.py
git commit -m "feat(jokers): Blueprint copy joker (+ copyable exclusions)"
```

---

### Task 10: Engine integration — jokers in `step`, on_play scaling

**Files:**
- Modify: `balatro_rl/engine/state.py` (add `jokers` field)
- Modify: `balatro_rl/engine/engine.py` (use jokers in scoring; call on_play)
- Test: `tests/engine/test_engine_jokers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_engine_jokers.py
import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, step
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def test_reset_starts_with_no_jokers():
    s = reset(seed=1)
    assert s.jokers == ()


def test_play_uses_jokers_in_score():
    s = reset(seed=1)
    # Force a known hand and a Joker(+4); play the pair.
    hand = (Card(13, 0), Card(13, 1), Card(3, 2), Card(7, 2), Card(9, 2),
            Card(2, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, jokers=(JokerState(JokerType.JOKER),))
    s2, info = step(s, (Verb.PLAY, (0, 1)))
    # Pair of Kings: chips 30, mult 2+4=6 -> 180.
    assert info["score"] == 180
    assert s2.round_score == 180


def test_ride_the_bus_counter_advances_through_step():
    s = reset(seed=1)
    hand = (Card(2, 0), Card(2, 1), Card(7, 2), Card(9, 2), Card(3, 2),
            Card(4, 0), Card(5, 0), Card(6, 0))
    s = dataclasses.replace(s, hand=hand,
                            jokers=(JokerState(JokerType.RIDE_THE_BUS, counter=0.0),),
                            required=10_000_000)  # don't clear; keep playing
    s2, _ = step(s, (Verb.PLAY, (0, 1)))           # faceless pair -> counter 1
    assert s2.jokers[0].counter == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/test_engine_jokers.py -v`
Expected: FAIL — `GameState` has no field `jokers`

- [ ] **Step 3: Write minimal implementation**

In `balatro_rl/engine/state.py`, add a field to `GameState` (after `money`):

```python
    jokers: tuple = ()   # tuple[JokerState, ...]; empty until acquired (shop is a later plan)
```

In `balatro_rl/engine/engine.py`:

Add these imports near the top of `engine.py`, alongside the existing `from .` imports:

```python
from .jokers.base import REGISTRY, aggregate_rules
from .hands import evaluate
```

In `reset(...)`, add `jokers=(),` to the `GameState(...)` constructor.

Replace the PLAY scoring call. Currently:

```python
    res = score_play(selected)
```

with joker-aware scoring + the on_play lifecycle (held = the unplayed portion of the hand):

```python
    held = remaining  # cards still in hand (not played) score in the held phase
    res = score_play(selected, jokers=state.jokers, held=tuple(held))
    # Lifecycle: let scaling jokers (e.g. Ride the Bus) update from this hand.
    rules = aggregate_rules(state.jokers) if state.jokers else None
    if state.jokers:
        _, scoring_idx = evaluate(list(selected), rules)
        new_jokers = tuple(
            REGISTRY[js.type].on_play(state, list(selected), list(scoring_idx), rules, js)
            for js in state.jokers
        )
    else:
        new_jokers = state.jokers
```

Then thread `new_jokers` into the returned states. In each `dataclasses.replace(state, ...)` on the PLAY paths (cleared, lost, continue), add `jokers=new_jokers`. For the cleared path, jokers carry over into `_advance_blind` — pass them through by replacing on the pre-advance state:

```python
    if round_score >= state.required:
        carried = dataclasses.replace(state, jokers=new_jokers)
        return _advance_blind(carried, round_score, info)
```

And in the lose / continue paths add `jokers=new_jokers` to the `dataclasses.replace(...)` calls.

> `_advance_blind` already uses `dataclasses.replace(state, ...)`, which preserves `jokers` automatically.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/engine/test_engine_jokers.py tests/engine/test_engine.py -v`
Expected: PASS (new + all Plan-1 engine tests still green)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/state.py balatro_rl/engine/engine.py tests/engine/test_engine_jokers.py
git commit -m "feat(engine): integrate jokers into step scoring + on_play lifecycle"
```

---

### Task 11: Full-suite integration check

**Files:**
- Test: `tests/engine/jokers/test_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/jokers/test_integration.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_additive_left_of_xmult_beats_reverse_order():
    # Joker(+4) then Cavendish(x3) vs Cavendish(x3) then Joker(+4), High-card Ace (16 chips, base mult 1).
    add_then_x = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER), J(JokerType.CAVENDISH)))
    x_then_add = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.CAVENDISH), J(JokerType.JOKER)))
    assert add_then_x.mult == (1 + 4) * 3          # 15
    assert x_then_add.mult == (1 * 3) + 4          # 7
    assert add_then_x.score > x_then_add.score


def test_stacked_jokers_end_to_end():
    # Greedy + Scary Face + Joker on a Pair of K♦.
    # base (10,2)+20 chips=30; Scary +30*2=+60 -> chips 90; Greedy +3*2=+6, Joker +4 -> mult 12.
    res = score_play([C(13, 3), C(13, 3), C(3, 0), C(7, 0), C(9, 0)],
                     jokers=(J(JokerType.GREEDY), J(JokerType.SCARY_FACE), J(JokerType.JOKER)))
    assert res.chips == 90 and res.mult == 12.0 and res.score == 90 * 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/engine/jokers/test_integration.py -v`
Expected: FAIL (until the proof set is complete — should PASS after Tasks 5–10)

- [ ] **Step 3: Implementation**

No new code — this validates the assembled engine. If it fails, fix the responsible joker/pipeline task.

- [ ] **Step 4: Run the FULL suite**

Run: `python3 -m pytest -v`
Expected: ALL tests pass (Plan 1 + Plan 2). Also run `python3 -m balatro_rl.engine 7` — still terminates.

- [ ] **Step 5: Commit**

```bash
git add tests/engine/jokers/test_integration.py
git commit -m "test(jokers): scoring-order + stacked-joker integration checks"
```

---

## Self-Review

**1. Spec coverage (joker-engine architecture from the design spec §3 + the joker hooks):**
- Hook protocol + registry → Tasks 1–2 ✓
- RuleFlags / `evaluate` integration / `is_face` → Task 3 ✓
- Scoring pipeline fold (retrigger, held, independent, +before-×) → Task 4 ✓
- Copy resolution (Blueprint, copyable) → Tasks 2, 9 ✓
- Scaling lifecycle (on_play) → Tasks 2, 8, 10 ✓
- Every proof-set joker (11) with wiki-cited values → Tasks 5–9 ✓
- Engine integration (jokers in `step`) → Task 10 ✓
- **Deferred (correctly):** acquisition/shop, on_round_end economy (incl. Cavendish self-destroy), tarot/spectral generation, enhancements, the other ~139 jokers — later plans.

**2. Placeholder scan:** none — every step has complete code + exact expected values.

**3. Type consistency:** `Effect(chips,mult,xmult)`, `JokerState(type,edition,counter)`, `ScoreResult(score,hand_type,chips,mult,scoring_idx)`, hook signatures (`on_score(ctx,card,index,js)`, `on_play(state,played,scoring_idx,rules,js)`) used identically across Tasks 2–11. Suit ints (♠0 ♥1 ♣2 ♦3) consistent. `score_play(played, jokers=(), held=())` signature consistent across Tasks 4–11 and Task 10's engine call.

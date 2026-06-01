# Tier-0 Engine Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, parity-testable Python simulator of Balatro's base game loop (no jokers/shop/enhancements yet) — shuffle, draw, play/discard, score poker hands, clear blinds across antes 1–8 — behind the clean `reset / legal_actions / step` seam the rest of the project depends on.

**Architecture:** Pure-functional engine. `GameState` is a frozen, plain-data dataclass carrying its own seedable RNG, so `step(state, action) -> (state', info)` is a pure deterministic function — which makes runs reproducible, bugs replayable, and a future Rust port a contained swap. No RL concepts in this layer.

**Tech Stack:** Python ≥3.11, `dataclasses` (frozen, slots), `pytest`. No third-party runtime deps.

**Plan sequence (this is Plan 1 of ~6):**
1. **Tier-0 engine core** ← this plan
2. Tier-0 content (planets, ~30 jokers, shop, economy, full ante loop)
3. RL env (obs encoder, action space + masking, reward modules, single + vec env)
4. Observability (replay viewer, dashboard) — before the learning loop, per the spec's spine principle
5. JAX agent + maskable PPO (training loop)
6. Eval & parity (balatrobot bridge, parity harness, baselines)

**Conventions:** ranks are ints 2–14 (J=11, Q=12, K=13, A=14); suits are ints 0–3 (♠♥♣♦). Run all tests with `python -m pytest`. Commit after every green task (no co-authors in commit messages).

---

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `balatro_rl/__init__.py`
- Create: `balatro_rl/engine/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports():
    import balatro_rl
    assert balatro_rl.__version__ == "0.0.1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "balatro-rl"
version = "0.0.1"
description = "Reinforcement learning agent for Balatro"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["balatro_rl"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# balatro_rl/__init__.py
__version__ = "0.0.1"
```

```python
# balatro_rl/engine/__init__.py
```

```python
# tests/__init__.py
```

Then install in editable mode: `pip install -e ".[dev]"`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml balatro_rl/ tests/
git commit -m "chore: project scaffolding and smoke test"
```

---

### Task 1: Portable, seedable RNG

**Files:**
- Create: `balatro_rl/engine/rng.py`
- Test: `tests/engine/test_rng.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_rng.py
from balatro_rl.engine.rng import RNG


def test_same_seed_same_sequence():
    a = RNG.from_seed(42)
    b = RNG.from_seed(42)
    seq_a, seq_b = [], []
    for _ in range(5):
        x, a = a.randint(0, 100)
        y, b = b.randint(0, 100)
        seq_a.append(x)
        seq_b.append(y)
    assert seq_a == seq_b


def test_different_seeds_differ():
    a = RNG.from_seed(1)
    b = RNG.from_seed(2)
    xa, _ = a.randint(0, 1_000_000)
    xb, _ = b.randint(0, 1_000_000)
    assert xa != xb


def test_random_in_unit_interval():
    rng = RNG.from_seed(7)
    for _ in range(100):
        x, rng = rng.random()
        assert 0.0 <= x < 1.0


def test_randint_inclusive_bounds():
    rng = RNG.from_seed(7)
    seen = set()
    for _ in range(500):
        x, rng = rng.randint(0, 3)
        seen.add(x)
    assert seen == {0, 1, 2, 3}


def test_shuffle_is_deterministic_and_a_permutation():
    items = list(range(10))
    s1, _ = RNG.from_seed(123).shuffle(items)
    s2, _ = RNG.from_seed(123).shuffle(items)
    assert s1 == s2
    assert sorted(s1) == items
    assert s1 != items  # extremely unlikely to be identity for seed 123
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_rng.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.rng'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/rng.py
"""Portable, seedable PRNG (splitmix64), threaded purely as immutable state.

Chosen over Python's `random`/`numpy.random` because splitmix64 is trivial to
re-implement bit-for-bit in Rust/C, which preserves cross-implementation parity
when (if) the hot path is ported. Every call returns a NEW RNG — never mutates.
"""
from __future__ import annotations

import dataclasses

_MASK64 = (1 << 64) - 1


@dataclasses.dataclass(frozen=True, slots=True)
class RNG:
    state: int

    @staticmethod
    def from_seed(seed: int) -> "RNG":
        return RNG(state=seed & _MASK64)

    def _next_u64(self) -> tuple[int, "RNG"]:
        s = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z = (z ^ (z >> 31)) & _MASK64
        return z, RNG(state=s)

    def random(self) -> tuple[float, "RNG"]:
        """Uniform float in [0, 1)."""
        z, rng = self._next_u64()
        return (z >> 11) / float(1 << 53), rng

    def randint(self, lo: int, hi: int) -> tuple[int, "RNG"]:
        """Uniform integer in the inclusive range [lo, hi]."""
        z, rng = self._next_u64()
        n = hi - lo + 1
        return lo + (z % n), rng

    def shuffle(self, items: list) -> tuple[list, "RNG"]:
        """Fisher-Yates. Returns a new shuffled list and the advanced RNG."""
        arr = list(items)
        rng = self
        for i in range(len(arr) - 1, 0, -1):
            j, rng = rng.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr, rng
```

Also create `tests/engine/__init__.py` (empty) so pytest can import the package.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_rng.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/rng.py tests/engine/
git commit -m "feat(engine): portable seedable splitmix64 RNG"
```

---

### Task 2: Card encoding and the standard deck

**Files:**
- Create: `balatro_rl/engine/cards.py`
- Test: `tests/engine/test_cards.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_cards.py
from balatro_rl.engine.cards import Card, rank_chip_value, standard_deck, card_str


def test_rank_chip_values():
    assert rank_chip_value(14) == 11   # Ace
    assert rank_chip_value(13) == 10   # King
    assert rank_chip_value(11) == 10   # Jack
    assert rank_chip_value(10) == 10   # Ten
    assert rank_chip_value(7) == 7
    assert rank_chip_value(2) == 2


def test_standard_deck_is_52_unique_cards():
    deck = standard_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52
    assert all(2 <= c.rank <= 14 for c in deck)
    assert all(0 <= c.suit <= 3 for c in deck)


def test_card_defaults_have_no_modifiers():
    c = Card(rank=14, suit=0)
    assert (c.enhancement, c.edition, c.seal) == (0, 0, 0)


def test_card_str():
    assert card_str(Card(rank=13, suit=0)) == "K♠"   # K♠
    assert card_str(Card(rank=7, suit=1)) == "7♥"    # 7♥
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_cards.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.cards'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/cards.py
"""Playing-card encoding. Plain data so it crosses a future FFI boundary cleanly.

Modifier fields (enhancement/edition/seal) exist now but default to 0 ("none");
they stay unused until the Tier-2 card-modification plan.
"""
from __future__ import annotations

import dataclasses

RANK_MIN, RANK_MAX = 2, 14  # J=11, Q=12, K=13, A=14
SUIT_NAMES = {0: "Spades", 1: "Hearts", 2: "Clubs", 3: "Diamonds"}
_RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}
_SUIT_GLYPH = {0: "♠", 1: "♥", 2: "♣", 3: "♦"}  # ♠♥♣♦


@dataclasses.dataclass(frozen=True, slots=True)
class Card:
    rank: int            # 2..14
    suit: int            # 0..3
    enhancement: int = 0  # 0 = none (Tier-2+)
    edition: int = 0      # 0 = none (Tier-2+)
    seal: int = 0         # 0 = none (Tier-2+)


def rank_chip_value(rank: int) -> int:
    if rank == 14:        # Ace
        return 11
    if rank >= 11:        # J, Q, K
        return 10
    return rank           # 2..10


def standard_deck() -> list[Card]:
    return [Card(rank=r, suit=s)
            for s in range(4)
            for r in range(RANK_MIN, RANK_MAX + 1)]


def card_str(c: Card) -> str:
    r = _RANK_NAMES.get(c.rank, str(c.rank))
    return f"{r}{_SUIT_GLYPH[c.suit]}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_cards.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/cards.py tests/engine/test_cards.py
git commit -m "feat(engine): card encoding and standard 52-card deck"
```

---

### Task 3: Poker-hand detection

**Files:**
- Create: `balatro_rl/engine/hands.py`
- Test: `tests/engine/test_hands.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_hands.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import HandType, HAND_BASE, evaluate


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def test_high_card_scores_only_highest():
    ht, idx = evaluate([C(14), C(7), C(3)])
    assert ht == HandType.HIGH_CARD
    assert idx == [0]  # the Ace


def test_pair_scores_only_the_pair():
    # K K 3 7 9 -> Pair, scoring cards are the two Kings (indices 0,1)
    ht, idx = evaluate([C(13), C(13), C(3), C(7), C(9)])
    assert ht == HandType.PAIR
    assert sorted(idx) == [0, 1]


def test_two_pair():
    ht, idx = evaluate([C(13), C(13), C(7), C(7), C(2)])
    assert ht == HandType.TWO_PAIR
    assert sorted(idx) == [0, 1, 2, 3]


def test_three_of_a_kind_scores_only_the_three():
    ht, idx = evaluate([C(9), C(9), C(9), C(2), C(5)])
    assert ht == HandType.THREE_OF_A_KIND
    assert sorted(idx) == [0, 1, 2]


def test_four_of_a_kind_excludes_kicker():
    ht, idx = evaluate([C(9), C(9), C(9), C(9), C(5)])
    assert ht == HandType.FOUR_OF_A_KIND
    assert sorted(idx) == [0, 1, 2, 3]  # kicker (index 4) does NOT score


def test_full_house_scores_all_five():
    ht, idx = evaluate([C(9), C(9), C(9), C(2), C(2)])
    assert ht == HandType.FULL_HOUSE
    assert sorted(idx) == [0, 1, 2, 3, 4]


def test_flush_scores_all_five():
    ht, idx = evaluate([C(2, 1), C(5, 1), C(7, 1), C(9, 1), C(13, 1)])
    assert ht == HandType.FLUSH
    assert sorted(idx) == [0, 1, 2, 3, 4]


def test_straight_ace_high():
    ht, _ = evaluate([C(10, 0), C(11, 1), C(12, 2), C(13, 3), C(14, 0)])
    assert ht == HandType.STRAIGHT


def test_straight_ace_low():
    ht, _ = evaluate([C(14, 0), C(2, 1), C(3, 2), C(4, 3), C(5, 0)])
    assert ht == HandType.STRAIGHT


def test_straight_flush_beats_straight_and_flush():
    ht, _ = evaluate([C(6, 1), C(7, 1), C(8, 1), C(9, 1), C(10, 1)])
    assert ht == HandType.STRAIGHT_FLUSH


def test_hand_base_table_values():
    assert HAND_BASE[HandType.HIGH_CARD] == (5, 1)
    assert HAND_BASE[HandType.PAIR] == (10, 2)
    assert HAND_BASE[HandType.STRAIGHT] == (30, 4)
    assert HAND_BASE[HandType.FLUSH] == (35, 4)
    assert HAND_BASE[HandType.FOUR_OF_A_KIND] == (60, 7)
    assert HAND_BASE[HandType.STRAIGHT_FLUSH] == (100, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_hands.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.hands'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/hands.py
"""Poker-hand identification for a played subset of <=5 cards.

Returns the best hand type and the indices of the cards that *score* (which
differs by hand: a Four-of-a-Kind's kicker does not score; a Flush scores all).
Secret hands (Five of a Kind, Flush House, Flush Five) are encoded and reachable
only once duplicate ranks/wilds exist (Tier-2+); the base 52-card deck can't form
them, but the detection is here so scoring never needs to change later.
"""
from __future__ import annotations

from collections import Counter
from enum import IntEnum

from .cards import Card


class HandType(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    FIVE_OF_A_KIND = 9
    FLUSH_HOUSE = 10
    FLUSH_FIVE = 11


# (base_chips, base_mult) at level 1.
HAND_BASE: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_OF_A_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_OF_A_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}


def _is_straight(ranks: list[int]) -> bool:
    u = sorted(set(ranks))
    if len(u) != 5:
        return False
    if u[-1] - u[0] == 4:
        return True
    return u == [2, 3, 4, 5, 14]  # Ace-low: A-2-3-4-5


def evaluate(cards: list[Card]) -> tuple[HandType, list[int]]:
    """Best (HandType, scoring-card indices) for 1..5 played cards."""
    n = len(cards)
    if n == 0:
        raise ValueError("evaluate() requires at least one card")
    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    is_flush = n == 5 and len(set(suits)) == 1
    is_straight = n == 5 and _is_straight(ranks)
    all_idx = list(range(n))

    def idx_with_count(k: int) -> list[int]:
        targets = {r for r, c in rank_counts.items() if c == k}
        return [i for i, r in enumerate(ranks) if r in targets]

    if is_flush and counts == [5]:
        return HandType.FLUSH_FIVE, all_idx
    if is_flush and counts == [3, 2]:
        return HandType.FLUSH_HOUSE, all_idx
    if counts == [5]:
        return HandType.FIVE_OF_A_KIND, all_idx
    if is_flush and is_straight:
        return HandType.STRAIGHT_FLUSH, all_idx
    if counts and counts[0] == 4:
        return HandType.FOUR_OF_A_KIND, idx_with_count(4)
    if counts == [3, 2]:
        return HandType.FULL_HOUSE, all_idx
    if is_flush:
        return HandType.FLUSH, all_idx
    if is_straight:
        return HandType.STRAIGHT, all_idx
    if counts and counts[0] == 3:
        return HandType.THREE_OF_A_KIND, idx_with_count(3)
    if counts[:2] == [2, 2]:
        return HandType.TWO_PAIR, idx_with_count(2)
    if counts and counts[0] == 2:
        return HandType.PAIR, idx_with_count(2)
    hi = max(range(n), key=lambda i: ranks[i])
    return HandType.HIGH_CARD, [hi]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_hands.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/hands.py tests/engine/test_hands.py
git commit -m "feat(engine): poker-hand detection and base value table"
```

---

### Task 4: Base scoring pipeline

**Files:**
- Create: `balatro_rl/engine/scoring.py`
- Test: `tests/engine/test_scoring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_scoring.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.scoring import score_play


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def test_pair_of_kings():
    # base (10,2); scoring cards = two Kings -> chips 10 + 10 + 10 = 30; 30*2 = 60
    res = score_play([C(13), C(13), C(3), C(7), C(9)])
    assert res.hand_type == HandType.PAIR
    assert res.chips == 30
    assert res.mult == 2
    assert res.score == 60


def test_high_card_ace():
    # base (5,1); Ace chips 11 -> chips 16; 16*1 = 16
    res = score_play([C(14), C(7), C(2)])
    assert res.hand_type == HandType.HIGH_CARD
    assert res.chips == 16
    assert res.mult == 1
    assert res.score == 16


def test_flush_all_cards_score():
    # base (35,4); all five score: 2+5+7+9+10(K? use 10) chips
    res = score_play([C(2, 1), C(5, 1), C(7, 1), C(9, 1), C(10, 1)])
    assert res.hand_type == HandType.FLUSH
    assert res.chips == 35 + (2 + 5 + 7 + 9 + 10)  # 68
    assert res.mult == 4
    assert res.score == 68 * 4


def test_four_of_a_kind_excludes_kicker_chips():
    # base (60,7); four 9s score (9*4=36), kicker King does NOT add chips
    res = score_play([C(9), C(9), C(9), C(9), C(13)])
    assert res.hand_type == HandType.FOUR_OF_A_KIND
    assert res.chips == 60 + 36
    assert res.score == (60 + 36) * 7


def test_scoring_idx_recorded():
    res = score_play([C(13), C(13), C(3)])
    assert sorted(res.scoring_idx) == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.scoring'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/scoring.py
"""Base scoring pipeline: hand base value + scoring-card chips, then chips x mult.

Tier-0 only — no jokers, held cards, enhancements, editions, seals, or hand
levels (all level 1). Later plans extend THIS function's pipeline; the structured
ScoreResult is what the replay viewer renders as a score breakdown.
"""
from __future__ import annotations

import dataclasses

from .cards import Card, rank_chip_value
from .hands import HAND_BASE, HandType, evaluate


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreResult:
    score: int
    hand_type: HandType
    chips: int
    mult: int
    scoring_idx: list[int]


def score_play(cards: list[Card]) -> ScoreResult:
    hand_type, scoring_idx = evaluate(cards)
    base_chips, base_mult = HAND_BASE[hand_type]
    chips = base_chips + sum(rank_chip_value(cards[i].rank) for i in scoring_idx)
    mult = base_mult
    return ScoreResult(
        score=chips * mult,
        hand_type=hand_type,
        chips=chips,
        mult=mult,
        scoring_idx=scoring_idx,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_scoring.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/scoring.py tests/engine/test_scoring.py
git commit -m "feat(engine): base scoring pipeline (chips x mult)"
```

---

### Task 5: Blind score requirements

**Files:**
- Create: `balatro_rl/engine/blinds.py`
- Test: `tests/engine/test_blinds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_blinds.py
import pytest

from balatro_rl.engine.blinds import required_score, ANTE_BASE, BLIND_MULT


def test_ante1_blinds():
    assert required_score(1, 0) == 300   # small  = 1.0x
    assert required_score(1, 1) == 450   # big    = 1.5x
    assert required_score(1, 2) == 600   # boss   = 2.0x


def test_ante8_boss():
    assert required_score(8, 2) == 100_000  # 50_000 * 2.0


def test_base_table_matches_spec():
    assert ANTE_BASE == {1: 300, 2: 800, 3: 2000, 4: 5000,
                         5: 11000, 6: 20000, 7: 35000, 8: 50000}
    assert BLIND_MULT == (1.0, 1.5, 2.0)


def test_invalid_blind_index_raises():
    with pytest.raises(IndexError):
        required_score(1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_blinds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.blinds'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/blinds.py
"""Blind score requirements for antes 1-8 (White-stake base values).

required = ANTE_BASE[ante] * BLIND_MULT[blind_index], floored.
Boss-blind *effects* (debuffs) and higher-stake scaling arrive in later plans;
Tier-0 uses only the score requirement and the score-multiplier bosses implicitly
(via the 2.0x boss requirement).
"""
from __future__ import annotations

ANTE_BASE: dict[int, int] = {
    1: 300, 2: 800, 3: 2000, 4: 5000,
    5: 11000, 6: 20000, 7: 35000, 8: 50000,
}
BLIND_MULT: tuple[float, float, float] = (1.0, 1.5, 2.0)  # small, big, boss


def required_score(ante: int, blind_index: int) -> int:
    return int(ANTE_BASE[ante] * BLIND_MULT[blind_index])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_blinds.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/blinds.py tests/engine/test_blinds.py
git commit -m "feat(engine): blind score requirements for antes 1-8"
```

---

### Task 6: Game state (plain data, carries its RNG)

**Files:**
- Create: `balatro_rl/engine/state.py`
- Test: `tests/engine/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_state.py
import dataclasses

import pytest

from balatro_rl.engine.cards import Card
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.state import GameState, Phase


def make_state(**overrides) -> GameState:
    base = dict(
        deck=(Card(2, 0),), hand=(Card(3, 0),),
        ante=1, blind_index=0, round_score=0, required=300,
        hands_left=4, discards_left=3, hand_size=8,
        levels=tuple([1] * 12), money=4,
        rng=RNG.from_seed(0), phase=Phase.PLAYING, done=False, won=False,
    )
    base.update(overrides)
    return GameState(**base)


def test_state_is_frozen():
    s = make_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.money = 99


def test_levels_has_twelve_entries():
    s = make_state()
    assert len(s.levels) == 12


def test_replace_produces_new_state():
    s = make_state(money=4)
    s2 = dataclasses.replace(s, money=10)
    assert s.money == 4 and s2.money == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.state'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/state.py
"""GameState: frozen, plain-data, carries its own RNG so step() is pure.

POD by design (tuples + scalars + a frozen RNG, no object graphs) so it crosses
a future FFI boundary cleanly and a replay is fully reconstructable from a seed.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

from .cards import Card
from .rng import RNG


class Phase(IntEnum):
    PLAYING = 0
    WON = 1
    LOST = 2


@dataclasses.dataclass(frozen=True, slots=True)
class GameState:
    deck: tuple[Card, ...]      # remaining draw pile (front = next to draw)
    hand: tuple[Card, ...]      # current hand
    ante: int
    blind_index: int            # 0 = small, 1 = big, 2 = boss
    round_score: int            # chips scored so far this blind
    required: int               # score needed to clear this blind
    hands_left: int
    discards_left: int
    hand_size: int
    levels: tuple[int, ...]     # 12 hand-type levels (HandType order)
    money: int
    rng: RNG
    phase: Phase
    done: bool
    won: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_state.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/state.py tests/engine/test_state.py
git commit -m "feat(engine): frozen plain-data GameState with embedded RNG"
```

---

### Task 7: The engine — reset / legal_actions / step

**Files:**
- Create: `balatro_rl/engine/engine.py`
- Test: `tests/engine/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_engine.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, legal_actions, step
from balatro_rl.engine.state import Phase


def test_reset_initial_conditions():
    s = reset(seed=1)
    assert len(s.hand) == 8
    assert s.ante == 1 and s.blind_index == 0
    assert s.required == 300
    assert s.hands_left == 4 and s.discards_left == 3
    assert s.phase == Phase.PLAYING and not s.done


def test_reset_is_deterministic():
    assert reset(seed=5).hand == reset(seed=5).hand
    assert reset(seed=5).hand != reset(seed=6).hand


def test_legal_actions_present_and_bounded():
    s = reset(seed=1)
    acts = legal_actions(s)
    assert len(acts) > 0
    for verb, idx in acts:
        assert verb in (Verb.PLAY, Verb.DISCARD)
        assert 1 <= len(idx) <= 5
        assert len(set(idx)) == len(idx)


def test_discard_consumes_a_discard_and_refills_hand():
    s = reset(seed=1)
    s2, info = step(s, (Verb.DISCARD, (0, 1)))
    assert info["verb"] == "discard"
    assert s2.discards_left == s.discards_left - 1
    assert len(s2.hand) == 8


def test_play_consumes_a_hand_and_adds_score():
    s = reset(seed=1)
    s2, info = step(s, (Verb.PLAY, (0,)))
    assert info["verb"] == "play"
    assert s2.hands_left == s.hands_left - 1
    assert s2.round_score == info["score"]
    assert len(s2.hand) == 8


def test_clearing_a_blind_advances_and_resets_counters():
    # Force a clear by handing the engine a state already at the threshold-1
    # via a high-scoring play. Use a constructed hand of four-of-a-kind Kings.
    import dataclasses
    s = reset(seed=1)
    big_hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
                Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=big_hand, required=10)  # trivially clearable
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert info.get("cleared") is True
    assert s2.blind_index == 1            # advanced small -> big
    assert s2.round_score == 0            # reset for the new blind
    assert s2.hands_left == 4 and s2.discards_left == 3
    assert len(s2.hand) == 8


def test_losing_when_hands_exhausted_without_clearing():
    import dataclasses
    s = reset(seed=1)
    s = dataclasses.replace(s, hands_left=1, required=10_000_000, round_score=0)
    s2, info = step(s, (Verb.PLAY, (0,)))   # tiny score, can't clear, no hands left
    assert s2.done and not s2.won
    assert s2.phase == Phase.LOST


def test_winning_after_clearing_ante8_boss():
    import dataclasses
    s = reset(seed=1)
    s = dataclasses.replace(s, ante=8, blind_index=2, required=10,
                            hand=(Card(14, 0),) + reset(seed=1).hand[1:])
    s2, info = step(s, (Verb.PLAY, (0,)))
    assert s2.done and s2.won
    assert s2.phase == Phase.WON
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/engine.py
"""The Tier-0 engine seam: reset / legal_actions / step.

step(state, action) -> (state', info) is a pure function (RNG rides inside the
state). Action = (Verb, tuple-of-hand-indices). The flat-id encoding + legal mask
used by the RL agent live in the env layer (Plan 3); here we use plain tuples.

Tier-0 has no shop: clearing a blind advances directly to the next blind with a
freshly shuffled deck and full hand. Shop/economy arrive in Plan 2.
"""
from __future__ import annotations

import dataclasses
import itertools
from enum import IntEnum

from .blinds import required_score
from .cards import standard_deck
from .scoring import score_play
from .state import GameState, Phase

STARTING_MONEY = 4
HANDS_PER_BLIND = 4
DISCARDS_PER_BLIND = 3
HAND_SIZE = 8
MAX_SELECT = 5


class Verb(IntEnum):
    PLAY = 0
    DISCARD = 1


def _draw(hand: list, deck: list, hand_size: int) -> tuple[list, list]:
    """Draw from the front of the (pre-shuffled) deck up to hand_size."""
    need = max(0, hand_size - len(hand))
    drawn = deck[:need]
    return hand + drawn, deck[need:]


def reset(seed: int) -> GameState:
    from .rng import RNG
    rng = RNG.from_seed(seed)
    deck, rng = rng.shuffle(standard_deck())
    hand, deck = _draw([], deck, HAND_SIZE)
    return GameState(
        deck=tuple(deck), hand=tuple(hand), ante=1, blind_index=0,
        round_score=0, required=required_score(1, 0),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND,
        hand_size=HAND_SIZE, levels=tuple([1] * 12), money=STARTING_MONEY,
        rng=rng, phase=Phase.PLAYING, done=False, won=False,
    )


def legal_actions(state: GameState) -> list[tuple[Verb, tuple[int, ...]]]:
    if state.done:
        return []
    actions: list[tuple[Verb, tuple[int, ...]]] = []
    n = len(state.hand)
    for size in range(1, min(MAX_SELECT, n) + 1):
        for combo in itertools.combinations(range(n), size):
            if state.hands_left > 0:
                actions.append((Verb.PLAY, combo))
            if state.discards_left > 0:
                actions.append((Verb.DISCARD, combo))
    return actions


def _advance_blind(state: GameState, round_score: int, info: dict):
    if state.blind_index < 2:
        new_ante, new_blind = state.ante, state.blind_index + 1
    else:
        new_ante, new_blind = state.ante + 1, 0
    if new_ante > 8:
        won = dataclasses.replace(state, round_score=round_score,
                                  done=True, won=True, phase=Phase.WON)
        return won, {**info, "cleared": True, "result": "won"}
    deck, rng = state.rng.shuffle(standard_deck())
    hand, deck = _draw([], deck, state.hand_size)
    nxt = dataclasses.replace(
        state, ante=new_ante, blind_index=new_blind,
        deck=tuple(deck), hand=tuple(hand), round_score=0,
        required=required_score(new_ante, new_blind),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND, rng=rng,
    )
    return nxt, {**info, "cleared": True, "result": "blind_cleared"}


def step(state: GameState, action: tuple[Verb, tuple[int, ...]]):
    assert not state.done, "step() called on a terminal state"
    verb, idx = action
    assert 1 <= len(idx) <= MAX_SELECT, "must select 1..5 cards"
    assert len(set(idx)) == len(idx), "duplicate card indices"
    assert all(0 <= i < len(state.hand) for i in idx), "index out of range"

    selected = [state.hand[i] for i in idx]
    chosen = set(idx)
    remaining = [c for i, c in enumerate(state.hand) if i not in chosen]

    if verb == Verb.DISCARD:
        assert state.discards_left > 0, "no discards left"
        hand, deck = _draw(remaining, list(state.deck), state.hand_size)
        nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                  discards_left=state.discards_left - 1)
        return nxt, {"verb": "discard", "discarded": len(idx)}

    # PLAY
    assert state.hands_left > 0, "no hands left"
    res = score_play(selected)
    round_score = state.round_score + res.score
    hands_left = state.hands_left - 1
    info = {"verb": "play", "score": res.score, "hand_type": int(res.hand_type),
            "chips": res.chips, "mult": res.mult}
    hand, deck = _draw(remaining, list(state.deck), state.hand_size)

    if round_score >= state.required:
        cleared_state = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                            hands_left=hands_left)
        return _advance_blind(cleared_state, round_score, info)

    if hands_left <= 0:
        lost = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                   round_score=round_score, hands_left=0,
                                   done=True, won=False, phase=Phase.LOST)
        return lost, {**info, "result": "lost"}

    nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                              round_score=round_score, hands_left=hands_left)
    return nxt, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_engine.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/engine.py tests/engine/test_engine.py
git commit -m "feat(engine): reset/legal_actions/step game loop for antes 1-8"
```

---

### Task 8: Text render + deterministic random playthrough

**Files:**
- Create: `balatro_rl/engine/render.py`
- Create: `balatro_rl/engine/__main__.py`
- Test: `tests/engine/test_playthrough.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_playthrough.py
from balatro_rl.engine.render import render
from balatro_rl.engine.engine import reset
from balatro_rl.engine.__main__ import play_random


def test_render_contains_key_fields():
    s = reset(seed=1)
    text = render(s)
    assert "Ante 1" in text
    assert "/300" in text
    assert "hand:" in text


def test_random_playthrough_terminates_and_is_deterministic():
    a = play_random(seed=3, verbose=False)
    b = play_random(seed=3, verbose=False)
    assert a.done
    assert a.won == b.won
    assert a.ante == b.ante
    assert a.round_score == b.round_score


def test_two_seeds_can_differ():
    # Not guaranteed identical; just ensure both terminate cleanly.
    a = play_random(seed=1, verbose=False)
    b = play_random(seed=2, verbose=False)
    assert a.done and b.done
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/engine/test_playthrough.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'balatro_rl.engine.render'`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/engine/render.py
"""Minimal ASCII render of a GameState — the seed of the Plan-4 replay viewer."""
from __future__ import annotations

from .cards import card_str
from .state import GameState


def render(state: GameState) -> str:
    hand = " ".join(card_str(c) for c in state.hand)
    head = (f"Ante {state.ante} blind {state.blind_index}  "
            f"score {state.round_score}/{state.required}  "
            f"hands {state.hands_left} discards {state.discards_left}  ${state.money}")
    return f"{head}\nhand: {hand}"
```

```python
# balatro_rl/engine/__main__.py
"""Play a deterministic random game: `python -m balatro_rl.engine [seed]`.

Uses a SEPARATE RNG (seeded from the game seed) to choose actions, so the run is
fully reproducible without touching the engine's own RNG stream.
"""
from __future__ import annotations

import sys

from .engine import legal_actions, reset, step
from .render import render
from .rng import RNG
from .state import GameState

_MAX_STEPS = 10_000


def play_random(seed: int = 0, verbose: bool = True) -> GameState:
    state = reset(seed)
    chooser = RNG.from_seed(seed ^ 0xABCDEF)
    for _ in range(_MAX_STEPS):
        if state.done:
            break
        acts = legal_actions(state)
        i, chooser = chooser.randint(0, len(acts) - 1)
        state, info = step(state, acts[i])
        if verbose and info.get("verb") == "play":
            print(render(state))
    return state


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    final = play_random(seed)
    print("RESULT:", "WON" if final.won else "LOST", "| ante", final.ante)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/engine/test_playthrough.py -v`
Then sanity-check the CLI: `python -m balatro_rl.engine 1`
Expected: tests PASS (3 tests); CLI prints play lines and a final `RESULT: ... | ante N`.

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/engine/render.py balatro_rl/engine/__main__.py tests/engine/test_playthrough.py
git commit -m "feat(engine): ASCII render and deterministic random playthrough"
```

---

### Final verification

- [ ] **Run the whole suite**

Run: `python -m pytest -v`
Expected: all tests PASS (Tasks 0–8).

- [ ] **Confirm a full game simulates end-to-end**

Run: `python -m balatro_rl.engine 7`
Expected: prints play-by-play and terminates with a WON/LOST result.

---

## Self-Review

**1. Spec coverage (engine-core slice of the spec §3 `engine/`):**
- `rng.py` (portable explicit RNG) → Task 1 ✓
- `cards.py` (card int-encoding) → Task 2 ✓
- `hands.py` (poker detection + base tables) → Task 3 ✓
- `scoring.py` (base pipeline) → Task 4 ✓
- `blinds.py` (antes/requirements) → Task 5 ✓
- `state.py` (POD state) → Task 6 ✓
- `engine.py` (reset/legal_actions/step seam) → Task 7 ✓
- Deterministic replay seed property (spec §7.4) → established by Tasks 1, 6, 8 ✓
- **Deferred to later plans (correctly out of scope):** jokers/planets/shop/economy (Plan 2), enhancements/editions/seals (Plan 2/Tier-2), obs/action-mask/reward (Plan 3), boss-blind *effects* (Plan 2+). These are explicitly Tier-0-excluded in the spec.

**2. Placeholder scan:** No TBD/TODO; every code and test step contains complete, runnable content. ✓

**3. Type consistency:** `evaluate -> (HandType, list[int])` used identically in Tasks 3–4; `ScoreResult` fields (`score/hand_type/chips/mult/scoring_idx`) consistent between Task 4 and Task 7; `GameState` field names identical across Tasks 6–8; `Verb`/`(Verb, tuple)` action shape identical across Task 7–8. ✓

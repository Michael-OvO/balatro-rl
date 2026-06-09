# E7 Phase 2: JAX Joker Scoring Kernel + Env Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a branchless, `jit`/`vmap`-able JAX joker scoring kernel that reproduces the Python oracle `engine.scoring.score_play` for ~45 pure-scoring jokers on plain cards, wired into the core engine via a fixed per-episode loadout, proven bit-for-bit at both the scoring-kernel and full-episode levels.

**Architecture:** A new `engine_jax/jokers.py` holds the dense-id map, four `lax.switch` branch tables (`on_score`, `on_held`, `independent`, `retrigger`) — one tiny pure branch per in-scope joker — and `score_with_jokers(...)`, the exact ordered fold (base → scored cards L→R with retriggers folding `on_score` → held → independent in slot order → `floor(chips×mult)`) that reduces to Phase-1 `score_core` when the loadout is empty. `engine_jax/scoring.py` only gains a shared `_scoring_mask` helper (extracted from `score_core`, no behavior change). `CoreState` gains a `jokers` field; `step`/`reset` thread a fixed loadout; `obs` fills the joker keys. Two parity gates (component vs `score_play`; fixed-loadout episode) plus a PPO smoke and a benchmark.

**Tech Stack:** Python 3.11, JAX (CPU backend for dev/test via `JAX_PLATFORMS=cpu`), `jax.lax.switch` for branchless per-joker dispatch, `chex`/`pytest`. Oracle: `balatro_rl/engine/` (Python), unchanged.

**Spec:** `docs/specs/2026-06-09-jax-engine-phase02-jokers-design.md`. Read it before starting.

---

## Conventions (read once)

- **Run tests CPU-only:** prefix every pytest invocation with `JAX_PLATFORMS=cpu`.
- **Suit ids** (from `engine/cards.py`, confirmed in `library.py`): `0=Spade, 1=Heart, 2=Club, 3=Diamond`.
- **Rank chip value** (`engine/cards.py::rank_chip_value`, already mirrored by `scoring._card_chip`): Ace(14)→11, J/Q/K(11/12/13)→10, else rank.
- **Face card:** rank ∈ {11,12,13}, OR `all_face` (Pareidolia). Ace is NOT a face.
- **`MAX_SELECT=5`** (played slots), **`MAX_HAND=8`** (hand/held slots), **`MAX_JOKERS=6`** (loadout array; `JOKER_SLOTS=5` is the real cap, the 6th is the Antimatter slot — loadouts fill ≤5), all in `engine_jax/config.py` / `envs/actions.py`.
- **No co-authors** in any commit (repo rule).
- **Each task ends green:** the listed pytest must pass before committing.

---

## File Structure

- **Create `balatro_rl/engine_jax/jokers.py`** — joker id constants, the dense-id map, the four branch tables, and the `score_with_jokers` fold (kept here, not in `scoring.py`, so `scoring.py` stays the small plain-card kernel and the joker dispatch lives in one focused file).
- **Modify `balatro_rl/engine_jax/state.py`** — add `CoreState.jokers: int32[MAX_JOKERS]` + `zeros_state`.
- **Modify `balatro_rl/engine_jax/scoring.py`** — extract a reusable `_scoring_mask(ht, ranks, suits, mask)` helper (the per-hand-type scoring-card mask) so both `score_core` and `score_with_jokers` share it; no behavior change.
- **Modify `balatro_rl/engine_jax/step.py`** — `reset`/`reset_jax`/`batched_reset` accept `jokers`; `step` computes held + context and calls `score_with_jokers`; `step_with_action` preserves the loadout on auto-reset.
- **Modify `balatro_rl/engine_jax/obs.py`** — fill `joker_types`/`joker_counter`/`joker_mask` + `global[10]`.
- **Modify `balatro_rl/envs/jax_vec_env.py`**, **`balatro_rl/agent/train.py`** — `joker_loadout` knob.
- **Create `tests/engine_jax/test_joker_scoring_parity.py`** — Gate A (component parity).
- **Modify `tests/engine_jax/test_core_parity_gate.py`** — Gate B (fixed-loadout episode parity) + empty-loadout regression.
- **Modify `tests/agent/test_jax_engine_smoke.py`** — PPO smoke with a loadout.
- **Modify `scripts/bench_jax_engine.py`**, **`docs/RUNPOD_M2.md`** — benchmark + docs.

---

## Task 2.1: `CoreState.jokers` + reset wiring + empty-loadout invariance

**Files:**
- Modify: `balatro_rl/engine_jax/state.py`
- Modify: `balatro_rl/engine_jax/step.py` (`reset`, `reset_jax`, `step_with_action`, `batched_reset`)
- Test: `tests/engine_jax/test_state.py`, `tests/engine_jax/test_batched.py`

- [ ] **Step 1: Write the failing test** — append to `tests/engine_jax/test_state.py`:

```python
def test_corestate_has_jokers_field():
    from balatro_rl.engine_jax.state import zeros_state
    import jax.numpy as jnp
    from balatro_rl.envs.actions import MAX_JOKERS
    s = zeros_state()
    assert hasattr(s, "jokers")
    assert s.jokers.shape == (MAX_JOKERS,)
    assert s.jokers.dtype == jnp.int32
    assert int(jnp.sum(s.jokers)) == 0  # empty by default
```

- [ ] **Step 2: Run it; expect FAIL** — `JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_state.py::test_corestate_has_jokers_field -v` → AttributeError: no `jokers`.

- [ ] **Step 3: Add the field.** In `state.py`, import `MAX_JOKERS`:

```python
from balatro_rl.envs.actions import MAX_JOKERS
```

Add to `CoreState` (after the RNG block or near the deck block — pick a stable position; put it right after `hand_mask`):

```python
    # -- Jokers (fixed per-episode loadout; acquisition is Phase 3) ------------
    jokers: jnp.ndarray             # int32[MAX_JOKERS]  JokerType id per slot, 0 = empty
```

Add to `zeros_state()`:

```python
        jokers=jnp.zeros((MAX_JOKERS,), dtype=jnp.int32),
```

- [ ] **Step 4: Thread `jokers` through `reset`, `reset_jax`, `step_with_action`, `batched_reset` in `step.py`.**

In `reset(...)` add a kwarg `jokers=None` and build the array:

```python
def reset(deck_rank, deck_suit, required, required_table=None, scale_unused=1.0, jokers=None):
    ...
    from balatro_rl.envs.actions import MAX_JOKERS
    jk = (jnp.zeros((MAX_JOKERS,), dtype=jnp.int32) if jokers is None
          else jnp.asarray(jokers, dtype=jnp.int32))
    return CoreState(..., jokers=jk, ...)   # add jokers= to the constructor call
```

In `reset_jax(key, required_table, jokers=None)` do the same (default zeros) and pass `jokers=jk` into the `CoreState(...)`. Update its docstring's signature line.

In `step_with_action`, the auto-reset must preserve the loadout — change:

```python
    fresh_after = reset_jax(next_state.rng, next_state.required_table, next_state.jokers)
```

Update the batched reset to map a per-env loadout array:

```python
# batched_reset: vmap over keys AND per-env jokers; required_table broadcast.
batched_reset = jax.vmap(reset_jax, in_axes=(0, None, 0))
```

- [ ] **Step 5: Add a regression test** to `tests/engine_jax/test_batched.py` proving an empty loadout leaves Phase-1 behavior identical:

```python
def test_batched_reset_accepts_jokers_and_empty_is_phase1():
    import jax, jax.numpy as jnp
    from balatro_rl.engine_jax.step import batched_reset, reset_jax
    from balatro_rl.engine_jax.curriculum import build_required_table
    from balatro_rl.envs.actions import MAX_JOKERS
    n = 4
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    rt = build_required_table(1.0)
    jk = jnp.zeros((n, MAX_JOKERS), dtype=jnp.int32)
    st = batched_reset(keys, rt, jk)
    assert st.jokers.shape == (n, MAX_JOKERS)
    # Single-env reset_jax with default jokers matches the all-zero loadout.
    s0 = reset_jax(keys[0], rt)
    assert int(jnp.sum(s0.jokers)) == 0
```

- [ ] **Step 6: Run all touched tests; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_state.py tests/engine_jax/test_batched.py -v
```

- [ ] **Step 7: Run the FULL Phase-1 suite to prove no regression** (reset/step/obs/reward/gate still green — `jokers` is additive):

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax -q
```

Expected: all pass (the new field is unused by `step` yet; existing tests that build `CoreState` via `reset*`/`zeros_state` get the zero loadout automatically).

> If any existing test constructs `CoreState(...)` positionally/explicitly (not via `reset*`/`zeros_state`), it will now need the `jokers=` field. Grep `rg "CoreState\("` under `tests/` and fix those call sites by adding `jokers=jnp.zeros((MAX_JOKERS,), dtype=jnp.int32)`.

- [ ] **Step 8: Commit.**

```bash
git add balatro_rl/engine_jax/state.py balatro_rl/engine_jax/step.py tests/engine_jax/test_state.py tests/engine_jax/test_batched.py
git commit -m "engine_jax: add CoreState.jokers (fixed loadout) + reset/auto-reset wiring"
```

---

## Task 2.2: Joker module scaffold — ids, dense map, branch-table infra, empty fold

This task builds `jokers.py` with the dispatch infrastructure and a `score_with_jokers` that, with NO branches wired (every dense slot is a no-op), already reduces to `score_core`. It proves the fold skeleton + the float/int accumulator + the empty-loadout reduction before any joker math exists.

**Files:**
- Modify: `balatro_rl/engine_jax/scoring.py` (extract `_scoring_mask` helper)
- Create: `balatro_rl/engine_jax/jokers.py`
- Test: `tests/engine_jax/test_joker_scoring_parity.py` (new)

- [ ] **Step 1: Extract `_scoring_mask` in `scoring.py`** (pure refactor — no behavior change). Pull the scoring-mask construction out of `score_core` into a helper both kernels call:

```python
def _scoring_mask(ht, ranks, suits, m):
    """The per-hand-type scoring-card mask (bool[5]) matching evaluate()'s scoring_idx.
    `ht` is the detected hand type; `m` is the played bool[5] mask."""
    ranks = jnp.asarray(ranks).astype(jnp.int32)
    m = jnp.asarray(m).astype(jnp.bool_)
    mi = m.astype(jnp.int32)
    rank_bucket = ranks - 2
    rank_oh = (rank_bucket[:, None] == jnp.arange(_N_RANK_BUCKETS)[None, :]).astype(jnp.int32) * mi[:, None]
    rank_counts = jnp.sum(rank_oh, axis=0)
    safe_bucket = jnp.clip(rank_bucket, 0, _N_RANK_BUCKETS - 1)
    slot_rank_count = rank_counts[safe_bucket] * mi
    mask_eq2 = (slot_rank_count == 2) & m
    mask_eq3 = (slot_rank_count == 3) & m
    mask_eq4 = (slot_rank_count == 4) & m
    valid_rank = jnp.where(m, ranks, jnp.int32(-1))
    hi_rank = jnp.max(valid_rank)
    is_hi = (valid_rank == hi_rank) & m
    first_hi = jnp.argmax(is_hi.astype(jnp.int32))
    mask_high = jnp.zeros_like(m).at[first_hi].set(True) & m
    mask_all = m
    is_pair_like2 = (ht == PAIR) | (ht == TWO_PAIR)
    is_all = ((ht == STRAIGHT) | (ht == FLUSH) | (ht == FULL_HOUSE)
              | (ht == STRAIGHT_FLUSH) | (ht == FIVE_OF_A_KIND)
              | (ht == FLUSH_HOUSE) | (ht == FLUSH_FIVE))
    sm = mask_high
    sm = jnp.where(is_pair_like2, mask_eq2, sm)
    sm = jnp.where(ht == THREE_OF_A_KIND, mask_eq3, sm)
    sm = jnp.where(ht == FOUR_OF_A_KIND, mask_eq4, sm)
    sm = jnp.where(is_all, mask_all, sm)
    return sm
```

Then rewrite `score_core`'s body to call it (`scoring_mask = _scoring_mask(ht, ranks, suits, m)`), deleting the now-duplicated inline block. Run `JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_scoring.py tests/engine_jax/test_core_parity_gate.py -q` → still PASS (pure refactor).

- [ ] **Step 2: Write the failing test** — create `tests/engine_jax/test_joker_scoring_parity.py`:

```python
"""Gate A: component parity of score_with_jokers vs engine.scoring.score_play."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from balatro_rl.engine_jax.jokers import score_with_jokers
from balatro_rl.engine_jax.scoring import score_core


def _empty_loadout():
    from balatro_rl.envs.actions import MAX_JOKERS
    return jnp.zeros((MAX_JOKERS,), dtype=jnp.int32)


def _pad5(ranks, suits):
    r = np.zeros(5, np.int8); s = np.zeros(5, np.int8); m = np.zeros(5, bool)
    for i, (rk, su) in enumerate(zip(ranks, suits)):
        r[i] = rk; s[i] = su; m[i] = True
    return jnp.asarray(r), jnp.asarray(s), jnp.asarray(m)


def test_empty_loadout_reduces_to_score_core():
    """With no jokers, score_with_jokers == score_core for random plain hands."""
    rng = np.random.default_rng(0)
    h_r = jnp.zeros(8, jnp.int8); h_s = jnp.zeros(8, jnp.int8); h_m = jnp.zeros(8, bool)
    levels = jnp.ones(12, jnp.int32)
    jk = _empty_loadout()
    for _ in range(300):
        n = rng.integers(1, 6)
        ranks = rng.integers(2, 15, size=n); suits = rng.integers(0, 4, size=n)
        pr, ps, pm = _pad5(ranks, suits)
        ht0, c0, m0, sc0 = score_core(pr, ps, pm, levels)
        ht1, c1, m1, sc1 = score_with_jokers(
            pr, ps, pm, h_r, h_s, h_m, levels, jk,
            money=jnp.int32(0), discards_left=jnp.int32(0), deck_count=jnp.int32(0),
            hand_plays_run=jnp.zeros(12, jnp.int32), hand_plays_round=jnp.zeros(12, jnp.int32))
        assert (int(ht0), int(c0), int(sc0)) == (int(ht1), int(c1), int(sc1))
        assert int(sc1) == int(jnp.floor(c1.astype(jnp.float32) * m1))
```

- [ ] **Step 3: Run it; expect FAIL** — `ModuleNotFoundError: balatro_rl.engine_jax.jokers`.

- [ ] **Step 4: Create `balatro_rl/engine_jax/jokers.py`** with the infra + the empty fold. Joker id constants mirror `engine/jokers/base.py::JokerType`; the dense map sends every id to a no-op until later tasks register branches.

```python
"""Branchless JAX joker scoring (Phase 2).

`score_with_jokers` reproduces engine.scoring.score_play's ordered fold for the
~45 pure-scoring jokers on plain cards. Dispatch is four lax.switch branch tables
(on_score / on_held / independent / retrigger), one tiny pure branch per in-scope
joker, indexed by a dense id (0 = empty/out-of-scope = no-op). With an all-zero
loadout the fold reduces exactly to scoring.score_core.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple

from balatro_rl.engine_jax.config import MAX_HAND, MAX_SELECT
from balatro_rl.engine_jax.scoring import (
    detect_hand_type, _card_chip, _scoring_mask, _N_RANK_BUCKETS,
    HAND_BASE_CHIPS, HAND_BASE_MULT, HAND_INC_CHIPS, HAND_INC_MULT,
)
from balatro_rl.envs.actions import MAX_JOKERS

F0 = jnp.float32(0.0)
F1 = jnp.float32(1.0)
I0 = jnp.int32(0)
JOKER_SLOTS = 5  # real cap; empty_joker_slots = JOKER_SLOTS - n_jokers

# --- in-scope JokerType ids (== engine.jokers.base.JokerType values) ----------
# Dense index 0 is the no-op; in-scope ids get dense indices 1..N in THIS order.
INSCOPE_IDS = (
    1,                              # JOKER
    2, 3, 4, 5, 119,                # suit +mult: Greedy Lusty Wrathful Gluttonous Onyx
    118,                            # suit +chips: Arrowhead
    33, 104, 78,                    # face: Scary Smiley Photograph
    31, 39, 40, 41, 101,            # rank: Fibonacci EvenSteven OddTodd Scholar WalkieTalkie
    36, 109,                        # retrigger: Hack SockAndBuskin
    6, 7, 8, 9, 10,                 # contains +mult: Jolly Zany Mad Crazy Droll
    11, 12, 13, 14, 15,             # contains +chips: Sly Wily Clever Devious Crafty
    131, 132, 133, 134, 135,        # contains xmult: Duo Trio Family Order Tribe
    16, 22, 23, 34, 17, 93, 53, 43, 62,  # context: Half Banner Mystic Abstract Stencil Bull Blue Supernova CardSharp
    128, 122,                       # scoring-suit-set xmult: SeeingDouble FlowerPot
    72, 48,                         # held: Baron(on_held) Blackboard(independent)
    52, 37,                         # rule flags: Splash Pareidolia (no effect branch)
)
N_INSCOPE = len(INSCOPE_IDS)        # 47
SPLASH_ID = 52
PAREIDOLIA_ID = 37

_MAX_ID = max(INSCOPE_IDS) + 1
import numpy as _np
_dense_np = _np.zeros(_MAX_ID, dtype=_np.int32)
for _d, _jid in enumerate(INSCOPE_IDS, start=1):
    _dense_np[_jid] = _d
DENSE_MAP = jnp.asarray(_dense_np)   # id -> dense index (0 for empty/unknown)


def _dense(jid):
    """Map a JokerType id (clamped to the table) to its dense index, 0 if absent."""
    jid = jnp.clip(jnp.asarray(jid, jnp.int32), 0, _MAX_ID - 1)
    return DENSE_MAP[jid]


# --- independent-branch context (read-only aggregates) ------------------------
class IndepCtx(NamedTuple):
    contains_pair: jnp.ndarray
    contains_two_pair: jnp.ndarray
    contains_trip: jnp.ndarray
    contains_quad: jnp.ndarray
    contains_straight: jnp.ndarray
    contains_flush: jnp.ndarray
    has_club_and_other: jnp.ndarray
    all_four_suits: jnp.ndarray
    all_dark: jnp.ndarray
    n_jokers: jnp.ndarray
    empty_slots: jnp.ndarray
    money: jnp.ndarray
    discards_left: jnp.ndarray
    deck_count: jnp.ndarray
    plays_run_ht: jnp.ndarray
    plays_round_ht: jnp.ndarray
    played_count: jnp.ndarray


# --- branch tables (populated in Tasks 2.3-2.5; all no-op for now) -------------
def _noop_score(r, s, f, ff):   return (I0, F0, F1)
def _noop_held(r, s, f):        return (I0, F0, F1)
def _noop_indep(c):             return (I0, F0, F1)
def _noop_retrig(r, s, f):      return I0

# Each table has N_INSCOPE + 1 entries (index 0 = no-op). Tasks 2.3-2.5 replace
# the no-ops at the dense indices they implement.
ON_SCORE_BRANCHES = [_noop_score] * (N_INSCOPE + 1)
ON_HELD_BRANCHES  = [_noop_held]  * (N_INSCOPE + 1)
INDEP_BRANCHES    = [_noop_indep] * (N_INSCOPE + 1)
RETRIG_BRANCHES   = [_noop_retrig] * (N_INSCOPE + 1)


def _contains_predicates(p_rank, p_suit, p_mask):
    """contains_* over the PLAYED cards (mirror engine.hands.contains)."""
    m = p_mask.astype(jnp.int32)
    n = jnp.sum(m)
    rb = p_rank.astype(jnp.int32) - 2
    oh = (rb[:, None] == jnp.arange(_N_RANK_BUCKETS)[None, :]).astype(jnp.int32) * m[:, None]
    rc = jnp.sum(oh, axis=0)                       # rank counts[13]
    n_ge2 = jnp.sum(rc >= 2)
    has_trip = jnp.any(rc >= 3)
    has_quad = jnp.any(rc >= 4)
    contains_pair = n_ge2 >= 1
    contains_two_pair = n_ge2 >= 2                 # full house -> trip rank + pair rank = 2
    # straight / flush reuse detect_hand_type's logic via a cheap recompute.
    so = (p_suit.astype(jnp.int32)[:, None] == jnp.arange(4)[None, :]).astype(jnp.int32) * m[:, None]
    suit_counts = jnp.sum(so, axis=0)
    is_flush = (n == 5) & (jnp.max(suit_counts) == 5)
    present = rc > 0
    ace = present[12]
    low_pad = jnp.concatenate([ace[None], present]).astype(jnp.int32)
    windows = jnp.stack([low_pad[i:i + 5] for i in range(low_pad.shape[0] - 4)], axis=0)
    has_run5 = jnp.any(jnp.sum(windows, axis=1) == 5)
    is_straight = (n == 5) & (jnp.sum(present) == 5) & has_run5
    return (contains_pair, contains_two_pair, has_trip, has_quad, is_straight, is_flush)


def score_with_jokers(p_rank, p_suit, p_mask, h_rank, h_suit, h_mask, levels, jokers,
                      *, money, discards_left, deck_count, hand_plays_run, hand_plays_round):
    """Ordered fold matching score_play (plain cards). Returns (hand_type, chips, mult, score)."""
    p_rank = jnp.asarray(p_rank, jnp.int32); p_suit = jnp.asarray(p_suit, jnp.int32)
    p_mask = jnp.asarray(p_mask, jnp.bool_)
    h_rank = jnp.asarray(h_rank, jnp.int32); h_suit = jnp.asarray(h_suit, jnp.int32)
    h_mask = jnp.asarray(h_mask, jnp.bool_)
    levels = jnp.asarray(levels, jnp.int32)
    jokers = jnp.asarray(jokers, jnp.int32)

    ht = detect_hand_type(p_rank, p_suit, p_mask)
    splash = jnp.any(jokers == SPLASH_ID)
    all_face = jnp.any(jokers == PAREIDOLIA_ID)

    base_sm = _scoring_mask(ht, p_rank, p_suit, p_mask)
    scoring_mask = jnp.where(splash, p_mask, base_sm)

    lvl = levels[ht]
    chips = (HAND_BASE_CHIPS[ht] + HAND_INC_CHIPS[ht] * (lvl - 1)).astype(jnp.int32)
    mult = (HAND_BASE_MULT[ht] + HAND_INC_MULT[ht] * (lvl - 1)).astype(jnp.float32)

    face_played = ((p_rank == 11) | (p_rank == 12) | (p_rank == 13) | all_face)
    face_held = ((h_rank == 11) | (h_rank == 12) | (h_rank == 13) | all_face)

    # first scoring face slot (played-slot index), else -1
    sf = scoring_mask & face_played
    first_face_idx = jnp.where(jnp.any(sf), jnp.argmax(sf.astype(jnp.int32)), jnp.int32(-1))

    # independent-branch aggregates
    (c_pair, c_two, c_trip, c_quad, c_str, c_flush) = _contains_predicates(p_rank, p_suit, p_mask)
    sc_suits = scoring_mask[:, None] & (p_suit[:, None] == jnp.arange(4)[None, :])
    suit_present = jnp.any(sc_suits, axis=0)         # bool[4] among scoring cards
    has_club_and_other = suit_present[2] & jnp.any(suit_present & (jnp.arange(4) != 2))
    all_four = jnp.all(suit_present)
    held_dark = (~h_mask) | (h_suit == 0) | (h_suit == 2)
    all_dark = jnp.all(held_dark)
    n_jokers = jnp.sum(jokers != 0)
    empty_slots = jnp.maximum(0, JOKER_SLOTS - n_jokers)
    idx = IndepCtx(c_pair, c_two, c_trip, c_quad, c_str, c_flush,
                   has_club_and_other, all_four, all_dark, n_jokers, empty_slots,
                   jnp.asarray(money, jnp.int32), jnp.asarray(discards_left, jnp.int32),
                   jnp.asarray(deck_count, jnp.int32),
                   hand_plays_run[ht], hand_plays_round[ht], jnp.sum(p_mask))

    dense_slots = _dense(jokers)                      # int32[MAX_JOKERS]

    # ---- Phase A: scored cards L->R, with retriggers ----
    for i in range(MAX_SELECT):
        r = p_rank[i]; s = p_suit[i]; f = face_played[i]
        in_scoring = scoring_mask[i]
        ff = (jnp.int32(i) == first_face_idx)
        retrig = I0
        for slot in range(MAX_JOKERS):
            retrig = retrig + lax.switch(dense_slots[slot], RETRIG_BRANCHES, r, s, f)
        for pk in range(1 + MAX_JOKERS):             # static unroll bound
            active = in_scoring & (jnp.int32(pk) < (1 + retrig))
            chips = chips + jnp.where(active, _card_chip(r), 0)
            for slot in range(MAX_JOKERS):
                dc, dm, xm = lax.switch(dense_slots[slot], ON_SCORE_BRANCHES, r, s, f, ff)
                chips = chips + jnp.where(active, dc, 0)
                mult = mult + jnp.where(active, dm, F0)
                mult = mult * jnp.where(active, xm, F1)

    # ---- Phase B: held cards ----
    for j in range(MAX_HAND):
        r = h_rank[j]; s = h_suit[j]; f = face_held[j]
        occ = h_mask[j]
        for slot in range(MAX_JOKERS):
            dc, dm, xm = lax.switch(dense_slots[slot], ON_HELD_BRANCHES, r, s, f)
            chips = chips + jnp.where(occ, dc, 0)
            mult = mult + jnp.where(occ, dm, F0)
            mult = mult * jnp.where(occ, xm, F1)

    # ---- Phase C: independent jokers, slot order ----
    for slot in range(MAX_JOKERS):
        dc, dm, xm = lax.switch(dense_slots[slot], INDEP_BRANCHES, idx)
        chips = chips + dc
        mult = mult + dm
        mult = mult * xm

    score = jnp.floor(chips.astype(jnp.float32) * mult).astype(jnp.int32)
    return ht.astype(jnp.int32), chips.astype(jnp.int32), mult.astype(jnp.float32), score
```

Note `_card_chip(r)` here is called on a scalar; the existing `scoring._card_chip` accepts arrays — calling on a scalar returns a scalar, which is fine. Confirm `detect_hand_type`, `_card_chip`, `_scoring_mask`, `_N_RANK_BUCKETS`, and the `HAND_*` tables are importable from `scoring.py` (add them to its module namespace if any are missing).

- [ ] **Step 5: Run the test; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_joker_scoring_parity.py::test_empty_loadout_reduces_to_score_core -v
```

- [ ] **Step 6: Add a jit/vmap structural test** (the on-device path):

```python
def test_score_with_jokers_jit_vmap_batches():
    import jax
    jf = jax.jit(lambda *a, **k: score_with_jokers(*a, **k))
    # build a tiny batch of 3 plain hands, empty loadout
    pr = jnp.array([[14,14,5,0,0],[2,3,4,5,6],[10,10,10,2,2]], jnp.int32)
    ps = jnp.array([[0,1,2,0,0],[0,0,0,0,0],[0,1,2,3,0]], jnp.int32)
    pm = jnp.array([[1,1,1,0,0],[1,1,1,1,1],[1,1,1,1,1]], bool)
    hr = jnp.zeros((3,8), jnp.int32); hs = jnp.zeros((3,8), jnp.int32); hm = jnp.zeros((3,8), bool)
    lv = jnp.ones((3,12), jnp.int32)
    from balatro_rl.envs.actions import MAX_JOKERS
    jk = jnp.zeros((3, MAX_JOKERS), jnp.int32)
    z = jnp.zeros(3, jnp.int32); z12 = jnp.zeros((3,12), jnp.int32)
    out = jax.vmap(lambda *a: score_with_jokers(
        a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
        money=a[8],discards_left=a[9],deck_count=a[10],
        hand_plays_run=a[11],hand_plays_round=a[12]))(
        pr,ps,pm,hr,hs,hm,lv,jk,z,z,z,z12,z12)
    assert out[3].shape == (3,)  # score per env
```

- [ ] **Step 7: Run it; expect PASS.** Commit.

```bash
git add balatro_rl/engine_jax/scoring.py balatro_rl/engine_jax/jokers.py tests/engine_jax/test_joker_scoring_parity.py
git commit -m "engine_jax: joker scoring scaffold (dense map, 4 branch tables, ordered fold); empty loadout == score_core"
```

---

## Task 2.3: Independent-hook joker branches (batch 1)

Wire every independent-only joker: flat, hand-contains (+mult/+chips/×mult), context-linear, scoring-suit-set, Blackboard. These read only `IndepCtx`.

**Files:** Modify `balatro_rl/engine_jax/jokers.py`, `tests/engine_jax/test_joker_scoring_parity.py`.

- [ ] **Step 1: Write the failing parity helper + targeted tests.** Add to the test file a host harness that calls the oracle and the kernel on the same inputs:

```python
from balatro_rl.engine.cards import Card
from balatro_rl.engine.jokers.base import JokerState
from balatro_rl.engine.scoring import score_play

def _oracle(played, jokers, held=(), levels=(), money=0, discards_left=0,
            deck_count=0, hand_plays_run=(), hand_plays_round=()):
    js = tuple(JokerState(type=j) for j in jokers)
    cards = [Card(rank=r, suit=s) for r, s in played]
    heldc = [Card(rank=r, suit=s) for r, s in held]
    return score_play(cards, js, tuple(heldc), joker_slots=5, money=money,
                      hands_left=0, discards_left=discards_left, deck_count=deck_count,
                      hand_plays_run=tuple(hand_plays_run), hand_plays_round=tuple(hand_plays_round),
                      levels=tuple(levels))

def _kernel(played, jokers, held=(), levels=None, money=0, discards_left=0,
            deck_count=0, hand_plays_run=None, hand_plays_round=None):
    from balatro_rl.envs.actions import MAX_JOKERS
    pr, ps, pm = _pad5([r for r, _ in played], [s for _, s in played])
    hr = np.zeros(8, np.int8); hs = np.zeros(8, np.int8); hm = np.zeros(8, bool)
    for i, (r, s) in enumerate(held):
        hr[i] = r; hs[i] = s; hm[i] = True
    jk = np.zeros(MAX_JOKERS, np.int32)
    for i, j in enumerate(jokers):
        jk[i] = j
    lv = jnp.ones(12, jnp.int32) if levels is None else jnp.asarray(levels, jnp.int32)
    hpr = jnp.zeros(12, jnp.int32) if hand_plays_run is None else jnp.asarray(hand_plays_run, jnp.int32)
    hpo = jnp.zeros(12, jnp.int32) if hand_plays_round is None else jnp.asarray(hand_plays_round, jnp.int32)
    return score_with_jokers(pr, ps, pm, jnp.asarray(hr), jnp.asarray(hs), jnp.asarray(hm),
                             lv, jnp.asarray(jk),
                             money=jnp.int32(money), discards_left=jnp.int32(discards_left),
                             deck_count=jnp.int32(deck_count),
                             hand_plays_run=hpr, hand_plays_round=hpo)

def _assert_match(played, jokers, **kw):
    o = _oracle(played, jokers, **kw)
    k = _kernel(played, jokers, **kw)
    assert (int(o.hand_type), int(o.chips), int(round(o.mult * 1000))) == \
           (int(k[0]), int(k[1]), int(round(float(k[2]) * 1000))), (played, jokers, kw, o, k)
    assert o.score == int(k[3]), (played, jokers, kw, o.score, int(k[3]))

def test_independent_batch1():
    # JOKER +4 mult on a pair of Aces
    _assert_match([(14,0),(14,1)], [1])
    # Jolly +8 mult (pair) ; Sly +50 chips (pair)
    _assert_match([(14,0),(14,1),(7,2)], [6]); _assert_match([(14,0),(14,1),(7,2)], [11])
    # The Duo x2 (pair) ; The Trio x3 (trips)
    _assert_match([(9,0),(9,1)], [131]); _assert_match([(9,0),(9,1),(9,2)], [132])
    # Abstract +3 per joker (here 2 jokers -> +6) ; Banner +30 per discard
    _assert_match([(2,0)], [34, 1]); _assert_match([(2,0)], [22], discards_left=3)
    # Bull +2 per $ ; Blue +2 per deck card ; Half +20 if <=3 cards
    _assert_match([(2,0)], [93], money=7); _assert_match([(2,0)], [53], deck_count=44)
    _assert_match([(2,0),(3,0),(4,0)], [16])
    # Mystic +15 if 0 discards ; Supernova +(plays_run+1) ; Card Sharp x3 if played this round
    _assert_match([(2,0)], [23], discards_left=0)
    _assert_match([(2,0),(2,1)], [43], hand_plays_run=[0,2,0,0,0,0,0,0,0,0,0,0])
    _assert_match([(2,0),(2,1)], [62], hand_plays_round=[0,1,0,0,0,0,0,0,0,0,0,0])
    # Joker Stencil x(empty_slots+1): 1 joker, 4 empty -> x5
    _assert_match([(2,0)], [17])
    # Flush jokers: Droll +10, The Tribe x2, Crafty +80
    flush = [(2,1),(5,1),(8,1),(11,1),(13,1)]
    _assert_match(flush, [10]); _assert_match(flush, [135]); _assert_match(flush, [15])
    # Straight jokers: Crazy +12, The Order x3, Devious +100
    straight = [(2,0),(3,1),(4,2),(5,3),(6,0)]
    _assert_match(straight, [9]); _assert_match(straight, [134]); _assert_match(straight, [14])
    # Two-pair: Mad +10, Clever +80 ; Trips: Zany +12, Wily +100 ; Quads: The Family x4
    _assert_match([(2,0),(2,1),(3,2),(3,3)], [8]); _assert_match([(2,0),(2,1),(3,2),(3,3)], [13])
    _assert_match([(4,0),(4,1),(4,2)], [7]); _assert_match([(4,0),(4,1),(4,2)], [12])
    _assert_match([(4,0),(4,1),(4,2),(4,3)], [133])
    # Seeing Double x2 (club + other) ; Flower Pot x3 (all four suits)
    _assert_match([(2,2),(3,1)], [128]); _assert_match([(2,0),(3,1),(4,2),(5,3),(6,0)], [122])
    # Blackboard x3 (all held spade/club) ; vacuous when none held
    _assert_match([(2,0)], [48]); _assert_match([(2,0)], [48], held=[(9,0),(9,2)])
```

- [ ] **Step 2: Run; expect FAIL** (no-op branches give wrong scores).

- [ ] **Step 3: Implement the independent branches** in `jokers.py`. Add factory helpers + register each at its dense index:

```python
def _indep_flag_add(getter, dchips=0, dmult=0.0):
    def fn(c): return (jnp.where(getter(c), jnp.int32(dchips), I0),
                       jnp.where(getter(c), jnp.float32(dmult), F0), F1)
    return fn

def _indep_flag_xmult(getter, x):
    def fn(c): return (I0, F0, jnp.where(getter(c), jnp.float32(x), F1))
    return fn

def _indep_linear_chips(getter):    # +coef * value chips
    def make(coef):
        def fn(c): return (jnp.int32(coef) * getter(c).astype(jnp.int32), F0, F1)
        return fn
    return make

def _set(table, jid, fn):
    table[_dense_np[jid]] = fn

# flat
_set(INDEP_BRANCHES, 1, lambda c: (I0, jnp.float32(4.0), F1))                  # Joker +4 mult
# hand-contains +mult
_set(INDEP_BRANCHES, 6, _indep_flag_add(lambda c: c.contains_pair, dmult=8))   # Jolly
_set(INDEP_BRANCHES, 7, _indep_flag_add(lambda c: c.contains_trip, dmult=12))  # Zany
_set(INDEP_BRANCHES, 8, _indep_flag_add(lambda c: c.contains_two_pair, dmult=10))  # Mad
_set(INDEP_BRANCHES, 9, _indep_flag_add(lambda c: c.contains_straight, dmult=12)) # Crazy
_set(INDEP_BRANCHES, 10, _indep_flag_add(lambda c: c.contains_flush, dmult=10))  # Droll
# hand-contains +chips
_set(INDEP_BRANCHES, 11, _indep_flag_add(lambda c: c.contains_pair, dchips=50))   # Sly
_set(INDEP_BRANCHES, 12, _indep_flag_add(lambda c: c.contains_trip, dchips=100))  # Wily
_set(INDEP_BRANCHES, 13, _indep_flag_add(lambda c: c.contains_two_pair, dchips=80))  # Clever
_set(INDEP_BRANCHES, 14, _indep_flag_add(lambda c: c.contains_straight, dchips=100)) # Devious
_set(INDEP_BRANCHES, 15, _indep_flag_add(lambda c: c.contains_flush, dchips=80))  # Crafty
# hand-contains xmult
_set(INDEP_BRANCHES, 131, _indep_flag_xmult(lambda c: c.contains_pair, 2.0))     # Duo
_set(INDEP_BRANCHES, 132, _indep_flag_xmult(lambda c: c.contains_trip, 3.0))     # Trio
_set(INDEP_BRANCHES, 133, _indep_flag_xmult(lambda c: c.contains_quad, 4.0))     # Family
_set(INDEP_BRANCHES, 134, _indep_flag_xmult(lambda c: c.contains_straight, 3.0)) # Order
_set(INDEP_BRANCHES, 135, _indep_flag_xmult(lambda c: c.contains_flush, 2.0))    # Tribe
# context-linear
_set(INDEP_BRANCHES, 16, _indep_flag_add(lambda c: c.played_count <= 3, dmult=20))  # Half
_set(INDEP_BRANCHES, 22, lambda c: (jnp.int32(30) * c.discards_left, F0, F1))        # Banner
_set(INDEP_BRANCHES, 23, _indep_flag_add(lambda c: c.discards_left == 0, dmult=15))  # Mystic Summit
_set(INDEP_BRANCHES, 34, lambda c: (I0, jnp.float32(3.0) * c.n_jokers.astype(jnp.float32), F1))  # Abstract
_set(INDEP_BRANCHES, 17, lambda c: (I0, F0, (c.empty_slots + 1).astype(jnp.float32)))            # Stencil
_set(INDEP_BRANCHES, 93, lambda c: (jnp.int32(2) * jnp.maximum(0, c.money), F0, F1))             # Bull
_set(INDEP_BRANCHES, 53, lambda c: (jnp.int32(2) * c.deck_count, F0, F1))                        # Blue
_set(INDEP_BRANCHES, 43, lambda c: (I0, (c.plays_run_ht + 1).astype(jnp.float32), F1))           # Supernova
_set(INDEP_BRANCHES, 62, _indep_flag_xmult(lambda c: c.plays_round_ht >= 1, 3.0))                # Card Sharp
# scoring-suit-set xmult
_set(INDEP_BRANCHES, 128, _indep_flag_xmult(lambda c: c.has_club_and_other, 2.0))   # Seeing Double
_set(INDEP_BRANCHES, 122, _indep_flag_xmult(lambda c: c.all_four_suits, 3.0))       # Flower Pot
# Blackboard (independent; held aggregate)
_set(INDEP_BRANCHES, 48, _indep_flag_xmult(lambda c: c.all_dark, 3.0))              # Blackboard
```

> `_set` mutates the list at the joker's dense index; place these statements at module load AFTER the `*_BRANCHES` lists are created and AFTER `_dense_np` exists.

- [ ] **Step 4: Run; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_joker_scoring_parity.py::test_independent_batch1 -v
```

- [ ] **Step 5: Commit.**

```bash
git add balatro_rl/engine_jax/jokers.py tests/engine_jax/test_joker_scoring_parity.py
git commit -m "engine_jax: independent joker branches (flat/contains/context/suit-set/blackboard) at parity"
```

---

## Task 2.4: `on_score` + retrigger joker branches (batch 2)

Wire the per-scored-card jokers (suit, face, rank), Photograph (first-face ×2, retrigger-aware), and the two retrigger jokers.

**Files:** Modify `balatro_rl/engine_jax/jokers.py`, `tests/engine_jax/test_joker_scoring_parity.py`.

- [ ] **Step 1: Write the failing targeted tests:**

```python
def test_on_score_batch2():
    # suit +mult: Greedy(♦+3), Lusty(♥+3), Wrathful(♠+3), Gluttonous(♣+3), Onyx(♣+7)
    _assert_match([(5,3),(7,3)], [2]); _assert_match([(5,1),(7,1)], [3])
    _assert_match([(5,0),(7,0)], [4]); _assert_match([(5,2),(7,2)], [5]); _assert_match([(5,2),(7,2)], [119])
    # suit +chips: Arrowhead(♠+50)
    _assert_match([(5,0),(7,0)], [118])
    # face: Scary +30 chips, Smiley +5 mult (on K/Q)
    _assert_match([(13,0),(12,1)], [33]); _assert_match([(13,0),(12,1)], [104])
    # rank: Fibonacci, Even Steven, Odd Todd, Scholar, Walkie Talkie
    _assert_match([(2,0),(3,1)], [31]); _assert_match([(2,0),(4,1)], [39])
    _assert_match([(3,0),(5,1)], [40]); _assert_match([(14,0),(14,1)], [41]); _assert_match([(10,0),(4,1)], [101])
    # Photograph x2 on first scoring face only (two faces -> still single x2)
    _assert_match([(13,0),(12,1)], [78])
    # Hack retriggers 2-5 (each adds its rank chips again); pair to make them score
    _assert_match([(3,0),(3,1)], [36])
    # Sock & Buskin retriggers faces; pair of Kings
    _assert_match([(13,0),(13,1)], [109])
    # Photograph + Sock&Buskin: first-face card retriggered -> x2 applies twice
    _assert_match([(13,0),(13,1)], [78, 109])
    # ordering: [The Duo x2, Joker +4] vs [Joker +4, The Duo x2] differ; both match oracle
    _assert_match([(14,0),(14,1)], [131, 1]); _assert_match([(14,0),(14,1)], [1, 131])
```

- [ ] **Step 2: Run; expect FAIL.**

- [ ] **Step 3: Implement the `on_score` and `retrigger` branches:**

```python
def _score_suit(suit_id, dchips=0, dmult=0.0):
    def fn(r, s, f, ff):
        hit = (s == suit_id)
        return (jnp.where(hit, jnp.int32(dchips), I0),
                jnp.where(hit, jnp.float32(dmult), F0), F1)
    return fn

def _score_face(dchips=0, dmult=0.0):
    def fn(r, s, f, ff):
        return (jnp.where(f, jnp.int32(dchips), I0),
                jnp.where(f, jnp.float32(dmult), F0), F1)
    return fn

def _score_rank_in(ranks, dchips=0, dmult=0.0):
    rset = jnp.asarray(ranks, jnp.int32)
    def fn(r, s, f, ff):
        hit = jnp.any(r == rset)
        return (jnp.where(hit, jnp.int32(dchips), I0),
                jnp.where(hit, jnp.float32(dmult), F0), F1)
    return fn

# suit on_score
_set(ON_SCORE_BRANCHES, 2, _score_suit(3, dmult=3))    # Greedy ♦
_set(ON_SCORE_BRANCHES, 3, _score_suit(1, dmult=3))    # Lusty ♥
_set(ON_SCORE_BRANCHES, 4, _score_suit(0, dmult=3))    # Wrathful ♠
_set(ON_SCORE_BRANCHES, 5, _score_suit(2, dmult=3))    # Gluttonous ♣
_set(ON_SCORE_BRANCHES, 119, _score_suit(2, dmult=7))  # Onyx Agate ♣
_set(ON_SCORE_BRANCHES, 118, _score_suit(0, dchips=50))  # Arrowhead ♠
# face on_score
_set(ON_SCORE_BRANCHES, 33, _score_face(dchips=30))    # Scary Face
_set(ON_SCORE_BRANCHES, 104, _score_face(dmult=5))     # Smiley Face
_set(ON_SCORE_BRANCHES, 78, lambda r, s, f, ff: (I0, F0, jnp.where(ff, jnp.float32(2.0), F1)))  # Photograph
# rank on_score
_set(ON_SCORE_BRANCHES, 31, _score_rank_in([14, 2, 3, 5, 8], dmult=8))    # Fibonacci
_set(ON_SCORE_BRANCHES, 39, _score_rank_in([2, 4, 6, 8, 10], dmult=4))    # Even Steven
_set(ON_SCORE_BRANCHES, 40, _score_rank_in([3, 5, 7, 9, 14], dchips=31))  # Odd Todd
_set(ON_SCORE_BRANCHES, 41, _score_rank_in([14], dchips=20, dmult=4))     # Scholar
_set(ON_SCORE_BRANCHES, 101, _score_rank_in([10, 4], dchips=10, dmult=4)) # Walkie Talkie
# retrigger
_set(RETRIG_BRANCHES, 36, lambda r, s, f: jnp.where(jnp.any(r == jnp.array([2,3,4,5], jnp.int32)), jnp.int32(1), I0))  # Hack
_set(RETRIG_BRANCHES, 109, lambda r, s, f: jnp.where(f, jnp.int32(1), I0))  # Sock & Buskin
```

- [ ] **Step 4: Run; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_joker_scoring_parity.py::test_on_score_batch2 -v
```

- [ ] **Step 5: Commit.**

```bash
git add balatro_rl/engine_jax/jokers.py tests/engine_jax/test_joker_scoring_parity.py
git commit -m "engine_jax: on_score + retrigger joker branches (suit/face/rank/photograph/hack/sock) at parity"
```

---

## Task 2.5: `on_held` joker (Baron) + full randomized Gate A

Wire Baron, then build the randomized parity gate with coverage, golden values, fold-order, negative control, retrigger-bound, and out-of-scope checks.

**Files:** Modify `balatro_rl/engine_jax/jokers.py`, `tests/engine_jax/test_joker_scoring_parity.py`.

- [ ] **Step 1: Write the failing Baron + randomized tests.** Append:

```python
def test_baron_held():
    # Baron x1.5 per held King; play a pair, hold two Kings -> x1.5*1.5
    _assert_match([(9,0),(9,1)], [72], held=[(13,0),(13,2)])
    _assert_match([(9,0),(9,1)], [72], held=[(13,0),(9,2)])   # one King

# --- full randomized parity over the whole in-scope set ---
from balatro_rl.engine_jax.jokers import INSCOPE_IDS, N_INSCOPE
_FIRED = set()  # dense ids observed firing across the corpus (coverage)

def _random_case(rng):
    n = int(rng.integers(1, 6))
    deck = rng.permutation([(r, s) for r in range(2, 15) for s in range(4)])
    played = [tuple(int(x) for x in deck[i]) for i in range(n)]
    nheld = int(rng.integers(0, 6))
    held = [tuple(int(x) for x in deck[n + i]) for i in range(nheld)]
    k = int(rng.integers(0, 6))
    jokers = [int(rng.choice(INSCOPE_IDS)) for _ in range(k)]
    levels = [int(rng.integers(1, 4)) for _ in range(12)]
    money = int(rng.integers(0, 30)); discards = int(rng.integers(0, 4)); deck_count = int(rng.integers(0, 45))
    hpr = [int(rng.integers(0, 4)) for _ in range(12)]; hpo = [int(rng.integers(0, 3)) for _ in range(12)]
    return dict(played=played, jokers=jokers, held=held, levels=levels, money=money,
                discards_left=discards, deck_count=deck_count, hand_plays_run=hpr, hand_plays_round=hpo)

@pytest.mark.parametrize("seed", range(200))
def test_random_parity_ci(seed):
    rng = np.random.default_rng(seed)
    case = _random_case(rng)
    _assert_match(**case)
    from balatro_rl.engine_jax.jokers import _dense_np
    for j in case["jokers"]:
        _FIRED.add(int(_dense_np[j]))

def test_coverage_every_inscope_joker_appears():
    # Drive enough cases to hit all ids; assert all dense indices fired.
    rng = np.random.default_rng(12345)
    seen = set()
    for _ in range(4000):
        case = _random_case(rng)
        for j in case["jokers"]:
            seen.add(j)
        if len(seen) == N_INSCOPE:
            break
    assert set(INSCOPE_IDS) <= seen, set(INSCOPE_IDS) - seen

def test_golden_values_oracle_free():
    # Hand-computed: pair of Aces (PAIR base 10c/2m), Aces score 11+11=22 -> 32c.
    # Joker +4 mult -> mult 6 ; The Duo x2 (pair) -> applied in slot order.
    # Slot order [Joker(+4), Duo(x2)]: (2+4)*2 = 12 mult -> 32*12 = 384.
    o = _kernel([(14,0),(14,1)], [1, 131]); assert int(o[3]) == 384
    # Slot order [Duo(x2), Joker(+4)]: (2*2)+4 = 8 mult -> 32*8 = 256.
    o = _kernel([(14,0),(14,1)], [131, 1]); assert int(o[3]) == 256

@pytest.mark.parametrize("order", [(1,131,6), (6,1,131), (131,6,1)])
def test_fold_order_matches_oracle(order):
    _assert_match([(14,0),(14,1)], list(order))

def test_negative_control_gate_has_teeth():
    # Order sensitivity: [Joker(+4), Duo(x2)] != [Duo(x2), Joker(+4)] (384 vs 256). If the
    # kernel were order-insensitive (a "sum-then-multiply" bug) these would be equal and the
    # episode/golden gates would silently pass a wrong kernel. They must differ.
    a = int(_kernel([(14,0),(14,1)], [1, 131])[3])
    b = int(_kernel([(14,0),(14,1)], [131, 1])[3])
    assert a != b and (a, b) == (384, 256)

def test_out_of_scope_id_is_noop():
    # A deferred joker id (e.g. RIDE_THE_BUS=44) must behave as an empty slot.
    base = _kernel([(14,0),(14,1)], [])
    with_oos = _kernel([(14,0),(14,1)], [44])
    assert int(base[3]) == int(with_oos[3])

def test_max_retrigger_path_parity():
    # Exercises the static unroll bound (1 + MAX_JOKERS passes) and verifies via parity:
    # 5 Hacks all retrigger a played 3 -> 5 retriggers -> 6 passes.
    _assert_match([(3,0),(3,1)], [36, 36, 36, 36, 36])
    # Pareidolia makes a low card count as a face, so Hack AND Sock & Buskin both fire on a 3
    # -> +2 retriggers on that card; parity must still hold.
    _assert_match([(3,0),(3,1)], [36, 109, 37])

def test_planet_levels_parity():
    # Spec §8.C: a leveled loadout (Planet upgrades) + jokers scores at parity. Level the
    # PAIR hand type (index 1) to 3 and add Joker; the kernel must honor levels[ht] exactly.
    lv = [1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    _assert_match([(14,0),(14,1)], [1], levels=lv)
    _assert_match([(9,0),(9,1),(9,2)], [132], levels=[1,1,1,4,1,1,1,1,1,1,1,1])  # trips lvl 4 + The Trio
```

> The `test_random_parity_ci` is the CI subset (200 cases). Add a slow ≥1000-case variant guarded by the Phase-1 pattern:

```python
import os
BALATRO_RUN_SLOW = os.environ.get("BALATRO_RUN_SLOW") == "1"

@pytest.mark.slow
@pytest.mark.skipif(not BALATRO_RUN_SLOW, reason="set BALATRO_RUN_SLOW=1")
def test_random_parity_1000():
    rng = np.random.default_rng(2024)
    for _ in range(1000):
        _assert_match(**_random_case(rng))
```

- [ ] **Step 2: Run; expect FAIL** (Baron not wired; possibly some random cases fail if any earlier branch is subtly off — fix against the oracle dump).

- [ ] **Step 3: Implement Baron** in `jokers.py`:

```python
_set(ON_HELD_BRANCHES, 72, lambda r, s, f: (I0, F0, jnp.where(r == 13, jnp.float32(1.5), F1)))  # Baron
```

- [ ] **Step 4: Run the full Gate A; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_joker_scoring_parity.py -v
JAX_PLATFORMS=cpu BALATRO_RUN_SLOW=1 python -m pytest tests/engine_jax/test_joker_scoring_parity.py::test_random_parity_1000 -v
```

Expected: all pass, including coverage (every in-scope joker fired) and golden values.

- [ ] **Step 5: Commit.**

```bash
git add balatro_rl/engine_jax/jokers.py tests/engine_jax/test_joker_scoring_parity.py
git commit -m "engine_jax: Baron (on_held) + full randomized Gate A (coverage/golden/fold-order/neg-control/oos)"
```

---

## Task 2.6: `step` integration — held + context wiring

Route `step` through `score_with_jokers`, deriving held cards and the scalar context from the state. Empty loadout must leave Phase-1 episode behavior identical.

**Files:** Modify `balatro_rl/engine_jax/step.py`. Test: `tests/engine_jax/test_step_parity.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/engine_jax/test_step_parity.py` a check that a fixed loadout changes the score the way the oracle says, and the empty loadout does not:

```python
def test_step_uses_jokers_loadout():
    import jax.numpy as jnp, numpy as np
    from balatro_rl.engine_jax.step import reset, step
    from balatro_rl.engine_jax.config import Verb
    from balatro_rl.envs.actions import MAX_JOKERS
    # Deterministic deck: first 8 cards are the opening hand. Make slots 0,1 a pair of Aces.
    ranks = [14,14] + [r for r in range(2,15) for _ in range(4)][:50]
    suits = [0,1]  + [s for _ in range(2,15) for s in range(4)][:50]
    rt = None
    jk = np.zeros(MAX_JOKERS, np.int32); jk[0] = 1  # JOKER +4 mult
    st = reset(ranks, suits, required=10**9, required_table=rt, jokers=jk)  # huge required -> never clears
    # PLAY the pair of Aces: subset {0,1}. Find its action id via _SUBSETS.
    from balatro_rl.envs.actions import _SUBSETS
    aid = _SUBSETS.index((0, 1))
    from balatro_rl.engine_jax.step import decode_action
    verb, sel = decode_action(jnp.int32(aid))
    ns, sig = step(st, verb, sel)
    # PAIR base 10c/2m; aces 11+11 -> 32c; Joker +4 -> mult 6 -> 192.
    assert int(sig.score) == 192
```

- [ ] **Step 2: Run; expect FAIL** (step still uses `score_core`; score would be 64, not 192).

- [ ] **Step 3: Edit `step` in `step.py`.** Replace the scoring call:

```python
from balatro_rl.engine_jax.jokers import score_with_jokers
from balatro_rl.engine_jax.state import DECK_SIZE
```

In `step(...)`, after `_gather_selected`, derive held + context and call the joker kernel:

```python
    sel_rank, sel_suit, sel_present = _gather_selected(state, sel_mask)
    held_mask = state.hand_mask & ~sel_mask                       # cards kept in hand
    deck_count = DECK_SIZE - state.deck_ptr
    n_jokers = jnp.sum(state.jokers != 0)
    hand_type, _chips, _mult, score = score_with_jokers(
        sel_rank, sel_suit, sel_present,
        state.hand_rank.astype(jnp.int32), state.hand_suit.astype(jnp.int32), held_mask,
        state.levels, state.jokers,
        money=state.money, discards_left=state.discards_left, deck_count=deck_count,
        hand_plays_run=state.hand_plays_run, hand_plays_round=state.hand_plays_round)
```

> `held_mask` is the hand minus the played selection (matches the oracle's `held` = non-played hand cards). `hand_plays_run/round` are read PRE-increment here (the bump to `play_plays_run/round` happens afterward), matching `score_play`'s pre-increment contract for Supernova/Card Sharp.

- [ ] **Step 4: Run; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_step_parity.py -v
```

- [ ] **Step 5: Run the FULL Phase-1 episode gate to prove empty-loadout invariance** (reset/reset_jax default to the zero loadout, so existing rollouts route through `score_with_jokers(empty) == score_core`):

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax -q
```

Expected: all pass. If `test_core_parity_gate` regresses, the held/context wiring diverged — debug against the oracle.

- [ ] **Step 6: Commit.**

```bash
git add balatro_rl/engine_jax/step.py tests/engine_jax/test_step_parity.py
git commit -m "engine_jax: step scores via score_with_jokers (held + context wiring); empty loadout unchanged"
```

---

## Task 2.7: Observation — joker keys

Fill `joker_types`/`joker_counter`/`joker_mask` and `global[10]` from `state.jokers`.

**Files:** Modify `balatro_rl/engine_jax/obs.py`. Test: `tests/engine_jax/test_obs_parity.py`.

- [ ] **Step 1: Write the failing test** — append to `tests/engine_jax/test_obs_parity.py`:

```python
def test_obs_joker_keys_match_python():
    import numpy as np, jax.numpy as jnp
    from balatro_rl.engine_jax.step import reset
    from balatro_rl.engine_jax.obs import encode_core
    from balatro_rl.envs.actions import MAX_JOKERS
    from balatro_rl.envs.obs import encode as py_encode
    # Build a JAX state with a loadout, and an oracle GameState with the same jokers.
    jk = np.zeros(MAX_JOKERS, np.int32); jk[0] = 1; jk[1] = 131  # Joker, The Duo
    ranks = [r for r in range(2,15) for _ in range(4)]; suits = [s for _ in range(2,15) for s in range(4)]
    st = reset(ranks, suits, required=300, jokers=jk)
    obs = encode_core(st)
    assert int(np.asarray(obs["joker_types"])[0]) == 1
    assert int(np.asarray(obs["joker_types"])[1]) == 131
    assert float(np.asarray(obs["joker_mask"])[0]) == 1.0 and float(np.asarray(obs["joker_mask"])[1]) == 1.0
    assert float(np.asarray(obs["joker_mask"])[2]) == 0.0
    assert float(np.asarray(obs["joker_counter"])[0]) == 0.0   # stateless -> symlog(0)=0
    assert float(np.asarray(obs["global"])[10]) == 2.0          # joker count
```

- [ ] **Step 2: Run; expect FAIL** (keys still zeroed).

- [ ] **Step 3: Edit `encode_core` in `obs.py`.** Replace the zeroed joker keys + set `g[10]`:

```python
    # -- Jokers (Phase 2): types from the loadout, counter=0 (stateless), mask from occupancy.
    jt = state.jokers.astype(jnp.int32)                       # [MAX_JOKERS]
    jmask = (jt != 0).astype(jnp.float32)
    g = g.at[10].set(jnp.sum(jt != 0).astype(jnp.float32))    # g[10] = #jokers
```

and in the returned dict replace:

```python
        "joker_types":      jt,
        "joker_counter":    jnp.zeros(MAX_JOKERS, dtype=jnp.float32),
        "joker_mask":       jmask,
```

(Keep `MAX_JOKERS` import already present in the function.)

- [ ] **Step 4: Run; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_obs_parity.py -v
```

- [ ] **Step 5: Commit.**

```bash
git add balatro_rl/engine_jax/obs.py tests/engine_jax/test_obs_parity.py
git commit -m "engine_jax: encode_core fills joker obs keys + global[10] joker count"
```

---

## Task 2.8: Gate B — fixed-loadout episode parity

Inject the same loadout into both engines at reset; assert within-blind parity over rollouts.

**Files:** Modify `tests/engine_jax/test_core_parity_gate.py` and `tests/engine_jax/parity_util.py`. Read both first to reuse `python_core_fields`, `jax_core_fields`, `deck_from_python`, `assert_states_equal`, `assert_hand_slots_equal`, the boundary resync helpers, and `_OBS_CORE_KEYS`.

- [ ] **Step 1: Add an obs key set + a loadout-aware harness.** In `test_core_parity_gate.py` extend the obs keys checked to include the joker keys:

```python
_OBS_JOKER_KEYS = ("joker_types", "joker_counter", "joker_mask")
```

and add a helper to construct the Python `GameState` with a given joker loadout (read how the existing gate builds the oracle state; inject `jokers=tuple(JokerState(type=j) for j in loadout)` via the engine's reset/replace path) and the JAX state via `reset(deck_from_python(gs)..., jokers=loadout)`.

- [ ] **Step 2: Write the failing parameterized gate:**

```python
import pytest
_LOADOUTS = [
    [],                       # empty (Phase-1 regression through the gate)
    [1],                      # Joker
    [3],                      # Lusty (suit on_score)
    [6],                      # Jolly (contains +mult)
    [131],                    # The Duo (xmult)
    [78, 109],                # Photograph + Sock&Buskin (retrigger interaction)
    [52, 3, 131, 109],        # Splash + suit + xmult + retrigger (high interaction)
    [72],                     # Baron (held)
    [22, 23, 53],             # Banner + Mystic + Blue (context-linear)
]

@pytest.mark.parametrize("loadout", _LOADOUTS)
def test_episode_parity_with_loadout(loadout):
    _run_loadout_episodes(loadout, n_rollouts=50, base_seed=0)  # reuses Phase-1 within-blind harness
```

`_run_loadout_episodes` mirrors the existing `test_core_parity_gate_200` loop but (a) seeds both engines' jokers with `loadout`, (b) asserts the joker obs keys in addition to `_OBS_CORE_KEYS`, (c) keeps the Phase-1 boundary-resync (the loadout is fixed and carries through `_advance_blind` unchanged on both sides).

- [ ] **Step 3: Run; expect FAIL until the harness is implemented; then PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax/test_core_parity_gate.py -v -k loadout
```

- [ ] **Step 4: Add the slow full variant** (full loadout set × ≥200 rollouts) guarded by `BALATRO_RUN_SLOW`, mirroring `test_core_parity_gate_1000`.

- [ ] **Step 5: Run the whole gate file + full engine_jax suite.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax -q
JAX_PLATFORMS=cpu BALATRO_RUN_SLOW=1 python -m pytest tests/engine_jax/test_core_parity_gate.py -q
```

- [ ] **Step 6: Commit.**

```bash
git add tests/engine_jax/test_core_parity_gate.py tests/engine_jax/parity_util.py
git commit -m "engine_jax: Gate B — fixed-loadout episode parity (state + slots + core/joker obs + reward)"
```

---

## Task 2.9: `JaxVectorEnv` + `TrainConfig` loadout knob + PPO smoke

**Files:** Modify `balatro_rl/envs/jax_vec_env.py`, `balatro_rl/agent/train.py`. Test: `tests/agent/test_jax_engine_smoke.py`. Read `jax_vec_env.py` first for the `reset()`/`batched_reset` call shape.

- [ ] **Step 1: Write the failing smoke test** — append to `tests/agent/test_jax_engine_smoke.py`:

```python
def test_ppo_smoke_with_jokers():
    from balatro_rl.agent.train import TrainConfig, train
    cfg = TrainConfig(engine="jax", num_envs=16, total_steps=2, num_steps=8,
                      joker_loadout=[1, 131])   # Joker + The Duo for every env
    out = train(cfg)  # must not raise; losses finite
    assert out is not None
```

(Match the real `TrainConfig`/`train` signature in `agent/train.py`; adapt field names — the key assertion is that a non-empty loadout trains end-to-end with finite losses.)

- [ ] **Step 2: Run; expect FAIL** (`TrainConfig` has no `joker_loadout`).

- [ ] **Step 3: Add the knob.** In `jax_vec_env.py`, `JaxVectorEnv.__init__` gains `joker_loadout=None`; build a per-env `[num_envs, MAX_JOKERS]` int32 array (broadcast the fixed loadout to all envs, 0-padded), store it, and pass it to `batched_reset(keys, req_table, self._jokers)`. In `train.py`, add `joker_loadout: list | None = None` to `TrainConfig` and forward it in the `engine=="jax"` factory branch:

```python
        venv = JaxVectorEnv(cfg.num_envs, reward_name=cfg.reward_name,
                            base_seed=cfg.seed + 1000, req_scale=cur_scale,
                            enable_bosses=cfg.enable_bosses, joker_loadout=cfg.joker_loadout)
```

- [ ] **Step 4: Run; expect PASS.**

```
JAX_PLATFORMS=cpu python -m pytest tests/agent/test_jax_engine_smoke.py -v
```

- [ ] **Step 5: Run the agent + envs suites for no regression.**

```
JAX_PLATFORMS=cpu python -m pytest tests/agent tests/envs -q
```

- [ ] **Step 6: Commit.**

```bash
git add balatro_rl/envs/jax_vec_env.py balatro_rl/agent/train.py tests/agent/test_jax_engine_smoke.py
git commit -m "engine_jax: JaxVectorEnv + TrainConfig joker_loadout knob; PPO smoke trains with jokers"
```

---

## Task 2.10: Benchmark, docs, memory

**Files:** Modify `scripts/bench_jax_engine.py`, `docs/RUNPOD_M2.md`, memory file.

- [ ] **Step 1: Add an optional loadout to the benchmark.** In `bench_jax_engine.py`, read `BENCH_JOKERS` (comma-separated ids, default empty) and pass it to `batched_reset` (broadcast to `[n, MAX_JOKERS]`). This lets the bench measure the with-jokers fold cost.

- [ ] **Step 2: Run the benchmark both ways** (CPU) and record env-steps/s:

```
JAX_PLATFORMS=cpu python scripts/bench_jax_engine.py
JAX_PLATFORMS=cpu BENCH_JOKERS=1,131,3,109,52 python scripts/bench_jax_engine.py
```

- [ ] **Step 3: Run the entire suite once more (CPU), incl. slow gates, to certify Phase 2.**

```
JAX_PLATFORMS=cpu python -m pytest tests/engine_jax tests/envs tests/agent -q
JAX_PLATFORMS=cpu BALATRO_RUN_SLOW=1 python -m pytest tests/engine_jax/test_joker_scoring_parity.py tests/engine_jax/test_core_parity_gate.py -q
```

- [ ] **Step 4: Update `docs/RUNPOD_M2.md` §8** with Phase-2 reality: ~45 pure-scoring jokers proven bit-for-bit (component + episode), the recorded throughput band with/without jokers, scope note (enhancements/economy/scaling/RNG jokers + shop = Phase 3).

- [ ] **Step 5: Update memory** `efficiency-wall-python-engine.md` with a "Phase 2 BUILT" line: joker scoring kernel + env integration, both gates green, joker count, next = Phase 3 (shop/economy/enhancements). Update `MEMORY.md` hook if needed.

- [ ] **Step 6: Commit.**

```bash
git add scripts/bench_jax_engine.py docs/RUNPOD_M2.md
git commit -m "engine_jax: Phase-2 benchmark loadout knob + RUNPOD_M2 §8 jokers update"
```

---

## Self-Review notes (for the planner / first executor)

- **Spec coverage:** every §7 family maps to a task (independent → 2.3; on_score/retrigger → 2.4; held/Baron → 2.5; rules Splash/Pareidolia → handled in 2.2's rule aggregation + exercised in 2.4/2.8). §8 test matrix items map to Tasks 2.5 (Gate A: coverage/golden/fold-order/neg-control/oos/retrigger-bound/batching), 2.6 + 2.8 (Gate B + empty-loadout regression), 2.7 (obs), 2.9 (smoke), 2.10 (benchmark). §4.3 float32 exactness is enforced by Gate A's exact-equality asserts.
- **Type consistency:** `score_with_jokers(...)` signature is identical in Tasks 2.2/2.6/2.9; `IndepCtx` field names used in 2.3 match its definition in 2.2; `_set`, `_dense_np`, `*_BRANCHES`, `INSCOPE_IDS`, `SPLASH_ID`, `PAREIDOLIA_ID` are all defined in 2.2 before use.
- **Known follow-ups (Phase 3):** card enhancements/editions/seals, scaling/economy/RNG jokers, Blueprint, Steel/Stone Joker, the shop/economy/acquisition. The Python engine remains the oracle.
- **Compile-time watch:** the fold unrolls ~5×(MAX_JOKERS + (1+MAX_JOKERS)·MAX_JOKERS) + 8·MAX_JOKERS + MAX_JOKERS `lax.switch`es. If trace/compile time is excessive, convert the per-card pass loop to a `lax.fori_loop`/`scan` (Task 2.2 note) — behavior identical, graph smaller. Measure in 2.10 before optimizing.

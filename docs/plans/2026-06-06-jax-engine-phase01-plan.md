# JAX-Native Balatro Engine — Phase 0 + Phase 1 Implementation Plan

> **Status — Phase 0 + Phase 1 COMPLETE (2026-06-08).** Implemented on branch `e7-jax-engine-phase01`
> (PR #32 → master) across 19 commits (Phase 0: `31ed008`..`fd17a43`; Phase 1: `cc2add3`..`a0c5b91`).
> ~76 `tests/engine_jax/` tests pass; the 1000-rollout parity gate and the PPO learning smoke are green.
> Task checkboxes below are left unchecked as the historical task list — **all are done**. (Original
> executor note: implement task-by-task with superpowers:subagent-driven-development / executing-plans.)

**Goal:** A GPU-vectorizable, branchless JAX reimplementation of Balatro's CORE loop (deal → play/discard → score → blind/ante progression → win/lose; NO jokers/shop/consumables/bosses), proven bit-for-bit equal to the Python engine by a parity harness, then dropped under the PPO trainer to run thousands of games on-device.

**Architecture:** New package `balatro_rl/engine_jax/` holding a fixed-shape pytree state + a `jit`/`vmap`-able `step`. The Python engine (`balatro_rl/engine/`) is the unchanged correctness oracle; a parity harness drives both from the same shuffled deck + action sequence and asserts identical transitions. A `JaxVectorEnv` exposes the existing `SyncVectorEnv` interface so the PPO loop is a drop-in swap.

**Tech Stack:** JAX (`jax.numpy`, `jit`, `vmap`, `lax.cond`/`lax.scan`/`lax.switch`), `chex` for shape/pytree asserts, pytest. Develop + parity-test on CPU (no GPU needed for correctness). Design: `docs/specs/2026-06-06-jax-engine-refactor-design.md`.

**Behavioral oracle:** the Python engine. Every "implement X" step's definition of done is "the parity/unit test (which encodes Python's exact values) passes." Mirror these (from the engine survey):
- `HandType` (hands.py:18): HIGH_CARD=0 … FLUSH_FIVE=11.
- `HAND_BASE` (base chips,mult @ lvl1) and `HAND_LEVEL_INC` (Δ per level) — exact tables reproduced in Task 1.2.
- Card chip values: Ace(14)→11, J/Q/K(11–13)→10, 2–10→rank.
- `reset(seed, scale, card_mods=None, enable_bosses=False)`; `step(state,(Verb,idxs))→(state,info)`; Verb.PLAY=0, DISCARD=1.
- Constants: HAND_SIZE=8, MAX_SELECT=5, HANDS_PER_BLIND=4, DISCARDS_PER_BLIND=3, STARTING_MONEY=4, NUM_ACTIONS=708, MAX_HAND=8, PLAY ids [0,218), DISCARD ids [218,436), `_SUBSETS` maps id→card-index tuple.
- Shaped reward (rewards.py): `Φ(s)=min(round_score/max(required,1),1.0)+0.05*symlog(money)+0.5*ante`; step reward `= gamma*Φ(nxt)-Φ(prev) (+1 if cleared, +10 if won)`.
- `required_score(ante, blind_index, scale)` — call the Python function for parity (do not reimplement the curve in Phase 1; expose it as a host-computed input).

---

## File Structure

- `balatro_rl/engine_jax/__init__.py` — exports `CoreState`, `reset`, `step`, `batched_reset`, `batched_step`.
- `balatro_rl/engine_jax/config.py` — `MAX_HAND`, `MAX_SELECT`, scoring tables as `jnp` arrays, `Phase`/`Verb` int constants (mirror the Python enums' int values).
- `balatro_rl/engine_jax/state.py` — `CoreState` (NamedTuple pytree of fixed-shape arrays).
- `balatro_rl/engine_jax/scoring.py` — `detect_hand_type`, `score_core`.
- `balatro_rl/engine_jax/step.py` — `reset`, `step` (single-game, branchless); `batched_reset`/`batched_step` = `vmap`.
- `balatro_rl/engine_jax/obs.py` — `encode_core(state)` → obs dict matching `OBS_SHAPES`; `legal_mask_core(state)`.
- `balatro_rl/engine_jax/rewards.py` — `shaped_core(prev, nxt, cleared, won, gamma)`.
- `balatro_rl/envs/jax_vec_env.py` — `JaxVectorEnv` (SyncVectorEnv-compatible) for the PPO loop.
- `tests/engine_jax/parity_util.py` — harness: run Python + JAX from one deck+action-seq, extract comparable core fields.
- `tests/engine_jax/test_*.py` — unit + parity tests per task.

Card encoding (used everywhere): a card is `(rank:int8 in 2..14, suit:int8 in 0..3)`. Hand/deck are parallel `rank[]`/`suit[]` arrays. Empty slot = rank 0.

---

## PHASE 0 — scaffold + parity harness

### Task 0.1: Package + config constants

**Files:** Create `balatro_rl/engine_jax/__init__.py`, `balatro_rl/engine_jax/config.py`; Test `tests/engine_jax/test_config.py`.

- [ ] **Step 1: Write the failing test**
```python
# tests/engine_jax/test_config.py
import jax.numpy as jnp
from balatro_rl.engine_jax import config as C
from balatro_rl.envs.actions import MAX_HAND, MAX_SELECT, NUM_ACTIONS

def test_constants_mirror_python():
    assert C.MAX_HAND == MAX_HAND == 8
    assert C.MAX_SELECT == MAX_SELECT == 5
    assert C.NUM_ACTIONS == NUM_ACTIONS == 708
    assert C.HANDS_PER_BLIND == 4 and C.DISCARDS_PER_BLIND == 3

def test_score_tables_shapes():
    assert C.HAND_BASE_CHIPS.shape == (12,) and C.HAND_BASE_MULT.shape == (12,)
    assert C.HAND_INC_CHIPS.shape == (12,) and C.HAND_INC_MULT.shape == (12,)
    # spot-check the table (PAIR=1): base (10,2), inc (15,1)
    assert int(C.HAND_BASE_CHIPS[1]) == 10 and int(C.HAND_BASE_MULT[1]) == 2
    assert int(C.HAND_INC_CHIPS[1]) == 15 and int(C.HAND_INC_MULT[1]) == 1
```
- [ ] **Step 2: Run → fails** (`python -m pytest tests/engine_jax/test_config.py -q`; ModuleNotFound).
- [ ] **Step 3: Implement `config.py`** — define `MAX_HAND=8`, `MAX_SELECT=5`, `NUM_ACTIONS=708`, `HANDS_PER_BLIND=4`, `DISCARDS_PER_BLIND=3`, `STARTING_MONEY=4`, `N_HAND_TYPES=12`, `Phase` ints (PLAYING=0,WON=1,LOST=2), `Verb` ints (PLAY=0,DISCARD=1), and the four `jnp.array` scoring tables in HandType order:
  - `HAND_BASE_CHIPS = jnp.array([5,10,20,30,30,35,40,60,100,120,140,160])`
  - `HAND_BASE_MULT  = jnp.array([1,2,2,3,4,4,4,7,8,12,14,16])`
  - `HAND_INC_CHIPS  = jnp.array([10,15,20,20,30,15,25,30,40,35,40,50])`
  - `HAND_INC_MULT   = jnp.array([1,1,1,2,3,2,2,3,4,3,4,3])`
- [ ] **Step 4: Run → passes.**
- [ ] **Step 5: Commit** (`git add balatro_rl/engine_jax tests/engine_jax/test_config.py && git commit -m "engine_jax: package + config constants/scoring tables"`).

### Task 0.2: `CoreState` pytree

**Files:** Create `balatro_rl/engine_jax/state.py`; Test `tests/engine_jax/test_state.py`.

`CoreState` is a `typing.NamedTuple` (auto-registers as a JAX pytree) with these fields (all `jnp` arrays, fixed shape):
`deck_rank int8[52]`, `deck_suit int8[52]`, `deck_ptr int32[]`, `hand_rank int8[8]`, `hand_suit int8[8]`, `hand_mask bool[8]`, `ante int32[]`, `blind_index int32[]`, `round_score int32[]`, `required int32[]`, `hands_left int32[]`, `discards_left int32[]`, `hand_size int32[]`, `money int32[]`, `levels int32[12]`, `hand_plays_run int32[12]`, `hand_plays_round int32[12]`, `phase int32[]`, `done bool[]`, `won bool[]`, `rng uint32[2]`.

- [ ] **Step 1: Write the failing test**
```python
# tests/engine_jax/test_state.py
import jax, jax.numpy as jnp
from balatro_rl.engine_jax.state import CoreState, zeros_state

def test_is_pytree_and_vmappable():
    s = zeros_state()
    leaves = jax.tree_util.tree_leaves(s)
    assert all(isinstance(x, jnp.ndarray) for x in leaves)
    # batchable: stack 4 states along a new leading axis, still a valid pytree
    b = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 4), s)
    assert b.hand_rank.shape == (4, 8)

def test_field_shapes():
    s = zeros_state()
    assert s.deck_rank.shape == (52,) and s.hand_rank.shape == (8,)
    assert s.levels.shape == (12,) and s.round_score.shape == ()
```
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** `CoreState(NamedTuple)` + `zeros_state()` returning all-zero arrays of the right dtype/shape.
- [ ] **Step 4: Run → passes. Step 5: Commit.**

### Task 0.3: Parity harness infra

**Files:** Create `tests/engine_jax/parity_util.py`; Test `tests/engine_jax/test_parity_harness.py`.

The harness extracts the *comparable core fields* from a Python `GameState` and from a `CoreState` into a plain dict, and runs a scripted rollout on both. Card order/values must align.

- [ ] **Step 1: Write `parity_util.py`** with:
  - `python_core_fields(gs) -> dict`: pull `ante, blind_index, round_score, required, hands_left, discards_left, hand_size, money, phase(int), done, won`, plus `hand` as a sorted list of `(rank, suit)` for set-comparison and `levels` tuple. (Sort hand because slot order need not match; the parity test compares the multiset of held cards + the scalars + obs separately.)
  - `jax_core_fields(cs) -> dict`: same shape from a `CoreState`.
  - `deck_from_python(gs) -> (ranks, suits)`: the post-reset deck order (remaining deck + hand in draw order) so the JAX engine can be seeded identically. (Read `gs.deck` and `gs.hand`; reconstruct the full 52-card draw order: hand was drawn first, then deck holds the rest.)
  - `assert_states_equal(py, jx)`: asserts every scalar equal and the held-card multiset equal.
- [ ] **Step 2: Write the failing test** that imports the harness and runs the Python engine alone (no JAX yet) to validate `python_core_fields` + `deck_from_python` on a `reset(seed=0, scale=0.2, enable_bosses=False)`:
```python
# tests/engine_jax/test_parity_harness.py
from balatro_rl.engine import engine
from tests.engine_jax.parity_util import python_core_fields, deck_from_python
def test_python_extractors():
    gs = engine.reset(0, 0.2, None, False)
    f = python_core_fields(gs)
    assert f["ante"] == 1 and f["hands_left"] == 4 and f["discards_left"] == 3
    assert len(f["hand"]) == 8
    ranks, suits = deck_from_python(gs); assert len(ranks) == 52
```
- [ ] **Step 3: Run → make it pass** (implement the extractors against the real `GameState`/`Card` fields; inspect `balatro_rl/engine/state.py` + `cards.py` for `Card.rank`/`Card.suit`). **Step 4: Commit.**

### Task 0.4: `reset` (deck-seeded) + reset parity

**Files:** Create `balatro_rl/engine_jax/step.py` (just `reset` for now); Test `tests/engine_jax/test_reset_parity.py`.

`reset(deck_rank, deck_suit, required, scale_unused=1.0) -> CoreState`: takes the full 52-card draw order (host-provided so it matches Python) + the host-computed `required` (from `engine.required_score(1,0,scale)`), draws the first `HAND_SIZE=8` into `hand`, sets `deck_ptr=8`, `ante=1`, `blind_index=0`, `round_score=0`, `hands_left=4`, `discards_left=3`, `hand_size=8`, `money=4`, `levels/plays=…`, `phase=PLAYING`, `done=False`, `won=False`.

- [ ] **Step 1: Write the failing parity test**
```python
# tests/engine_jax/test_reset_parity.py
from balatro_rl.engine import engine
from balatro_rl.engine_jax import step as J
from tests.engine_jax.parity_util import deck_from_python, python_core_fields, jax_core_fields, assert_states_equal
def test_reset_matches_python():
    gs = engine.reset(0, 0.2, None, False)
    ranks, suits = deck_from_python(gs)
    cs = J.reset(ranks, suits, required=gs.required)
    assert_states_equal(python_core_fields(gs), jax_core_fields(cs))
```
- [ ] **Step 2: Run → fails. Step 3: Implement `reset`. Step 4: Run → passes. Step 5: Commit.**

> **Phase 0 gate:** `python -m pytest tests/engine_jax -q` green; the harness can seed both engines identically and compare. This proves the approach end-to-end before any game logic.

---

## PHASE 1 — core loop, scoring, obs, PPO adapter

### Task 1.1: Hand-type detection (branchless)

**Files:** Create `balatro_rl/engine_jax/scoring.py` (`detect_hand_type`); Test `tests/engine_jax/test_handtype.py`.

`detect_hand_type(ranks int8[5], suits int8[5], mask bool[5]) -> int32` returns the HandType for the played cards (1–5 cards; `mask` marks valid). Branchless: compute rank-count histogram (`jnp.bincount`-style via `segment_sum` over 13 buckets), suit histogram, straight test (sorted distinct consecutive, with Ace-low A-2-3-4-5 and Ace-high 10-J-Q-K-A), then select the highest matching type with `jnp.where` chains. FIVE_OF_A_KIND/FLUSH_HOUSE/FLUSH_FIVE require 5 cards.

- [ ] **Step 1: Write unit tests with concrete hands** (use the Python `hands.py` classifier as the oracle for a battery of random hands, AND explicit cases):
```python
# tests/engine_jax/test_handtype.py
import jax.numpy as jnp
from balatro_rl.engine_jax.scoring import detect_hand_type
def H(cards):  # cards: list of (rank,suit); pad to 5
    r=[c[0] for c in cards]; s=[c[1] for c in cards]; m=[True]*len(cards)
    while len(r)<5: r.append(0); s.append(0); m.append(False)
    return int(detect_hand_type(jnp.array(r,jnp.int8),jnp.array(s,jnp.int8),jnp.array(m,bool)))
def test_pair(): assert H([(5,0),(5,1)]) == 1            # PAIR
def test_flush(): assert H([(2,0),(5,0),(9,0),(11,0),(13,0)]) == 5   # FLUSH
def test_straight(): assert H([(5,0),(6,1),(7,2),(8,3),(9,0)]) == 4  # STRAIGHT
def test_straight_flush(): assert H([(5,0),(6,0),(7,0),(8,0),(9,0)]) == 8
def test_wheel_straight(): assert H([(14,0),(2,1),(3,2),(4,3),(5,0)]) == 4  # A-2-3-4-5
def test_high_card(): assert H([(2,0),(7,1),(9,2),(11,3),(13,0)]) == 0
```
- [ ] **Step 2: Add a randomized parity test** vs `balatro_rl/engine/hands.py`'s classifier over 500 random 5-card hands (loop in Python, compare ints).
- [ ] **Step 3: Run → fails. Step 4: Implement `detect_hand_type`. Step 5: Run → passes. Step 6: Commit.**

### Task 1.2: Base scoring

**Files:** `scoring.py` (`score_core`); Test `tests/engine_jax/test_scoring.py`.

`score_core(ranks, suits, mask, levels int32[12]) -> (hand_type int32, chips int32, mult int32, score int32)`:
1. `ht = detect_hand_type(...)`.
2. `lvl = levels[ht]`; `base_chips = HAND_BASE_CHIPS[ht] + HAND_INC_CHIPS[ht]*(lvl-1)`; `mult = HAND_BASE_MULT[ht] + HAND_INC_MULT[ht]*(lvl-1)`.
3. Add **scoring-card** chips: only cards that participate in `ht` score (no jokers). Phase-1 rule to mirror Python `scoring.score_play` with empty jokers: determine the scoring subset per hand type (e.g. PAIR → the 2 paired cards; FLUSH/STRAIGHT/straight-flush/5-kind/full-house/flush-house/flush-five → all 5; HIGH_CARD → the single highest card). Card chip = Ace→11, 11–13→10, else rank. `chips = base_chips + sum(scoring_card_chips)`.
4. `score = chips * mult`.

- [ ] **Step 1: Write unit tests with exact values:**
```python
# pair of 5s, level 1: base_chips 10 + (5+5)=20, mult 2 -> score 40
# two pair (5,5,9,9) lvl1: base 20 + (5+5+9+9=28)=48, mult 2 -> 96
```
- [ ] **Step 2: Add randomized parity** vs `scoring.score_play(played, jokers=(), levels=...)` over 500 random plays — compare `(hand_type, chips, mult, score)`. (Disable enhancements/editions: build plain cards.)
- [ ] **Step 3: Run → fails. Step 4: Implement. Step 5: Run → passes. Step 6: Commit.**

### Task 1.3: `step` — PLAY/DISCARD + draw

**Files:** `step.py` (`step`); Test `tests/engine_jax/test_step_parity.py`.

`step(state: CoreState, verb int32, sel_mask bool[8]) -> (CoreState, reward_signals)`. (The flat action id is decoded to `(verb, sel_mask)` in the env adapter, Task 1.7; the engine takes the decoded form for testability.) Logic, branchless via `lax.cond`/`jnp.where`:
- **PLAY:** score the selected cards (`score_core` over the masked hand), `round_score += score`, `hands_left -= 1`; remove selected from hand; refill empty slots from `deck[deck_ptr:]` in order up to `hand_size`; advance `deck_ptr`. Then check blind clear (`round_score >= required`) → Task 1.4 progression; or loss if `hands_left==0 and round_score<required`.
- **DISCARD:** require `discards_left>0`; remove selected, refill, `discards_left -= 1`. (No scoring.)
- Return the next state + a small struct of `(cleared, won, hand_type, score)` for the reward/info.

Mirror Python's **exact draw order** (which deck cards refill which slots). Inspect `engine.py` play/discard to match slot-fill semantics.

- [ ] **Step 1: Write a parity test** that drives a *scripted legal action sequence* (e.g., always PLAY the first legal subset) on both engines from the same deck and asserts `assert_states_equal` after each step, for 30 steps × 50 seeds.
- [ ] **Step 2: Run → fails. Step 3: Implement PLAY+DISCARD+draw. Step 4: Run → passes. Step 5: Commit.**

### Task 1.4: Blind/ante progression + win/lose

**Files:** `step.py` (fold into `step`); Test extends `test_step_parity.py`.

On blind clear: `blind_index = (blind_index+1) % 3`; if it wrapped past boss → `ante += 1`; recompute `required` (host-provided table indexed by `(ante,blind_index)` — precompute a `required[ANTE_MAX,3]` array at reset from `engine.required_score`, store in state or pass in); reset `round_score=0`, `hands_left=4`, `discards_left=3`; redraw a fresh hand from the next deck cards (mirror Python: reshuffle? In core, Python redraws from the continuing deck — confirm and match). `won = ante > WIN_ANTE` (mirror Python's win condition). On loss: `phase=LOST, done=True`. On win: `phase=WON, won=True, done=True`.

- [ ] **Step 1: Extend the parity test** to run until episodes terminate (cap 300 steps), asserting equal `done/won/phase/ante` trajectories across 50 seeds with a "play-greedy" scripted policy.
- [ ] **Step 2: Run → fails. Step 3: Implement progression. Step 4: Run → passes. Step 5: Commit.**

### Task 1.5: Obs encoding + legal mask (core)

**Files:** Create `balatro_rl/engine_jax/obs.py`; Test `tests/engine_jax/test_obs_parity.py`.

`encode_core(state) -> dict` producing the `OBS_SHAPES` dict (import shapes from `balatro_rl/envs/obs.py`) with CORE fields filled and all joker/shop/consumable/pack/voucher/boss fields zeroed/masked. Match the Python `encode(state)` global layout exactly (g[0]=symlog(round_score), g[1]=symlog(required), g[2]=ratio, g[3]=hands_left, g[4]=discards_left, g[5]=symlog(money), g[6]=ante, g[7]=blind_index, g[9]=hand_size, g[12]=phase one-hot; jokers/shop counts=0) and the hand 8×37 card features (rank/suit one-hots; enhancement/edition/seal = NONE; debuff=0). `legal_mask_core(state)` → bool[708]: PLAY ids `[0,218)` and DISCARD ids `[218,436)` enabled only where the subset's indices are all `< hand_count` and (for discard) `discards_left>0`; everything else False. Reuse `_SUBSETS` from `balatro_rl/envs/actions.py` as a static `jnp` index table.

- [ ] **Step 1: Write a parity test**: for a battery of core states reached by the scripted rollout, assert `encode_core(cs)` equals Python `encode(gs)` per key (allclose for floats) and `legal_mask_core` equals Python `legal_mask` on `[0,436)` (core action range).
- [ ] **Step 2: Run → fails. Step 3: Implement. Step 4: Run → passes. Step 5: Commit.**

### Task 1.6: Shaped reward (core)

**Files:** Create `balatro_rl/engine_jax/rewards.py`; Test `tests/engine_jax/test_reward_parity.py`.

`shaped_core(prev, nxt, cleared, won, gamma=0.999) -> float32` = `gamma*Φ(nxt) - Φ(prev) + 1.0*cleared + 10.0*won`, `Φ(s)=min(round_score/max(required,1),1)+0.05*symlog(money)+0.5*ante`.

- [ ] **Step 1: Parity test** vs `balatro_rl/envs/rewards.py` Shaped over the scripted rollout transitions (allclose). **Step 2–5: implement, test, commit.**

### Task 1.7: Batched env (`vmap`) + flat-action adapter

**Files:** `step.py` (`batched_reset`, `batched_step`, `decode_action`); Test `tests/engine_jax/test_batched.py`.

`decode_action(action_id int32) -> (verb int32, sel_mask bool[8])` from the static `_SUBSETS` table. `batched_step = jax.vmap(step_with_action)`, `batched_reset = jax.vmap(reset)`. Wrap `step` so a done env auto-resets (mirror SyncVectorEnv) using a per-env fresh deck (carry a key in state; `jax.random.permutation` for the shuffle when standalone — parity tests still use host decks).

- [ ] **Step 1: Test** that `batched_step` over N=1024 envs returns obs dict with leading dim 1024, runs under `jit`, and one step matches the single-env `step` for each lane (vmap consistency). **Step 2–5: implement, test, commit.**

### Task 1.8: `JaxVectorEnv` PPO adapter + on-device rollout

**Files:** Create `balatro_rl/envs/jax_vec_env.py`; Test `tests/envs/test_jax_vec_env.py`.

`JaxVectorEnv(num_envs, reward_name="shaped", base_seed=0, req_scale=1.0, enable_bosses=False)` exposing `reset()->(obs_dict, masks)`, `step(actions int32[N])->(obs,rewards,dones,infos,masks)`, `set_req_scale`, `set_boss_rate` (no-op in core). Internally holds a batched `CoreState`; `step` calls `batched_step` (jit'd). `infos` provides `ante/round_score/verb/cleared` per env (as arrays or a light list) — enough for `train.py`'s curriculum + logging. (`enable_bosses` ignored in core; assert False or warn.)

- [ ] **Step 1: Test** `JaxVectorEnv(8)` reset/step return the SyncVectorEnv-shaped tuples/keys; a 5-step random-legal rollout runs without error and dones auto-reset.
- [ ] **Step 2: Run → fails. Step 3: Implement. Step 4: Run → passes. Step 5: Commit.**

### Task 1.9: Wire into PPO + learning smoke

**Files:** Modify `balatro_rl/agent/train.py` (env factory switch); Test `tests/agent/test_jax_engine_smoke.py`.

Add a `TrainConfig.engine: str = "python"` knob; when `"jax"`, build `JaxVectorEnv` instead of `SyncVectorEnv` (one factory line). No other train-loop change (interfaces match).

- [ ] **Step 1: Write a smoke test**: `train(TrainConfig(engine="jax", num_envs=64, num_steps=16, num_updates=2, d_model=64, eval_interval=0))` runs end-to-end and returns params; assert it produced a finite loss and stepped the JAX env. **Step 2: Run → fails. Step 3: Implement the factory switch. Step 4: Run → passes. Step 5: Commit.**

### Task 1.10: Full core parity gate + throughput benchmark

**Files:** Test `tests/engine_jax/test_core_parity_gate.py`; script `scripts/bench_jax_engine.py`.

- [ ] **Step 1: Parity gate test** — 1000 (seed × random-legal-action-seq) rollouts to termination; assert Python and JAX agree on the full transition (held-card multiset, all scalars, obs, reward, done/won) at every step. Mark `@pytest.mark.slow` if needed; a 200-rollout subset runs in CI.
- [ ] **Step 2: Run → fix any mismatches** (each failure is a real engine bug — debug against the Python oracle). **Step 3: Commit.**
- [ ] **Step 4: Write `scripts/bench_jax_engine.py`** — time `batched_step` over a `lax.scan` rollout at `num_envs∈{1k,10k,50k}`, report env-steps/sec; on a CUDA box also print `nvidia-smi` util. Document the numbers in `docs/RUNPOD_M2.md` (replace the §8 "measured" note with the new throughput).
- [ ] **Step 5: Commit.**

> **Phase 1 gate (the whole-bet go/no-go):** (1) `test_core_parity_gate` green on ≥1000 rollouts; (2) `bench_jax_engine.py` shows ≥10k envs at GPU ≥~80% and ≫100× the Python env-steps/sec; (3) the learning smoke shows blinds-cleared rising. If all three hold, Phase 2 (jokers) is justified.

---

## Notes for the executor
- **Develop on CPU** (`JAX_PLATFORMS=cpu`); correctness/parity needs no GPU. Only Task 1.10's benchmark wants a GPU.
- **Branchless discipline:** no Python `if` on traced values inside `step`/`scoring` — use `jnp.where`/`lax.cond`/`lax.select`. `chex.assert_shape` liberally.
- **The Python engine is the spec.** When unsure about a rule (draw order, scoring subset, required curve, win ante), read it in `balatro_rl/engine/` and mirror; the parity test is the arbiter, not intuition.
- **Branch hygiene:** work on a feature branch (`e7-jax-engine-phase01`), not master.

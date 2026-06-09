# JAX-Native Balatro Engine — Phase 2 (Jokers + Consumables) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans — implement task-by-task, parity-gated. Checkbox (`- [ ]`) = task tracking.

**Date:** 2026-06-09  **Branch:** `e7-jax-engine-phase02`  **Builds on:** Phase 0+1 (core engine, on master).
**Design:** `docs/specs/2026-06-06-jax-engine-refactor-design.md` §4 (Phase 2).

## Goal
Extend the parity-gated JAX **core** engine (`balatro_rl/engine_jax/`) with the **~25–45 highest-impact,
deterministic jokers** + the **12 Planet consumables**, keeping every kernel branchless / `jit` / `vmap`-able,
with the Python engine (`balatro_rl/engine/`) as the unchanged oracle.

## Scope & the RNG cut (important)
There is **no shop in the JAX engine yet** (that's Phase 3), so jokers/consumables are **granted host-side**
(mirroring how `reset()` already takes a host deck): both engines are driven from the **same loadout + deck +
action sequence** and asserted identical.

**In scope:** deterministic effects only — flat `+chips`/`+mult`, conditional `xmult`, hand-contains `xmult`,
state-reading, retrigger, simple counter-scaling jokers; **all 12 Planets** (`levels[ht] += 1`).

**DEFERRED to Phase 2b** (and *why*): every **rng-consuming** joker (Misprint, Bloodstone, Business Card,
Reserved Parking, self-destroy rolls) — **JAX uses `jax.random` (uint32[2] key) while Python uses splitmix64;
the streams differ, so rng rolls cannot be parity-checked**; **enhancement-lifecycle** jokers (Steel/Stone/
Glass/Lucky/Vampire/Midas, Golden Ticket, Rough Gem) — need card enhancement/edition/seal fields not in
`CoreState`; **per-round-random** (Ancient Joker, The Idol, Mail-In Rebate); **on-round-end-money** jokers and
**Popcorn** — the JAX core has no cash-out/economy phase; and **all 20 Tarots** — need card-mutation +
master-deck identity tracking absent from `CoreState`.

## Behavioral oracle: `balatro_rl/engine/scoring.py` `score_play` + `jokers/{base,library}.py`
Mirror the **fold order** exactly: (1) scoring cards L→R **with retriggers** (card chips + `on_score` effects +
card mods), (2) held cards (`on_held`), (3) independent jokers in slot order. **Additive (`+chips`/`+mult`)
applies before multiplicative (`xmult`) within each Effect.** `score = floor(chips * mult)`.

## State additions (`state.py` `CoreState` + `zeros_state` + `reset`/`reset_jax` + `step_with_action` auto-reset)
- `joker_ids: int32[MAX_JOKERS]` — `JokerType` int code per slot; `0` = empty.
- `joker_counters: float32[MAX_JOKERS]` — per-instance scaling counter (mirrors Python `js.counter`, a float); 0.0.
- `joker_mask: bool[MAX_JOKERS]` — occupied (`== joker_ids != 0`); explicit, matches deck/hand-mask convention.
- `consumable_ids: int32[MAX_CONSUMABLES]` — `PlanetType` 1..12 per slot; `0` = empty.

`config.py`: `MAX_JOKERS = 5` (== `engine.JOKER_SLOTS`), `MAX_CONSUMABLES = 2` (== `consumable_slots`),
`Verb.USE = 2`. New `engine_jax/jokers.py`: `JokerType` int codes **mirroring `engine/jokers/base.py` exactly**
+ a `PLANET_HAND int32[13]` lookup (PlanetType→HandType) mirroring `consumables.PLANET_HAND`.

**dtype:** the joker path returns `mult` as **float32** (jokers add `xmult`/float `+mult`); `score = floor(chips*mult)`
to match Python `int(ctx.chips*ctx.mult)`. Keep the existing `score_core` (int-mult) as-is; add a separate
`score_core_jokers` so Phase-1 tests stay green and the **empty-loadout path provably reduces** to `score_core`.

## Branchless fold (`scoring.py` `score_core_jokers(...) -> (ht, chips int32, mult float32, score int32)`)
Inputs: `sel_*` (scoring cards), `held_*` (kept cards), `levels`, `joker_ids/joker_counters/joker_mask`,
`money, discards_left, deck_count`, `hand_plays_run[ht], hand_plays_round[ht]` (PRE-increment), `contains_mask` (bool[12]).
- **Phase A — retrigger:** `retrig[c] = 1 + Σ_s mask[s]·retrig(id[s], card)` (HACK: ranks 2–5; SOCK_AND_BUSKIN: faces).
- **Phase B — `on_score` (×retrig):** base chips `+= Σ retrig[c]·card_chip`; per joker a `jnp.where(mask[s],…)` of its
  per-card additive effect (suit/rank/face), then PHOTOGRAPH ×2 on the first scoring face (per retrigger). Accumulate
  all additive `on_score` deltas first, then apply Photograph's ×2 — **verify equivalence in the gate**.
- **Phase C — `on_held`:** BARON ×1.5 per held King; BLACKBOARD ×3 if all held suits ∈ {0,2}.
- **Phase D — independent (slot order):** per slot `(add_chips, add_mult, xmult)` via `jnp.where` dispatch on
  `joker_ids[s]`, applied additive-then-xmult. (See the joker list below.)
`score = jnp.floor(chips.astype(f) * mult).astype(int32)` — **final product in float64** (cheap, scalar) to match
Python float64, or document a ±1 tolerance if x64 is off (decide in Task 2.1).

## Grant mechanism (no shop)
Host-provided loadout, mirroring the host deck. `reset(... joker_ids=None, joker_counters=None, consumable_ids=None)`
defaults empty; `joker_mask = (joker_ids != 0)`. `reset_jax(key, required_table, joker_ids=None, …)` broadcasts one
loadout across the batch (`in_axes=(0, None, None, …)`); `step_with_action` auto-reset re-grants the same loadout.
Python oracle: `dataclasses.replace(engine.reset(...), jokers=tuple(JokerState(type=…)…), consumables=…)`. Add
`make_loadout(joker_ids, consumable_ids)` + a `joker_counters` slot-by-slot extractor to `parity_util.py`.

## Parity strategy (two-tier, mirrors Phase 1)
1. **Scoring parity** (gates each joker batch): randomized `score_core_jokers` vs `score_play(jokers=loadout, …)`
   over (seed × loadout × hand × levels × state-scalars); assert `hand_type`, `chips` (int), `mult` (~1e-6), `score` (exact).
2. **Full-rollout gate** (extends `test_core_parity_gate.py`): grant both engines the same loadout, random-legal
   rollouts to termination; assert `assert_states_equal` + `assert_hand_slots_equal` + per-PLAY score + `joker_counters`
   slot-by-slot at every step.

## Joker subset (~45 deterministic; grouped by fold phase)
- **Flat:** JOKER(+4m). **on_score suit:** GREEDY/LUSTY/WRATHFUL/GLUTTONOUS(+3m per suit), ARROWHEAD(+50c/spade),
  ONYX_AGATE(+7m/club). **on_score rank/face:** SCARY_FACE(+30c/face), ODD_TODD(+31c), EVEN_STEVEN(+4m),
  SCHOLAR(+20c+4m/Ace), WALKIE_TALKIE(+10c+4m per 10/4), SMILEY_FACE(+5m/face), FIBONACCI(+8m). **on_score xmult:**
  PHOTOGRAPH(×2 first face).
- **Independent state:** ABSTRACT_JOKER(+3m/joker), JOKER_STENCIL(×(empty+1)), BULL(+2c/$), BANNER(+30c/discard),
  MYSTIC_SUMMIT(+15m if 0 discards), BLUE_JOKER(+2c/deck card), HALF(+20m if ≤3 played), GROS_MICHEL(+15m, no destroy),
  CAVENDISH(×3, no destroy), SUPERNOVA(+run plays+1 m), CARD_SHARP(×3 if round plays≥1).
- **Hand-contains** (compute `contains_mask` like `engine.contains`): THE_DUO/TRIO/FAMILY/ORDER/TRIBE (×2..×4),
  Jolly/Zany/Mad/Crazy/Droll(+mult), Sly/Wily/Clever/Devious/Crafty(+chips).
- **Scoring-suit-set:** SEEING_DOUBLE(×2 club+other), FLOWER_POT(×3 all 4 suits).
- **Held:** BARON(×1.5/held King), BLACKBOARD(×3 held∈spade/club).
- **Retrigger:** HACK(2–5), SOCK_AND_BUSKIN(faces).
- **Counter-scaling** (`on_play`/`on_discard` folds, Task 2.6): RIDE_THE_BUS, RUNNER, ICE_CREAM, SQUARE_JOKER,
  SPARE_TROUSERS, WEE_JOKER, GREEN_JOKER. (POPCORN deferred — round-lifecycle.)
- **Consumables:** all 12 Planets → `levels[PLANET_HAND[id]] += 1` via a `Verb.USE` action.

---

## Tasks (TDD, parity-gated)

### Task 2.1: CoreState joker/consumable fields + host grant + dtype decision
- [ ] Add the 4 `CoreState` fields + `zeros_state`; `config` `MAX_JOKERS=5`/`MAX_CONSUMABLES=2`/`Verb.USE=2`;
  new `jokers.py` (`JokerType` codes mirrored exactly + `PLANET_HAND`). Extend `reset`/`reset_jax` loadout kwargs.
  Add `make_loadout` to `parity_util`. Decide+document the float32-vs-float64 final-product policy.
- **Gate:** `test_state.py`/`test_reset_parity.py` — reset-with-loadout sets ids/mask/consumables, survives jit/vmap,
  `assert_states_equal` still holds; Python `make_loadout` matches the spec.

### Task 2.2: Joker fold scaffold + additive `on_score` jokers
- [ ] `score_core_jokers(...)` Phase-B additive `on_score` fold + flat JOKER. Cover the on_score suit/rank/face set.
  Return float32 mult + floored int32 score; retrigger=1, held/independent empty for now.
- **Gate:** new `test_joker_scoring_parity.py` — randomized vs `score_play` over single-joker loadouts × 200 hands × levels.

### Task 2.3: Independent state-reading + hand-contains + scoring-suit-set jokers
- [ ] Phase-D dispatch: ABSTRACT_JOKER, JOKER_STENCIL, BULL, BANNER, MYSTIC_SUMMIT, BLUE_JOKER, HALF, GROS_MICHEL,
  CAVENDISH, SUPERNOVA, CARD_SHARP; contains-family (compute `contains_mask`); SEEING_DOUBLE, FLOWER_POT. Verify
  additive-before-xmult slot ordering vs Python.
- **Gate:** extend `test_joker_scoring_parity.py` — multi-joker mixes (+mult & xmult), money/plays sweeps, slot-order test.

### Task 2.4: Retrigger + held (`Baron`, `Blackboard`) + `Photograph`
- [ ] Phase-A retrigger (HACK/SOCK_AND_BUSKIN) re-applying per-card on_score ×retrig; Phase-C held fold; PHOTOGRAPH.
- **Gate:** extend parity with held cards + retrigger/photograph cases (incl. PHOTOGRAPH+SMILEY_FACE under retrigger).

### Task 2.5: Wire jokers into `step()` + within-blind rollout parity
- [ ] `step`/`step_with_action` call `score_core_jokers` with joker state + kept (held) cards + scalars; loadout rides
  through PLAY/DISCARD/advance/win + auto-reset re-grant. (Money read-only — fine for the subset.)
- **Gate:** extend `test_core_parity_gate.py` with non-counter loadouts; per-PLAY score + state parity each step.

### Task 2.6: Counter-scaling jokers (`on_play`/`on_discard` folds)
- [ ] Branchless counter updates in `step` (after score, before the hand_plays bump): RIDE_THE_BUS, RUNNER, ICE_CREAM,
  SQUARE_JOKER, SPARE_TROUSERS, WEE_JOKER, GREEN_JOKER (+ discard). **Do NOT reset counters on blind advance** (Python
  keeps them; verify).
- **Gate:** new `test_joker_counter_parity.py` + extend the gate — scripted counter growth/reset, assert
  `joker_counters[s] == state.jokers[s].counter` slot-by-slot every step + per-PLAY score.

### Task 2.7: Planet consumables — `USE` action + level-up parity
- [ ] `decode_action`: ids `[436, 436+MAX_CONSUMABLES)` → `(Verb.USE, slot)`; branchless USE branch
  `levels.at[PLANET_HAND[consumable_ids[ci]]].add(1)` + clear slot. Grant Planets host-side.
- **Gate:** new `test_consumable_parity.py` — USE each of 12 Planets, assert `levels` match + slot emptied + a
  subsequent PLAY scores identically; USE leaves other scalars unchanged.

### Task 2.8: Obs/env wiring + batched loadout + full Phase-2 gate
- [ ] Extend `encode_core` to expose joker/consumable presence (mirror the Python encoder's covered fields; zero the
  rest); `legal_mask_core` enables USE ids only where `consumable_ids != 0`. Thread the loadout through `JaxVectorEnv`
  + `batched_reset/step`. Consolidated gate over many (seed × loadout × action) rollouts.
- **Gate:** full Phase-2 `test_core_parity_gate` (≥200 rollouts; 1000 slow) — transition + obs + reward + score +
  counters; `JaxVectorEnv` batched-loadout smoke; (optional) PPO learning smoke with a fixed loadout.

> **Phase 2 gate:** all per-batch scoring-parity tests green + the full-rollout gate green across representative
> loadouts (additive / xmult-heavy / retrigger / counter-scaling / Planets) ⇒ Phase 3 (shop/economy) is justified.

## Open decisions (resolve in Task 2.1 unless noted)
- **Float precision:** do the final `chips*mult` product in **float64** inside `score_core_jokers` (recommended;
  scalar, cheap) to match Python float64 exactly, vs a documented ±1-chip tolerance with chips+mult asserted exact.
- **Joker obs layout (Task 2.8):** read `envs/obs.py` `OBS_SHAPES`; populate the joker-presence subset PPO can
  condition on, zero anything not representable (editions/sell-values out of scope).
- **Counter persistence:** Python keeps joker counters across blinds (only `hand_plays_round` resets) — JAX advance
  must carry `joker_counters` unchanged (verify Ride-the-Bus across a boundary in the gate).
- **Separate kernel:** keep `score_core` (Phase-1) untouched; add `score_core_jokers`; add a test that the
  empty-loadout joker path == `score_core` (so Phase-1 parity is preserved).

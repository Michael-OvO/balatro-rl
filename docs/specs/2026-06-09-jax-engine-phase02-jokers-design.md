# E7 Phase 2: JAX Joker Scoring Kernel + Env Integration — Design

**Status:** design (pre-plan)
**Date:** 2026-06-09
**Branch:** continues `e7-jax-engine-phase01` work (Phase 1 = PR #32)
**Implements:** `docs/specs/2026-06-06-jax-engine-refactor-design.md` §4, Phase 2 row
**Oracle:** `balatro_rl/engine/` (Python) — unchanged; remains the parity oracle.

## 1. Problem & goal

Phase 1 delivered a branchless, `vmap`/`jit`-able JAX **core** engine (deal → play/discard →
score → blind/ante → win/lose), bit-for-bit equal to the Python oracle and dropped under PPO via
`JaxVectorEnv`. It deliberately excluded jokers. Jokers are the single biggest driver of Balatro
scoring, so a jokers-blind engine trains an agent on a game that is not the real game.

**Phase 2 goal:** a JAX **joker scoring kernel** — the precise, ordered fold the Python oracle
performs when jokers are present — proven bit-for-bit equal to `engine.scoring.score_play`, and
wired into the core engine so PPO can train with jokers in play.

**The acquisition constraint.** Jokers are bought in the shop, which is **Phase 3**. So Phase 2
does not *acquire* jokers; it carries a **fixed per-episode loadout** (set at `reset`, constant
across the episode, surviving auto-reset). This isolates the hard part — the branchless scoring
math — from the acquisition game, exactly as the refactor design's phasing intends.

## 2. Scope

### 2.1 In scope

- A JAX scoring kernel `score_with_jokers(...)` that reproduces `score_play` for the **pure-scoring
  joker set** operating on **plain cards** (no enhancements/editions/seals).
- The pure-scoring set = every registered joker whose **only** hooks are `on_score`, `on_held`,
  `independent`, `retrigger`, and/or `rules` — and which has a non-trivial effect on plain cards.
  Mechanically: no `on_play` / `on_round_start` / `on_round_end` / `on_discard` / `on_hand_events`
  hook, no scoring-time RNG (`ctx.rng`), no `ctx.money_delta` mutation, no card-enhancement reads.
  This yields **~45 jokers** (enumerated in §7).
- `CoreState.jokers: int32[MAX_JOKERS]` (type-id per slot, 0 = empty).
- `reset` / `reset_jax` accept a `jokers` loadout (default all-zero ⇒ Phase-1 behavior unchanged).
- `step` computes held cards + a scalar scoring context and scores via the kernel.
- `obs.encode_core` fills the joker observation keys (`joker_types`, `joker_counter`, `joker_mask`)
  and `global[10]` (joker count) to match `envs.obs.encode`.
- `JaxVectorEnv` + `TrainConfig` gain a `joker_loadout` knob so PPO trains with jokers.
- **Consumables (Planets):** already honored — Planets only bump `levels[ht]`, which `CoreState`
  carries and the kernel reads. No new mechanic is needed; we add a unit test asserting a leveled
  loadout scores at parity. (Tarots mutate cards ⇒ deferred to Phase 3 with the shop.)
- Two parity gates (§5) + a PPO learning smoke + a throughput re-benchmark.

### 2.2 Out of scope (deferred)

- **Shop / economy / acquisition / vouchers / packs** → Phase 3.
- **Card enhancements / editions / seals** (Bonus/Mult/Glass/Steel/Stone, Foil/Holo/Poly, Gold/Red
  seals) → Phase 3 (they are *produced* by Tarots/shop; a plain Phase-2 deck has none).
- **Stateful / scaling jokers** (Ride the Bus, Runner, Ice Cream, Square Joker, Spare Trousers, Wee
  Joker, Popcorn, Green Joker, Obelisk) — need per-joker mutable `counter` + `on_play` lifecycle.
- **Economy / lifecycle jokers** (Golden Joker, Egg, To the Moon, Delayed Gratification, Faceless
  Joker, Cavendish & Gros Michel self-destruct).
- **Scoring-RNG jokers** (Misprint, Bloodstone, Business Card, Reserved Parking, Lucky Cat, Glass
  Joker) and **per-round-randomized** jokers (Ancient Joker, The Idol, Mail-In Rebate).
- **Card-mutation jokers** (Vampire, Midas Mask) and the **copy** joker (Blueprint).
- **Steel Joker / Stone Joker** — pure-scoring (only `independent`) but read the full-deck
  enhancement histogram; on a plain deck they contribute 0, so they are deferred with enhancements
  rather than shipped as no-ops.
- **Bosses** (Flint halving, debuffs) — still off (Phase 4).

These boundaries keep Phase 2 **deterministic** (no RNG in the scored path) and **stateless** (no
per-joker counters), so both parity gates compare pure functions.

## 3. The oracle fold this kernel must reproduce

`engine.scoring.score_play` (verified at `balatro_rl/engine/scoring.py`) computes, in order:

1. **Hand type + scoring set.** `evaluate(played, rules)` → `(hand_type, scoring_idx)`. `rules` is
   `aggregate_rules(jokers)`; `splash` widens `scoring_idx` to **all** played cards; the hand *type*
   is unchanged. (Stone forcing is out of scope — no enhancements.)
2. **Leveled base.** `base_chips, mult = leveled_base(hand_type, levels)` =
   `HAND_BASE + HAND_INC*(lvl-1)`. `chips` is integer-valued; `mult` is a float accumulator.
3. *(Flint boss — skipped, no boss. Blueprint resolve — skipped, excluded.)*
4. **Scored cards, left→right, with retriggers.** For each card in `scoring_idx`:
   - `retriggers = sum(eff.retrigger(...))` over jokers (+ Red seal — none here).
   - For each of `1 + retriggers` passes: `chips += card_chip(card)`, then **fold every joker's
     `on_score` Effect left→right** applying `chips += e.chips; mult += e.mult; mult *= e.xmult`
     immediately. (Card mods — none on plain cards.)
5. **Held cards, left→right.** Fold every joker's `on_held` Effect (Baron). (No Steel — plain.)
6. **Independent jokers, slot order.** Fold every joker's `independent` Effect.
7. *(Glass shatter — skipped, no Glass.)*
8. **Final:** `score = int(chips * mult)` (floor).

**The ordering is load-bearing.** Effects apply *immediately* in slot order, so a slot-1 `×mult`
multiplies before a slot-2 `+mult` adds. A "sum all +, then apply all ×" shortcut would break
parity. The kernel reproduces the exact interleaved sequence.

**Context the in-scope jokers read — and where it comes from.** Every field is already in
`CoreState` or derivable in `step` (data-availability proof; no new state beyond `jokers`):

| Context field | Read by | Source in `CoreState` / `step` |
|---|---|---|
| `played` count | Half | popcount of the played selection |
| scoring-set suits | Seeing Double, Flower Pot | suits of cards in `scoring_mask` |
| `held` cards | Baron, Blackboard | hand slots minus the played selection |
| `first_face_idx` | Photograph | first scoring face slot (respects `all_face`) |
| `rules` (splash / all_face) | Splash, Pareidolia + every face/score reader | reduced over `jokers` |
| `n_jokers` / `empty_joker_slots` | Abstract Joker, Joker Stencil | `count(jokers != 0)` / `JOKER_SLOTS − n_jokers` |
| `money` | Bull | `state.money` |
| `discards_left` | Banner, Mystic Summit | `state.discards_left` |
| `deck_count` | Blue Joker | `52 − deck_ptr` (pre-refill) |
| `hand_plays_run[ht]` / `_round[ht]`, pre-increment | Supernova, Card Sharp | `state.hand_plays_run/round[ht]` read **before** `step` increments |

## 4. Architecture

### 4.1 State (`engine_jax/state.py`)

Add one field to `CoreState`:

```python
jokers: jnp.ndarray   # int32[MAX_JOKERS]  JokerType id per slot, 0 = empty
```

`MAX_JOKERS = 6` (matches `envs.actions.MAX_JOKERS` and the obs `joker_types` shape; loadouts use
≤5). No per-joker counter/edition arrays (excluded by scope). `zeros_state()` seeds all-zero ⇒ the
empty loadout, which makes `score_with_jokers` reduce **exactly** to Phase-1 `score_core`,
preserving every Phase-1 parity test unchanged.

### 4.2 Scoring kernel (`engine_jax/scoring.py`)

New `score_with_jokers(played_rank, played_suit, played_mask, held_rank, held_suit, held_mask,
levels, jokers, ctx_scalars) -> (hand_type, chips, mult, score)` using **Approach A — unrolled
ordered fold + `lax.switch`**:

- **Dense id map.** A static table maps each in-scope sparse `JokerType` id → a dense index
  `0..K-1`; id 0 (empty) and any out-of-scope id → a no-op branch. Built once at import.
- **Rule aggregation.** `splash = any(jokers == SPLASH)`, `all_face = any(jokers == PAREIDOLIA)`,
  reduced over the loadout. `splash` selects `scoring_mask = played_mask` (else the Phase-1
  hand-type subset from `score_core`'s logic); `all_face` forces the face predicate True.
- **`first_face_idx`** = lowest scoring-card slot whose card is a face (respecting `all_face`),
  else a sentinel (−1) so Photograph never fires.
- **The fold.** `chips` (int32) and `mult` (float32) accumulators. Static nested loops over the
  fixed bounds (`MAX_SELECT=5` scoring slots × `MAX_JOKERS=6` joker slots for `on_score`;
  `MAX_HAND=8` held × `MAX_JOKERS` for `on_held`; `MAX_JOKERS` for `independent`), each calling
  `lax.switch(dense_id, joker_branches)` to get that joker's `(Δchips, Δmult, ×mult)` for the
  current card/context, applied immediately. Retriggers: each on_score slot is folded
  `1 + retrigger_count` times (retrigger count itself a `lax.switch` over the loadout, summed),
  with a **static** unroll bound. Each joker's `retrigger` returns ≤1, so total retriggers per card
  ≤ `MAX_JOKERS` ⇒ the safe exact bound is `1 + MAX_JOKERS` passes (covers duplicate Hacks and the
  Hack+Sock&Buskin+Pareidolia case where a low card counts as a face and both fire). A test asserts
  the oracle never exceeds it.
- **Final:** `score = floor(chips_f32 * mult).astype(int32)`.

Each joker branch is a tiny pure function of `(card_rank, card_suit, is_face, slot_ctx)` returning
`(Δchips:int, Δmult:float, ×mult:float)`. Adding a joker = adding one branch + one dense-map row.

### 4.3 Numerical parity (float32 is exact here — a real risk, mitigated)

Python accumulates `mult` in float64. The kernel uses float32. This is **bit-exact for the
in-scope set** because every multiplicative factor is an *exact dyadic/integer* value
(×1.5 = 3/2, ×2, ×3, ×4, ×(empty_slots+1)), every additive value is an integer, and all
intermediate magnitudes stay well below 2²⁴ ≈ 16.7M for core-game (low-ante, ≤5-joker) play —
the regime where float32 represents these values *exactly*, so float32 and float64 agree and
`int(chips*mult)` floors identically. The component gate (§8.A) asserts exact equality over
randomized loadouts/hands and would catch any boundary violation. **No non-dyadic xmult is in
scope** (Ramen's ×0.01, Misprint, etc. are all excluded), so the property holds by construction.
If Phase 3 introduces a non-dyadic factor, revisit with `jax_enable_x64` for the scoring kernel.

### 4.4 Step integration (`engine_jax/step.py`)

- `reset` / `reset_jax` / `batched_reset` gain a `jokers` argument (int32[MAX_JOKERS], default
  zeros) stored into `CoreState.jokers`.
- `step` computes **held cards** (hand slots minus the played selection) and the **scalar context**
  (money, discards_left, deck_count = 52 − deck_ptr = undrawn slots, as in the Phase-1 obs and
  matching the oracle's `len(state.deck)`, computed at score time **before** the play's refill;
  n_jokers = count of nonzero loadout slots,
  hand_plays_run/round for the detected hand type, pre-increment), then calls `score_with_jokers`
  instead of `score_core`. The blind/ante/win/lose machinery is unchanged.
- `step_with_action` auto-reset preserves the loadout (the fresh state re-seeds with the same
  `jokers`).

### 4.5 Observation (`engine_jax/obs.py`)

Fill the three joker keys + `global[10]` from `state.jokers`:
`joker_types[i] = jokers[i]`, `joker_mask[i] = (jokers[i] != 0)`,
`joker_counter[i] = symlog(0) = 0` (stateless ⇒ counter always 0), `global[10] = count(jokers!=0)`.
Everything else stays zeroed. Confirmed: the Python encoder's `aggregate_rules` call
(`envs/obs.py:145`) feeds only `boss_debuffed_idx`, which is empty with no boss — so Splash /
Pareidolia in the loadout do **not** alter the observation, and no rules-dependent obs handling is
needed.

### 4.6 Env + trainer (`envs/jax_vec_env.py`, `agent/train.py`)

`JaxVectorEnv(... , joker_loadout=None)` — `None` ⇒ empty (Phase-1 behavior); a list of type-ids ⇒
that fixed loadout for every env; the plan may add per-env sampling from a pool for diversity.
`TrainConfig.joker_loadout` threads it through the existing `engine="jax"` factory branch. Acquisition
stays out — the loadout is constant for the run.

## 5. Parity strategy (two-tier)

Parity is proven at two levels; each catches a class of bug the other cannot. The concrete inputs,
assertions, and case counts live in the §8 test matrix — this section is the rationale.

- **Gate A — component parity (primary).** Drive `engine.scoring.score_play` and
  `score_with_jokers` from the **same directly-injected** inputs (loadout, played/held cards,
  levels, context) and assert identical `(hand_type, chips, mult, score)`. This isolates the
  scoring **math** with a minimal harness and high case throughput, so it is the cheapest and
  strongest signal on the hard part — the ordered fold.
- **Gate B — episode parity (integration).** Inject the **same** fixed loadout into both engines at
  `reset` and run rollouts, reusing the Phase-1 boundary-resync harness. This proves the kernel is
  *wired in* correctly — held-card derivation, context assembly, obs, reward, auto-reset — which
  Gate A's direct-injection harness deliberately bypasses.

Gate A finds math errors; Gate B finds wiring errors. The Python engine is the oracle for both, and
golden hand-computed values (§8) guard against a harness that mis-builds the oracle's inputs.

## 6. Joker dispatch design (Approach A details)

- **Why A over a data-driven table (B).** The parity gate *is* the product; A mirrors the oracle
  1:1 so review is a line-by-line check, and bit-exact ordering is structural, not interpreted.
  B (descriptor "VM") is more compact and scales to 100+ jokers but risks subtle interpreter
  parity bugs and needs escape hatches for the irregular jokers (Photograph, Flower Pot,
  retriggers, rule flags). We can refactor A→B in Phase 4 if the switch grows past ~50 branches.
- **Branch signature.** Every branch is `(ctx) -> (dchips:int32, dmult:float32, xmult:float32)`
  where `ctx` bundles the current card (rank/suit/is_face), the scoring-set summaries (suit set,
  counts), held summaries, and the scalar game context. `on_score` branches read the card;
  `independent` branches read only summaries/scalars; `on_held` branches read the held card.
- **Families** (the branches cluster, so the implementer builds them in batches): flat additive;
  suit `on_score` ±mult/±chips; face `on_score`; rank `on_score`; retrigger; hand-contains
  independent (+mult / +chips / ×mult); context-linear independent; scoring-suit-set ×mult;
  held-card; rule flags. See §7.

## 7. In-scope joker set (~45, by family) with type-ids

Verified against `engine/jokers/library.py` + `base.py`. `(id)` = `JokerType` value.

- **Flat additive (independent):** Joker(1, +4 mult).
- **Suit on_score +mult:** Greedy(2, ♦+3), Lusty(3, ♥+3), Wrathful(4, ♠+3), Gluttonous(5, ♣+3),
  Onyx Agate(119, ♣+7).
- **Suit on_score +chips:** Arrowhead(118, ♠+50).
- **Face on_score:** Scary Face(33, +30 chips), Smiley Face(104, +5 mult),
  Photograph(78, ×2 on **first** scoring face, re-applies per retrigger).
- **Rank on_score:** Fibonacci(31, A/2/3/5/8 +8 mult), Even Steven(39, even +4 mult),
  Odd Todd(40, odd incl. Ace +31 chips), Scholar(41, Ace +20 chips +4 mult),
  Walkie Talkie(101, 10/4 +10 chips +4 mult).
- **Retrigger:** Hack(36, retrigger 2/3/4/5), Sock & Buskin(109, retrigger faces).
- **Hand-contains independent +mult:** Jolly(6, pair +8), Zany(7, trips +12), Mad(8, two-pair +10),
  Crazy(9, straight +12), Droll(10, flush +10).
- **Hand-contains independent +chips:** Sly(11, pair +50), Wily(12, trips +100),
  Clever(13, two-pair +80), Devious(14, straight +100), Crafty(15, flush +80).
- **Hand-contains independent ×mult:** The Duo(131, pair ×2), The Trio(132, trips ×3),
  The Family(133, quads ×4), The Order(134, straight ×3), The Tribe(135, flush ×2).
- **Context-linear independent:** Half(16, ≤3 played cards +20 mult), Banner(22, +30 chips ×
  discards_left), Mystic Summit(23, +15 mult if discards_left==0), Abstract Joker(34, +3 mult ×
  n_jokers), Joker Stencil(17, ×(empty_joker_slots+1)), Bull(93, +2 chips × max(0,money)),
  Blue Joker(53, +2 chips × deck_count), Supernova(43, +mult = hand_plays_run+1),
  Card Sharp(62, ×3 if hand_plays_round≥1).
- **Scoring-suit-set independent ×mult:** Seeing Double(128, ×2 if a Club + any other suit among
  scoring cards), Flower Pot(122, ×3 if all four suits among scoring cards).
- **Held-card:** Baron(72, on_held ×1.5 per held King), Blackboard(48, independent ×3 if every held
  card is ♠/♣ — vacuously true if none held).
- **Rule flags:** Splash(52, all played cards score), Pareidolia(37, all cards are faces).

The plan may stage these in batches (e.g. start with the independent-only families to validate the
fold skeleton, then on_score, then retrigger, then held/rule families), each batch gated by Gate A
before the next.

## 8. Test matrix

All tests run CPU-only (`JAX_PLATFORMS=cpu`). "Slow" = `@pytest.mark.slow`, opt-in via
`BALATRO_RUN_SLOW=1`; the default CI subset is sized to run in a few seconds.

**A. Component parity — `tests/engine_jax/test_joker_scoring_parity.py` (primary).**
- *Harness:* one helper turns a sampled case into matched inputs — plain `Card`s + `JokerState`s
  for `score_play`, equivalent arrays for `score_with_jokers` — asserts identical
  `(hand_type, chips, mult, score)`, and dumps the case (loadout ids, hand, held, levels, context)
  on mismatch.
- *Randomized sampler:* loadout = 0–5 ids drawn **with replacement** from the in-scope set
  (duplicates exercise stacking + multi-retrigger); played = 1–5 distinct cards from a 52-card
  deck; held = 0–7 of the remainder; `levels[ht] ∈ {1,2,3}`; `money`, `discards_left`,
  `deck_count`, and per-hand-type play counts sampled over realistic ranges. CI subset ≈200 cases;
  slow variant ≥1000.
- *Coverage assertion:* the corpus records which in-scope ids actually **fired** (via the dense
  map) and asserts **every** in-scope id appears in ≥1 case — so a large case count can't silently
  skip a rare joker. Each family also gets ≥1 hand-written targeted case (Photograph first-face +
  retrigger re-application; Splash widening; Pareidolia faces; Baron/Blackboard held; Seeing
  Double/Flower Pot scoring-suit sets; Supernova/Card Sharp pre-increment counts).
- *Golden values (oracle-free):* a small table of **hand-computed** expected scores (e.g. pair of
  Aces + Joker `+4` + The Duo `×2`; a flush with Droll `+10` and The Tribe `×2`; a King retriggered
  by Sock & Buskin with Photograph `×2`) asserted directly against `score_with_jokers`. Catches a
  harness that mis-builds the oracle's inputs — a blind spot pure parity cannot see.
- *Fold-order adversarial:* the same joker **multiset in several slot orders**, each checked against
  the oracle — pins the non-commutative `+`/`×` interleaving (the core risk).
- *Negative control:* a deliberately order-swapped / off-by-one expectation is asserted to **fail**
  the comparison, proving the gate has teeth (mirrors the Phase-1 slot-order negative control).
- *Retrigger bound:* assert the oracle's per-card retrigger count never exceeds the static unroll
  bound (`1 + MAX_JOKERS`) anywhere in the corpus.
- *Out-of-scope defensiveness:* a loadout slot holding a deferred/unknown id resolves to the no-op
  branch (asserted == empty-slot result).
- *Batching:* `score_with_jokers` is `jit`- and `vmap`-able, and batched results equal the
  per-element results — guards the on-device path the env actually uses.

**B. Episode parity — extend `tests/engine_jax/test_core_parity_gate.py` (or a sibling).**
- Inject the **same** fixed loadout into both engines at `reset`; drive random-legal rollouts;
  assert within-blind parity on state scalars + ordered hand slots + the core **and** joker obs
  keys + the shaped reward, reusing the Phase-1 boundary-resync harness.
- *Loadout set:* one loadout per family + ≥1 high-interaction mix (e.g. Splash + a suit joker + a
  `×mult` + Sock & Buskin) + the empty loadout. CI subset = a few loadouts × ~50 rollouts; slow
  variant = full set × ≥200 rollouts.

**C. Regression (Phase-1 invariance).**
- *Empty-loadout bit-identity:* `score_with_jokers(empty) == score_core` elementwise over the
  Phase-1 scoring corpus, **and** the existing Phase-1 episode gate (now routed through
  `step → score_with_jokers` with an empty loadout) passes **unchanged**.
- *Planet levels:* a leveled loadout (`levels` 2–3) + jokers scores at component parity — proves the
  kernel honors `levels` exactly as `score_core` did (the only "consumable" surface in scope).

**D. Learning smoke — extend `tests/agent/test_jax_engine_smoke.py`.** PPO trains end-to-end on
`JaxVectorEnv` with a non-empty loadout; finite losses; the joker obs keys are non-zero in rollouts.

**E. Throughput — `scripts/bench_jax_engine.py`.** Re-run with a loadout; confirm env-steps/s stays
within a documented band of Phase 1 (the unrolled fold adds bounded work — record the dip). GPU
number deferred to a CUDA box, as in Phase 1.

## 9. Success criteria

1. **Component parity:** `score_with_jokers` == `score_play` (plain cards, in-scope loadouts) over
   ≥1000 randomized cases — identical hand_type/chips/mult/score, zero mismatches — with **every**
   in-scope joker exercised at least once (coverage assertion, §8.A).
2. **Episode parity:** fixed-loadout rollouts agree with the oracle on every within-blind
   transition (state + slots + obs + reward), reusing the Phase-1 boundary harness.
3. **No Phase-1 regression:** empty-loadout path is bit-identical to Phase 1 (all existing gates
   green).
4. **Learning:** PPO trains on the jokers env with finite losses.
5. **Throughput:** stays within a documented band of the Phase-1 numbers (no order-of-magnitude
   regression).

## 10. Risks & mitigations

- **Fold-order parity** — the core risk. *Mitigation:* Approach A reproduces the oracle's immediate
  in-slot application exactly; Gate A includes mixed +/× loadouts that are order-sensitive.
- **float32 vs float64** — §4.3; exact for the dyadic-only in-scope set under 2²⁴; Gate A asserts
  exact equality.
- **Retrigger unroll bound** — retriggers multiply the inner fold. Each joker's `retrigger` returns
  ≤1, so the exact static bound is `1 + MAX_JOKERS` passes (§4.2); §8.A asserts the oracle never
  exceeds it across the corpus.
- **Pre-increment play counts (Supernova/Card Sharp)** — must read the count *before* `step`
  increments it, for the *detected* hand type. *Mitigation:* compute the context value from
  `CoreState.hand_plays_run/round[ht]` inside the kernel/step before the increment; Gate A covers it.
- **Two engines drifting** — unchanged policy: the Python engine is the oracle; the parity gates
  force the JAX engine to follow.

## 11. Build order (for the plan)

1. `CoreState.jokers` + `zeros_state` + empty-loadout-reduces-to-`score_core` test.
2. Dense-id map + fold skeleton + independent-only families → Gate A (batch 1).
3. `on_score` families (suit/face/rank) + Photograph + retrigger → Gate A (batch 2).
4. Held (Baron/Blackboard) + scoring-suit-set + rule flags (Splash/Pareidolia) → Gate A (batch 3).
5. `reset`/`step` integration + held/context wiring → empty-loadout episode parity unchanged.
6. `obs` joker keys + Gate B (fixed-loadout episode parity).
7. `JaxVectorEnv`/`TrainConfig` knob + PPO smoke.
8. Benchmark re-run + docs (`RUNPOD_M2.md` §8) + memory update.

Same TDD + two-stage-review (spec-compliance, then code-quality) subagent flow as Phase 1, CPU-only
(`JAX_PLATFORMS=cpu`); GPU benchmark deferred to a CUDA box as in Phase 1.

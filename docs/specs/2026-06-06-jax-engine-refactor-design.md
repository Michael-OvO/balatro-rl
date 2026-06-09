# E7: JAX-Native GPU-Vectorized Balatro Engine — Design

**Status:** **Phase 0 + Phase 1 SHIPPED** (merged via PR #32, parity-gated); Phases 2–4 pending. Original status: design (approved direction).
**Date:** 2026-06-06 (Phase 1 landed 2026-06-09)
**Supersedes the efficiency assumptions in:** `docs/RUNPOD_M2.md` §8, memory `efficiency-wall-python-engine`

> **Phase-1 outcome (2026-06-09):** the JAX *core* engine is built and **bit-for-bit parity-gated** against
> the Python oracle (1000-rollout gate green) and a PPO learning smoke passes — success criteria **#1
> (parity) ✅** and **#3 (learning) ✅**. Criterion **#2 (≥10k envs, GPU ≥80 %, ≫100×) is still pending GPU
> validation** — measured only on CPU so far (engine alone ~191k env-steps/s; full PPO is net-bound on CPU).
> The engine is **core-only** (no jokers/shop/consumables/bosses); the Python engine stays the full game +
> oracle + default trainer. Next: Phase 2 (jokers + consumables).

## 1. Problem

Both training tracks underutilize the GPU because they funnel through the **pure-Python,
object-based Balatro engine** (`balatro_rl/engine/`, ~5K LOC):

- **JAX PPO** — the net is tiny (4.7M params, ms backward); the cost is `SyncVectorEnv` stepping
  `BalatroEnv` in a **serial Python loop** (~72 s/update). The GPU does nothing; a single run can't
  use more than a few cores.
- **Agentic LLM (verl + vLLM)** — each turn emits a tiny action, then the GPU **waits on CPU
  env-stepping + turn coordination** → ~0–20 % util (bursty).

Measured 2026-06-06 on an H100: a right-sized 8B GiGPO run still projected ~6–7 h; PPO ~3 h/run; the
two contend for CPU when co-run. The H100 is over-provisioned for both. **No flag fixes this — the
environment has to live on the GPU.**

## 2. Goal & non-goals

**Goal:** a single **GPU-native, vectorized (JAX) Balatro engine** that both trainers consume, so:
- **JAX PPO** runs 10k–100k games in parallel on-device (env *and* net) → GPU pinned ~100 %,
  env-steps/sec orders of magnitude over today.
- **Agentic LLM** rollout stops stalling on Python env-steps; with large concurrent batches + async
  rollout it becomes generation-bound (GPU work), not coordination-bound.

**Non-goals:**
- Deleting the Python engine. It stays as the **parity oracle**, replay/eval renderer, and the home
  of features the JAX engine hasn't absorbed yet.
- Making the LLM agentic track reach 100 % GPU. An 8B emitting a ~tens-of-tokens JSON action per turn
  is inherently lighter than dense training; "fast" there = *no longer env-stalled + big batches*.
- Full game coverage on day one (see phasing). Phase 1 deliberately excludes jokers/shop.

## 3. Architecture

```
balatro_rl/engine_jax/          NEW — pure-JAX engine
  state.py     EngineState = pytree of fixed-shape device arrays (+ masks) — no Python objects
  step.py      step(state, action, key) -> (state, reward, done, info); branchless (lax.cond/where,
               jnp.where, segment_sum); jit-able; vmap over a leading batch dim = N parallel games
  scoring.py   vectorized poker-hand detection + chips/mult (Phase 1: no jokers)
  obs.py       EngineState -> obs array (reuse the existing encode contract / NUM_ACTIONS space)
  config.py    MAX_HAND/MAX_SELECT/MAX_JOKERS/MAX_SHOP/... (reuse existing constants)

tests/engine_jax/test_parity.py  THE GATE — drive JAX + Python engines from the SAME initial deck
  order + SAME action sequence; assert identical obs/reward/done/terminal across the rollout, on
  the covered feature subset. Nothing expands coverage without green parity.

consumers (thin adapters over the batched JAX env):
  • JAX PPO  (balatro_rl/agent/train.py): replace SyncVectorEnv with the JAX env; run the rollout
              on-device with lax.scan + vmap (no Python per-step). Net (flax ActorCritic) unchanged.
  • Agentic  (balatro_rl/llm/verl_env.py BalatroVecEnv): back it with the batched JAX engine + a
              vectorized text renderer; large concurrent-episode batch + verl-agent async rollout.
```

### 3.1 State representation
Fixed-shape pytree (padded + masked — the standard JAX-RL move; `MAX_*` constants already exist):
- `deck`: card ids `[52]` + draw pointer/mask; `hand`: `[MAX_HAND]` ids + valid-mask.
- scalars: `ante, blind_idx, chips_required, chips_scored, hands_left, discards_left, money, round`.
- `rng`: a JAX PRNGKey carried in-state (shuffles/draws are functional).
- Phase ≥2 adds: `jokers[MAX_JOKERS]` (id + per-joker state), `consumables[MAX_CONSUM]`,
  `shop[MAX_SHOP]` (id + price). Phase 1 omits these fields (or carries them inert/masked).

### 3.2 Step (branchless)
`step` takes the flat action id (existing `NUM_ACTIONS` space, `legal_mask` reused), decodes verb+arg
with `jnp.where`, and applies play/discard/advance via `lax.cond`/`lax.switch` over a **fixed** set of
branches — no Python control flow, so `jit`+`vmap` compile once and run N games on-device.

### 3.3 RNG / determinism (parity-critical)
Parity requires the two engines to face the *same* game. Approach: generate the initial **deck order
on the host once per seed** (deterministic), feed it to both engines, then drive both with the same
action sequence and compare deterministic transitions. (Avoids matching two different PRNG streams.)
In-engine randomness after reset (e.g. future pack contents) is deferred to the phases that add it,
each with its own parity strategy.

**Cross-blind parity strategy (implemented in Phase 1).** Within-blind transitions are directly
comparable (same host-seeded deck + same action sequence). At each blind **clear**, the Python engine
enters the SHOP (RNG- and economy-divergent) and reshuffles, while the JAX core advances straight to the
next blind. The parity harness bridges this: it walks Python through shop→PLAYING **buying nothing**,
asserts the new-blind **scalars** match (ante / blind_index / required / hands_left / discards_left), then
**re-syncs** the JAX deck/hand from Python's. RNG and economy diverge by design (out of JAX-core scope);
parity is scoped to within-blind core transitions **plus** the advance scalars at each boundary. See
`tests/engine_jax/test_core_parity_gate.py` (`_python_leave_shop` / `_assert_advance_scalars` /
`_resync_jax_from_python`). The 1000-rollout gate exercised ~12,147 within-blind transitions + ~1,914
blind boundaries with zero mismatches.

## 4. Phasing (each phase: trainable + parity-green before the next)

| Phase | Scope | Deliverable / gate |
|---|---|---|
| **0 — scaffold** | State pytree, parity harness infra, trivial deal/observe step | Parity green on reset+observe; CI runs the harness |
| **1 — core loop** | deal → select → play/discard → **vectorized scoring** (poker hands, base chips/mult) → blind/ante progression → win/lose. **No jokers/shop/consumables.** | Parity vs Python (jokers disabled); **first GPU-saturating PPO run** (§5) |
| **2 — jokers + consumables** | ~20–30 highest-impact jokers (additive/mult/retrigger as masked array contributions) + basic consumables | Parity on the covered jokers; PPO retrains |
| **3 — shop/economy/packs/vouchers** | the acquisition game | Parity; PPO retrains |
| **4 — bosses + long-tail jokers** | full coverage | Parity; deploy-game eval matches Python |

The **agentic LLM track plugs in from Phase 1 onward** (it consumes the same batched engine via the
text boundary); it does not need its own engine.

## 5. Success criteria (Phase 1 — the go/no-go for the whole investment)
1. **Parity:** JAX engine == Python engine across ≥1000 (seed × action-sequence) rollouts on the
   core subset — identical obs, reward, done, terminal info.
2. **Throughput:** PPO trains at ≥10k parallel envs with **GPU util ≥ ~80 %**, and env-steps/sec
   ≥ ~100× the current Python pace (today ~72 s/update at 64 envs).
3. **Learning:** mean blinds-cleared rises on the core game (the agent still *learns*, not just runs).

## 6. Risks & mitigations
- **Vectorized scoring + branchless joker math** (the core challenge): poker detection via sort/count
  on the played `[MAX_SELECT]` cards; joker effects (Phase ≥2) as masked `jnp.where`/`segment_sum`
  contributions. *Mitigation:* Phase 1 has **no jokers**; phasing isolates the hard part.
- **Variable-length state** (hand/jokers/shop): fixed `MAX_*` padded arrays + masks.
- **Novelty:** no public GPU-vectorized Balatro exists → Phase 0/1 is also feasibility de-risking.
  *Mitigation:* the parity gate makes "is it correct?" objective at every step.
- **Two engines drifting:** the Python engine is the oracle; any rule change lands there first, then
  the parity test forces the JAX engine to follow.
- **Compute:** develop + parity-test on **CPU** (fast; a laptop/M4 is fine — JAX CPU backend); scale
  PPO runs on a GPU (JAX CUDA). Parity tests run in CI without a GPU.

## 7. Testing
- **Parity harness** (the gate) — per §3, expanded each phase to its new coverage.
- **Unit tests** for scoring (hand classification, chips/mult) against known Balatro values.
- **Throughput micro-benchmark** — env-steps/sec + GPU util at a target batch size (Phase 1 metric).
- **Learning smoke** — a short PPO run on the JAX core engine shows blinds-cleared rising.

## 8. Open questions (resolve during planning)
- Exact Phase-1 scoring coverage (which poker hands / edge cases like flushes-of-5 vs played-subset).
- Whether the agentic text renderer batches on host or stays per-env (likely per-env, cheap).
- JAX version / CUDA build matrix for the GPU box (the pod ships CPU jaxlib today).

# Card-Aware Agent Rebuild — Design Spec

**Goal:** Replace the flat 465-way softmax (which can't link a play-ID to the cards it
selects) with a **card-aware network** that learns hand value **from the cards** (no
engine hand-type features), plus a **curriculum** that makes clearing learnable.

**Why:** The agent is stuck at blind 1 — clears 0 blinds in 2M steps, plays High Card /
Pair (~150 of 300 chips), flat learning trajectory. A reward+exploration sweep moved
nothing → the bottleneck is structural: a flat softmax produces each play-logit from one
pooled state vector with no link to *which* cards a play selects, so it can never learn
that 5 hearts beat a pair. (See memory `card-aware-agent-rebuild`.)

**Decision:** learn-eval-from-cards (purist "from scratch", user-chosen). Companion
curriculum is essential (the net gives capacity; the curriculum gives the experience of
clearing). Critic-verified against the repo.

---

## Hard invariants (do not break)

- `net.apply(params, obs, mask) -> (logits[B,465], value_logits[B,255])` **byte-for-byte
  preserved** → `ppo.py`, `value_head.py`, `spec.py`, `obs.py` **UNCHANGED**.
- Everything jit-stable: FIXED shapes, static gather, masked pool, **no per-step
  recompiles** (the net jits exactly twice: `act`@B=num_envs, `update`@B=mb_size).
- No engine-computed hand-type / chips / mult features fed to the net. `obs["hand"]` is
  `[8,17]` = rank one-hot(13) + suit one-hot(4) only.
- Curriculum is **host-side** (env is numpy) and defaults to scale 1.0 ⇒ all existing
  tests/behavior unchanged.

## Architecture (new `networks.py`)

Keep `class ActorCritic(action_dim, d_model=128, n_bins=NBINS)`; ADD only defaulted
fields `num_heads=4, n_layers=2, pool="deepsets"`. `JOKER_VOCAB=200`. **Full reference
code in `/tmp/arch_full.txt`** (read it for the exact lines). Flow:

```
obs["hand"][B,8,17] → Dense(d) + pos[8] + seg_card        # per-card tokens [B,8,d]
  L=2 pre-LN masked self-attention (MHA, 4 heads, mask = hand_mask[:,None,None,:]>0) + FFN
context: shared Embed(200) over jokers/shop + Dense(global|levels|deck) → ctx[B,d]
FiLM: H = attended*(1+gamma) + beta,  (gamma,beta)=split(Dense(2d)(ctx))      # [B,8,d]

# CANDIDATE-SCORING play/discard head (the fix):
gathered = H[:, SUBSET_IDX, :]            # static [218,5] gather → [B,218,5,d]
eff = (SUBSET_CNT[None] * hand_mask[:,SUBSET_IDX])[...,None]                   # [B,218,5,1]
phi = Dense(d)(gelu(Dense(d)(gathered)))
pooled = concat([(phi*eff).sum(2), s_sum/clip(n,1), n], -1)    # masked DeepSets sum+mean+count
set_emb = gelu(Dense(d)(pooled))                                              # [B,218,d]
play_logits = Dense(1)(gelu(Dense(d)(concat([set_emb, broadcast(ctx)],-1))))[...,0]  # [B,218]
disc_logits = <SEPARATE final scorer, SHARED phi/set>                                 # [B,218]

# shop head (29): per-slot MLPs over je/se/ctx; reorder via PAIR_I/PAIR_J [20]
logits = concat([play218, disc218, shop29], -1)                              # [B,465]  EXACT actions.py order
logits = jnp.where(mask, logits, finfo.min)        # identical to current net
value_logits = Dense(255)(gelu(Dense(d)(ctx)))     # distributional head kept, on ctx
```

Static module constants (built once from `actions._SUBSETS`/`_PAIRS`): `SUBSET_IDX[218,5]`
int32, `SUBSET_CNT[218,5]` f32 (1=real,0=pad), `PAIR_I[20]`, `PAIR_J[20]`. Plain
module-level jnp arrays (baked as constants, never params/recompiled).

Param count ~0.9–1.1M. Only memory cost: `gathered/phi [B,218,5,d]` ≈ 1.1 GB at default
mb_size=2048, ~4.6 GB at num_minibatches=1 → guardrail.

## Curriculum (trainability) — full text in `/tmp/arch_training.txt`

Host-side `required_score` scaling so clearing is frequent early, closed-loop ramp to 1.0:
- `blinds.required_score(ante, blind_index, scale=1.0)` → `max(1, round(base*mult*scale))`.
- `GameState` gains trailing defaulted `req_scale: float = 1.0`.
- `engine.reset(seed, scale=1.0)`: set `required=required_score(1,0,scale)` AND
  `req_scale=scale` (keyword); `_advance_blind` passes `state.req_scale`.
- `BalatroEnv(reward_name, req_scale=1.0)` + `set_req_scale`; `SyncVectorEnv` forwards +
  `set_req_scale` loops EVERY sub-env.
- `train.py`: TrainConfig `req_scale_schedule=1.0, curr_floor=0.2, ramp_clear_rate=0.7,
  ramp_step=0.05, ramp_window=20`; capture `venv.step` infos → rolling clear-rate →
  closed-loop `venv.set_req_scale(s)` before each rollout (raise toward 1.0 only when
  window clear-rate > threshold, clamp [floor,1]); `_req_scale_at` helper; decaying
  entropy default (~0.04→0.008); `num_minibatches>=4` guardrail; log `train/req_scale`,
  `train/clear_rate`. Reward: keep **`shaped`** (NOT `hand_quality` — leaks hand-type).
- Because `req_scale` only changes the integer `state.required` (→ obs floats), the
  jitted path never recompiles.

## Files to change

| file | change |
|---|---|
| `agent/networks.py` | **full rewrite** (CardSAB encoder + candidate head + shop head) |
| `engine/blinds.py` | `required_score(..., scale=1.0)` |
| `engine/state.py` | add trailing `req_scale: float = 1.0` |
| `engine/engine.py` | `reset(seed, scale=1.0)`; `_advance_blind` uses `state.req_scale` |
| `envs/balatro_env.py` | `__init__(req_scale=1.0)` + `set_req_scale` |
| `envs/vec_env.py` | forward `req_scale` + `set_req_scale` (loop sub-envs) |
| `agent/train.py` | curriculum loop + clear-rate accumulator + decaying entropy + guardrail |
| `ppo.py`, `value_head.py`, `spec.py`, `obs.py` | **NONE** |

## Test plan

1. **Static tensors:** `SUBSET_IDX.shape==(218,5)`, dtype int32, max==7; `SUBSET_CNT.sum(1)`
   == each `_SUBSETS` row length; `PAIR_I/J` == `_PAIRS` cols.
2. **Contract/shape:** `net.init(dummy_obs(1), ones((1,465),bool))`; `net.apply` returns
   `logits[B,465]`,`value[B,255]` for B∈{1,64,2048}; dtypes/keys match current net.
3. **Logit-order round-trip:** for a seeded state+legal_mask, every illegal index ==
   finfo.min and non-min set == legal_mask True set; assert shop offsets
   BUY→436-437, SELL→438-442, REROLL→443, REORDER→444-463, LEAVE→464.
4. **Card-awareness:** perturb one heart's obs feature → ONLY subsets containing that slot
   change their play/disc logit.
5. **Masked-pool safety:** with `hand_mask==0` slots, pad/absent rows contribute zero;
   flipping a padded slot's raw obs changes no surviving legal logit.
6. **jit-stability:** jit `act`@num_envs and `update`@mb_size → exactly 2 compiles, no
   recompile across calls.
7. **Curriculum plumbing:** `required_score(1,0,1.0)==300`, `==0.2)==60`;
   `engine.reset(seed,0.2).required==60` and `.req_scale==0.2`; `_advance_blind` reflects
   scale; `SyncVectorEnv.set_req_scale` updates every sub-env incl. post-auto-reset.
8. **Curriculum jit-safety:** obs at scale 0.2 vs 1.0 same shapes/dtypes/keys.
9. **Closed-loop ramp:** `_req_scale_at`/accumulator raises scale when clear-rate >
   threshold (clamped), holds below.
10. **Smoke:** `train(num_updates=3, num_envs=16, num_steps=64, req_scale_schedule=0.2)`
    runs, losses finite, clear-rate>0 within a few updates at low scale.

## Risks (mitigated)

- Memory of the gather → `num_minibatches>=4` guardrail (assert/warn) + optional bf16/d_cand.
- **Logit-assembly order is silent & load-bearing** → round-trip test (#3) pins it.
- Degenerate curriculum floor (single High Card clears) → `curr_floor` high enough + ramp.
- `engine.reset` MUST set `req_scale` via keyword (else default 1.0 silently disables it).
- Eval at scale 1.0 while training low reads ~0 clears early (metric trap; eval at both).

## Build order (subagent-driven, TDD)

1. Curriculum engine plumbing (`blinds`, `state`, `engine`) + tests 7.
2. Curriculum env plumbing (`balatro_env`, `vec_env`) + tests 7.
3. `networks.py` rewrite + tests 1–6 (the crux).
4. `train.py` curriculum loop + decaying entropy + guardrail + tests 9, 10.
5. Validation: smoke + a short curriculum run that manufactures clears.

Est. effort ~1–1.5 days.

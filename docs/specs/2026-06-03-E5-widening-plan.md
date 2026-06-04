# E5 — Coordinated obs/action/network widening + GPU retrain

> Execution: inline TDD (the obs/action/network core is one load-bearing index-for-index
> contract — splitting it across agents risks silent index drift). Each task = test → impl →
> full suite green.

**Goal:** Let the agent finally *see and use* every acquisition system built in E1–E4 (buy
consumables, open booster packs, buy vouchers, target Tarots), then retrain on the full game
on GPU.

**Architecture:** One obs revision + one action-space revision + matching network heads, all
landing together. The agent's blindfold (the `legal_actions` filters added in E1–E4) comes
off. A small env-level boss-rate curriculum folds in the plateau lesson. No warm-start — obs
shape changes, so the GPU retrain is from scratch (which is the point).

**Tech stack:** JAX/Flax (`networks.py`), the pure engine, numpy obs/action codecs.

---

## The new action layout (the contract everything else matches)

```
PLAY         [0, 218)        _SUBSETS[id]
DISCARD      [218, 436)      _SUBSETS[id-218]
SHOP_BASE=436:
  BUY        436..437  (2)   BUY offer slot — NOW kind-agnostic (joker OR consumable)
  SELL       438..442  (5)
  REROLL     443       (1)
  REORDER    444..463  (20)
  LEAVE      464       (1)
USE          465..466  (2)   no-target consumables (Planets / no-target Tarots) — existing
--- new E5 blocks (appended; phase-masked) ---
USE_TARGET   467..684  (218) apply the PENDING targeting-Tarot to _SUBSETS[s]
OPEN         685..686  (2)   buy+open pack_offers slot 0..1
PICK         687..691  (5)   pick pack_open item 0..4
SKIP_PACK    692       (1)
BUY_VOUCHER  693       (1)
NUM_ACTIONS = 694
```

Network logit assembly order (must equal the above):
`[play(218), disc(218), shop(29), use(2), use_target(218), open(2), pick(5), skip(1), voucher(1)]`
= 694.

## The targeting two-step (pending_consumable)

Card-targeting Tarots need hand indices the flat space can't pre-enumerate cheaply. Solution:
a tiny sub-state, reusing the existing card-aware pool for the "which cards" decision.

- New `GameState.pending_consumable: int = -1`.
- New `Verb.USE_TARGET = 12`.
- In **PLAYING** phase, `legal_actions` now emits a bare `(USE, ci)` for a targeting Tarot.
  Stepping it does **not** apply — it sets `pending_consumable = ci` (only entered when the
  hand is non-empty, so a valid target always exists; matches Balatro: Tarots target hand cards).
- While `pending_consumable >= 0`, `legal_actions` returns **only** `(USE_TARGET, subset)` for
  subsets of size `1..max_targets(tarot)` with all indices `< len(hand)`. Picking one applies
  `_use_consumable(state, pending, subset)` and clears pending. (Direct tuple path
  `step((USE, (ci, *targets)))` from E2 is untouched — 739 existing tests keep passing.)
- Defensive: `pending_consumable` resets to -1 across `_advance_blind` and any cash-out.

## The new obs schema (encode() in obs.py)

Re-layout `global` (g) to add the OPEN_PACK phase bit + pack/voucher/pending scalars:

```
g[0..11]   unchanged scalars
g[12..16]  phase one-hot (5: PLAYING/WON/LOST/SHOP/OPEN_PACK)   N_PHASES 4 -> 5
g[17]      len(consumables)
g[18]      consumable_slots
g[19]      boss-active
g[20]      pending_active (0/1)
g[21]      pending_max_targets
g[22]      len(pack_offers)
g[23]      pack_picks (remaining this OPEN_PACK)
GLOBAL_FEAT = 24
```

New/changed entity streams (kind is derivable from which id is nonzero, so no separate kind
one-hots — keeps it lean):

| key | shape | meaning |
|---|---|---|
| `shop_types` | (2,) | joker-vocab id of a JOKER offer, else 0 (unchanged) |
| `shop_consum` | (2,) | **NEW** consum-vocab id of a consumable offer, else 0 |
| `shop_cost`,`shop_mask` | (2,) | unchanged |
| `pack_kind` | (2,) | **NEW** PackKind (1..3; 0=pad) of each pack offer |
| `pack_size` | (2,) | **NEW** PackSize (1..3) |
| `pack_cost`,`pack_offer_mask` | (2,) | **NEW** |
| `pack_item_joker` | (5,) | **NEW** joker id of a revealed JOKER item, else 0 |
| `pack_item_consum` | (5,) | **NEW** consum-vocab id of a revealed CONSUMABLE item, else 0 |
| `pack_open_mask` | (5,) | **NEW** |
| `voucher_offer` | (1,) | **NEW** VoucherType id offered (0=none) |
| `voucher_offer_mask` | (1,) | **NEW** |
| `vouchers_owned` | (24,) | **NEW** multi-hot owned set |
| `pending_consum` | (1,) | **NEW** consum-vocab id of the pending targeting Tarot (0=none) |

## The new network heads (networks.py)

- `buy[2]`: unchanged structure; `se` now `= joker_embed(shop_types) + consum_embed(shop_consum)
  + cost + seg` so consumable offers are scored (one of the two embeds is the id-0 pad).
- `use_target[218]`: **one** `Dense(1)` over the existing `cand[B,218,2d]` pool — card-aware
  target selection for free. Conditioned on pending via g (pending_active/max/`pending_consum`).
- `open[2]`: new `pe = packkind_embed(pack_kind) + packsize_embed(pack_size) + cost + ctx`.
- `pick[5]`: new `pie = joker_embed(pack_item_joker) + consum_embed(pack_item_consum) + ctx`.
- `skip[1]`: `Dense(1)(ctx)`.
- `voucher[1]`: `voucher_embed(voucher_offer) + ctx`.
- `vouchers_owned`, pending scalars ride in the `g` global block (already concatenated to ctx).
- Still exactly two jits (act@num_envs, update@mb): everything fixed-shape.

## legal_actions unblock (engine.py)

- SHOP: emit `(BUY, i)` for affordable consumable offers (slot available); `(OPEN, i)` for
  affordable packs; `(BUY_VOUCHER, 0)` if offered, affordable, prereq met.
- PLAYING: emit `(USE, ci)` for targeting Tarots (hand non-empty, not pending) → enters pending.
- Pending: only `(USE_TARGET, subset)` ids (sized to the tarot).
- OPEN_PACK PICK/SKIP already emitted (E3).

## Boss-rate curriculum (env-level — no engine change)

The plateau came from bosses being full-strength while the score bar was still ramping.
`BalatroEnv` gains `boss_rate`: at each **episode** reset it rolls (deterministic in the seed)
whether that episode enables bosses. The training curriculum ramps `boss_rate` in lockstep with
`req_scale` (so bosses fade in as the score target rises); eval/deploy uses `boss_rate=1.0`.

## GPU / RunPod prep

- `requirements-cuda.txt` (jax[cuda12], matching pins) + `docs/RUNPOD.md` setup note.
- A `retrain_gpu` config: larger `d_model`/`num_envs` to fill the GPU, `NUM_UPDATES` for a full
  run, curriculum on, boss-rate ramp on, records replays on the real game.
- A `--smoke` short run proves it trains end-to-end locally before the user launches RunPod.

## Out of scope (unchanged from the audit): endless, deck/stake select, finisher-boss
## effects, Spectral cards, blind-skip/Tags, Fool/Wheel Tarots, the 5 deferred vouchers.

## Tasks
- [ ] T1 actions.py — new layout + decode/encode/legal_mask (+ tests)
- [ ] T2 engine — pending_consumable, USE_TARGET, legal_actions unblock (+ tests)
- [ ] T3 obs.py — new streams, g re-layout, N_PHASES=5, OBS_SHAPES (+ tests)
- [ ] T4 networks.py — new heads, assembly order, jit-count (+ shape tests)
- [ ] T5 env — boss_rate per-episode (+ test)
- [ ] T6 train/retrain — boss-rate ramp, GPU config, eval boss_rate=1.0
- [ ] T7 GPU/RunPod — requirements-cuda.txt, docs/RUNPOD.md
- [ ] T8 replay_data — record pack/voucher/pending fields (future runs fully detailed)
- [ ] T9 smoke train + full suite green + adversarial review → hand off retrain
```


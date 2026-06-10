# Experiment Retro — Full-Game PPO on CPU: curriculum stall → PPO non-convergence diagnosis

**Date:** 2026-06-10  **Status:** STOPPED (diagnosed; not converged — root cause identified, fix planned)
**Context:** E7 era. Benchmarking + training the *real* full-acquisition-game agent (`retrain.py` config,
Python `SyncVectorEnv` engine) on a Mac16,5 (16-core, 48 GB, **CPU**; no GPU). Live metrics via Trackio
(project `balatro-retrain-e5`); trajectory replays via the Gradio viewer.

## TL;DR
The agent **never learned to win** (eval win-rate 0% throughout, eval mean-ante pinned at 1.0 across ~2100
effective updates). The surface symptom was a *curriculum stall* at `req_scale=0.80`; the real cause is a
**broken learning signal: the critic is card-blind**, so advantages collapse to noise, the policy gradient
dies, and the entropy bonus pins the policy near-random. Curriculum/entropy tuning moved the stall point
but could not fix it — because the blocker is architectural, not compute or curriculum.

## What we ran
- **Config:** full game (jokers/shop/consumables/packs/vouchers/bosses), `d_model=256`, `num_envs=64`,
  `num_steps=64`, `gamma=0.999`, `gae_lambda=0.95`, `lr=3e-4`, `clip=0.2`, `vf_coef=0.5`, reward `shaped`,
  curriculum `curr_floor=0.2`, `ramp_clear_rate=0.70`, `ramp_step=0.05`, boss curriculum on
  (`boss_rate = cur_scale`), `eval_interval=50`, `early_stop_patience=8`.
- **Throughput (measured):** ~37.7 s/update single-process (~9 of 16 cores; serial Python env loop is the
  wall), ~21 h for a 2000-update run; 2 parallel runs ≈ 1.5× aggregate (box saturates at ~2). A GPU does
  ~nothing for this path (env-bound, tiny net) — the E7 thesis.
- **Runs:** `full-cpu-1round` (v1, ~1405 updates), `full-cpu-1round-v2fix` (resume, 738 updates).
- **Artifacts:** `~/balatro_runs/full-cpu-1round{,-v2fix}/` (checkpoints `ckpt.msgpack`, replays
  `replay_u*.episode.json`). View: `trackio show --project balatro-retrain-e5`;
  `uv run python -m balatro_rl.viz.viewer` then upload a replay.

## Timeline
1. **v1:** curriculum climbed 0.20 → 0.80 at max speed (~21 updates/step) while clears were easy, then
   **stalled at 0.80 for ~1000 updates** — on-policy `clear_rate` fell to ~0.50, below the 0.70 gate.
2. **Adjustments (v2):** lowered the gate 0.70→0.60, **decoupled bosses** (`boss_curriculum_power=2.0` →
   `boss_rate = scale²`, so 0.80→0.64), resumed at `curr_floor=0.70`, warm-started from the v1 ckpt.
3. **A resume bug:** the launcher reset the entropy schedule to update 0 → `ent_coef` back to 0.04 (4×
   exploration) → policy went random → `clear_rate` fell *below* the gate. Then we over-corrected toward
   `0.01`, then settled on `0.02`.
4. **v2fix:** climbed 0.70 → 0.75 → 0.80, then **re-stalled at 0.80** (`clear_rate` ~0.52), still 0% win.

## The data signature (the smoking gun)
Loss decomposition was flat across all windows — classic "not learning":
| signal | value | meaning |
|---|---|---|
| `loss/value` | **~1.62, FLAT** | critic learns the *mean* return (uniform CE would be ln255=5.54) but **can't resolve states** |
| `loss/policy` | **~−0.056, ~0** | clipped surrogate gets no consistent gradient |
| `loss/entropy` | **~3.3, not decaying** | ≈ ln(27) = uniform over the ~27 legal actions → policy stays random |
| eval mean_ante / win_rate | **1.0 / 0% ever** | no transfer to full difficulty |
| `max_round_score` | climbs then plateaus ~110k | improves on the *tail*, not consistency |

## Root-cause analysis (verified in code)
1. **PRIMARY — card-blind critic.** `networks.py:172-173`: `value_logits = Dense(255)(gelu(Dense(d)(ctx)))`
   reads **only the pooled global `ctx`**, while the **actor is fully card-aware** (per-subset DeepSets
   scorers over the `[B,218,5,d]` candidate pool, `networks.py:122-139`). So within a blind, `V` is
   ~constant regardless of which play the policy picks → GAE advantages (`ppo.py:35,40`) collapse to a
   per-blind constant → after per-minibatch normalization (`ppo.py:47`) they're pure noise → `pg≈0`.
2. **CO-PRIMARY — entropy is the only live actor gradient.** With `pg≈0`, `total = pg + vf·vl − ent_coef·ent`
   (`ppo.py:51`) is dominated by `−ent_coef·ent`, which *maximizes* entropy → policy pinned uniform. Self-
   reinforcing: a random policy never generates the trajectory diversity the critic would need to learn
   finer values.
3. **TERTIARY — reward saturates / is sparse.** Φ = `min(round_score/required,1) + 0.05·symlog(money) +
   0.5·ante` saturates at ratio=1 the instant a blind clears (`rewards.py:20-21`); the curriculum makes
   clears early, so the γ=0.999 potential-difference telescopes to ~0/step, leaving only sparse +1/+10 —
   high-variance, useless for fine credit assignment even with a perfect critic.

The value head's symlog support (`[−30,30]`, 255 bins, `value_head.py:11`) is correctly sized for the
10²–10¹² return range — **not** the problem. The stuck value loss is a *symptom* of card-blindness, not a
broken or mis-ranged head.

## Errors & misreads (honest log)
- **"value loss 1.6 ⇒ broken/uniform critic"** — WRONG. Uniform CE = ln(255) = 5.54; 1.62 ≈ mass over ~5
  bins. The multi-lens (4-agent) diagnosis caught this; a single read had it backwards. *Lesson: do the CE
  arithmetic; adversarial cross-checking beats one pass.*
- **Tuning entropy UP (0.02)** — backfired. At scale 0.80 the agent clears *better* with LOW entropy (v1:
  ~0.66 at ent≈0.01 vs v2fix: ~0.52 at ent 0.02). Raising it made play sloppier. The deeper truth: entropy
  was only a symptom of the dead policy gradient.
- **Resume reset the entropy schedule** to u=0 (launcher bug) → cranked exploration to 0.04 on a competent
  warm-started agent.
- **Early-stop blind spot** — `evals_no_improve` only increments when `cur_scale ≥ 1.0` (`train.py:306`).
  A curriculum that stalls *below* 1.0 (ours) leaves the plateau guard dormant forever. Not a bug per se,
  but a real failure mode: the anti-overfit guard can't catch a sub-1.0 stall.
- **Boss/score coupling** — `boss_rate = cur_scale` stacked two difficulty axes into one wall at 0.80.

## Lessons learned
- **Actor and critic must have symmetric state access.** A card-aware actor + card-blind critic ⇒ dead
  advantages ⇒ nothing learns. This is the single biggest lesson.
- **Diagnose the loss *decomposition*, not aggregate metrics.** value↑↔policy↔entropy together told the
  whole story; reward/clear-rate alone read as "stuck curriculum."
- **On CPU the engine wasn't the only limiter** — the *learning signal* was. More compute/GPU would NOT have
  fixed a card-blind critic. Fix the signal before scaling.
- **Curriculum gates on on-policy stochastic clear-rate**, which is highly sensitive to the entropy
  coefficient. Entropy schedule is a first-class hyperparameter here.
- **Always log `advantage std` and `value_decode` stats** — adv-std≈0 is direct confirmation of an
  uninformative critic vs a gradient/NaN bug.

## Directions to iterate (prioritized)
1. **[PRIMARY] Make the critic card-aware** — value head over `concat([ctx, masked_pool(H, hand_mask)])`
   (`networks.py:172-173`). Fixed-shape, jits identically. *Confirm:* value loss falls (no longer flat),
   `|loss/policy|` grows past ~0.1, entropy starts decaying within ~50–100 updates.
2. **Front-load entropy decay** — `0.02 → 0.001` by ~40% of the run (durable only WITH #1).
3. **Lower `gamma` 0.999 → 0.99** — the ~1000-step discount is mismatched to ~5-step-per-blind structure;
   shrinks advantage variance.
4. **Narrow value bins** `(−30,30) → (−10,15)` — ~92% of bins unused; ~2.4× resolution for free.
5. **Return normalization** — apply running mean/std consistently to stored values AND GAE targets (the
   `adv` is normalized but targets are raw, `ppo.py:47/49`).
6. **De-saturate reward** — add a `ShapedProgress` variant: `symlog(round_score/required)` instead of
   `min(ratio,1)`, so within-blind shaping stays dense (sweep, don't mutate `shaped`).
7. **Add diagnostics first** — log `value_decode` mean/std + per-minibatch advantage std (`train.py` ~271).
8. **Only then scale** — GPU + JAX engine (10k+ envs) once the signal is informative; not before.

Keep the `boss_curriculum_power` knob (added to `train.py`) for gentler boss fade-in in future curriculum
runs. Early-stop should also count a stall (e.g., scale frozen for N evals) so a sub-1.0 plateau can halt.

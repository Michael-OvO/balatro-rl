# JAX Maskable-PPO Agent — Implementation Reference

Condensed from a verified research pass (2024–2026 patterns, cross-checked against the repo's `obs.py`/`actions.py`). Full code patterns live in the plans; this captures the durable decisions, exact formulas, and the load-bearing pitfalls.

## Stack decisions (verified)
- **Flax Linen** (`flax.linen`) + **optax** — every reference RL impl (PureJaxRL, Stoix, rejax, JaxMARL) uses it; params-as-pytree threaded through `apply` composes with `jit`/`vmap`/`lax.scan`. (NNX = go-forward but no RL baselines; Haiku = maintenance-only; Equinox = more plumbing.)
- **No distrax needed** — hand-roll the masked categorical (Gumbel-max sample + `log_softmax` log-prob + masked entropy); avoids a dependency and the `pi.entropy()` NaN trap.
- `nn.SelfAttention` is **deprecated** → `nn.MultiHeadDotProductAttention`. Masked-logit sentinel is **`jnp.finfo(dtype).min`**, never literal `-inf`.

## Policy head — flat masked categorical over 465 (start simple)
465 actions is the small-action regime; flat masked categorical is provably correct (Huang & Ontañón, arXiv 2006.14171). Entity structure goes in the **encoder**, not the head.
```
logits = jnp.where(action_mask, logits, jnp.finfo(logits.dtype).min)   # Jumanji pattern
```
Candidate-scoring (per-entity) head is a later interface-preserving refactor (needs per-entity sub-masks in obs.py).

## Encoder — Deep-Sets first, attention second
- **Plan 5 (validate cheap):** per-group projection + `nn.Embed` for `joker_types`/`shop_types` (vocab ≥ max joker id + 1, e.g. 64) + shared learned segment tags → masked-mean (and/or sum/max) pool (Deep Sets, Zaheer 2017) → concat globals → MLP. Permutation-invariant, no attention NaN pitfalls.
- **Plan 6 (upgrade):** swap pool for 2× Set-Attention Blocks (`MultiHeadDotProductAttention`, key-padding mask via `nn.make_attention_mask(valid, valid)`, re-zero pads each block, masked-mean pool). Only `Encoder.__call__` changes.

## Value head — symlog two-hot (DreamerV3), exact formulas
```
symlog(x) = sign(x)*log1p(|x|);  symexp(x) = sign(x)*expm1(|x|)   # exact inverse
NBINS, LO, HI = 255, -30.0, 30.0     # widened for Balatro 1e12+ (symlog(1e12)≈27.6); odd -> a bin at 0
BINS = linspace(LO, HI, NBINS)
two_hot(scalar): t=symlog(scalar); place on two nearest bins, nearer bin gets MORE weight (cross-assign); mean(encoding)==t
value_loss = -(stop_grad(two_hot(target)) * log_softmax(value_logits)).sum(-1)   # categorical CE
value_decode = symexp((softmax(value_logits) * BINS).sum(-1))
```
(HL-Gauss soft target, arXiv 2403.03950, σ≈0.75·bin_width, is an optional more-robust variant; two-hot is the σ→0 case.)

## Maskable-PPO update (PureJaxRL skeleton, jit/scan)
- **GAE** via reverse `lax.scan`, γ≈0.999 (long horizons), λ=0.95. `targets = advantages + values`.
- **Loss:** clipped surrogate with **normalized advantages**; value **CE** (not MSE); **masked entropy** = `-jnp.where(mask, p*logp, 0.).sum(-1)`.
- **Minibatch epochs** via nested `lax.scan` (one permutation key → all leaves aligned).
- `optax.chain(clip_by_global_norm(0.5), adam(3e-4, eps=1e-5))`.
- **CRITICAL:** store `masks` in the rollout buffer and **re-apply the identical mask when recomputing log-probs/entropy in the update** (not only at sampling) — the #1 maskable-PPO correctness bug (naive masking → KL explosion).

## Python(numpy)-env ↔ JAX-agent loop
- Env steps in a **plain Python for-loop** (numpy/host); jit only (a) per-step policy `act` and (b) the whole PPO `update`. Do NOT pull the numpy env into `lax.scan` (host_callback ~10× slower).
- One **batched** policy call per step over all N envs; pass raw numpy obs/mask straight into the jitted policy (JAX transfers internally); `np.asarray(action)` is the sync point.
- Rollout buffer = preallocated numpy `(T, N, …)` incl. `masks[T,N,465]`; one device transfer per update.
- **Recompile avoidance (the silent killer):** fixed shapes (NUM_ENVS, NUM_STEPS as constants), stable dtypes (float32 obs / bool mask / int32 ids), stable pytree keys, hyperparams as closed-over constants, no Python control flow on traced values. `JAX_LOG_COMPILES=1` → compiles only at startup.

## Starting hyperparameters
d_model 128 · MLP 128×1 · embed vocab 64 · value bins 255 (±30 symlog) · Adam lr 3e-4 eps 1e-5 · grad-clip 0.5 · γ 0.999 · GAE λ 0.95 · clip 0.2 · VF 0.5 · ENT 0.01 · epochs 4 · minibatches 4 · NUM_ENVS 64 · NUM_STEPS 128 (batch 8192).

## Open / uncertain
- Two-hot bin window ±30 vs DreamerV3 valnorm — verify actual max return before fixing; pick one.
- HL-Gauss vs two-hot at Balatro variance — tune σ if used.
- Attention may be unnecessary for 15 tokens — measure Deep-Sets first.
- Candidate-scoring head deferred (needs per-entity sub-masks); only if `MAX_HAND` grows or flat learning stalls.

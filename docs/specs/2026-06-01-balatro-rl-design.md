# Balatro RL — Design Spec

- **Date:** 2026-06-01
- **Status:** Design approved (brainstorming complete) → ready for implementation planning
- **Goal:** Train an agent, from scratch with reinforcement learning, to play Balatro and score as high as possible.

---

## 1. Goal & scope

Balatro RL trains an RL agent to play [Balatro](https://balatrowiki.org/w/Balatro) for maximum score. The *precise* objective — win Ante 8 / maximize raw score / maximize depth — is deliberately treated as a **research variable**: the reward is a pluggable component, and comparing objective routes is a first-class experiment rather than a fixed choice.

**First milestone:** a trainable end-to-end pipeline on a reduced ("Tier-0") ruleset that learns to engine-build and clear early antes, paired with the observability to *see* what the agent is doing.

**Non-goals (v0 / YAGNI):** full 150-joker fidelity on day one; multiplayer/online; pixel-perfect UI; world-record seed hunting; a compiled engine before throughput actually demands it.

### Why Balatro is a hard RL problem
- **POMDP:** you see an 8-card hand; the rest of the deck is hidden.
- **Exponential reward:** scores span ~10² to 10¹²⁺, which breaks naive value regression.
- **Variable combinatorial action space:** play/discard any ≤5-of-8 subset; buy/sell/reroll/use over a randomized shop.
- **Heavy stochasticity:** card draws, shop rolls, pack contents, probabilistic card/joker effects.
- **Long horizon + engine building:** ~300 decisions per run; value compounds across a whole run (synergistic jokers).

---

## 2. Key decisions & rationale

| Decision | Choice | Why |
|---|---|---|
| Engine language | **Python**, behind a Rust-portable seam | Max clarity + instant iteration in the language we're fluent in; the ~150-joker zoo is heterogeneous dynamic dispatch that resists JAX's fixed-shape constraints. |
| Why not Rust now | Deferred behind a clean seam | Speed sensitivity is a **threshold, not a gradient**. The fatal gap is real-game (~0.3 steps/s) vs *any* code simulator (10⁴–10⁶× faster). Python-vs-Rust is only ~5–20× — diminishing returns that bind only at 10¹⁰+ steps × many runs (a late-stage concern). |
| Why not pure-JAX env | JAX for the **agent only** | A branchy Balatro env in JAX lands in Pgx's ~10⁵–10⁶ SPS band *and* is painful to write (`lax.switch` over 150 effects + masking everywhere). Python keeps the clarity JAX sacrifices and buys speed with CPU cores. |
| Agent / learning | **JAX + PPO** (maskable) | User preference; clean and fast for the policy/update; end-to-end `jit` on GPU. |
| Eval / fidelity | **balatrobot** (drives the real game) | Parity oracle + ground-truth agent evaluation. |
| Compute | **RunPod** (cloud, on-demand) | Containerized, checkpointable, resumable; the pod's vCPU:GPU ratio is a per-run throughput dial for the Python env. |

### The Python → Rust escape hatch (preserved by design)
Difficulty of a future port is a step-function in one variable: is game logic *contained in the engine* or *smeared across the obs encoder / training loop*? We keep it contained:
1. **Isolate the engine** behind `reset / legal_actions / step` — no RL concepts leak in, no game logic leaks out.
2. **`State` is plain data** (int arrays + scalars + index-based references), not a rich object graph — crosses PyO3 trivially, ports directly to Rust structs, and is what the obs encoder/JAX want anyway. (Small ergonomic cost: jokers reference each other by slot index, not object pointer.)
3. **Explicit, portable RNG** (our own seedable PCG/splitmix64 threaded through `step`) — not Python's `random`/`numpy.random` — making `step` a pure function and enabling bit-exact cross-impl parity.
4. **A golden parity corpus** validated against the real game — the regression test *and* the future Rust-port acceptance test.

A port then becomes a contained, golden-test-driven, **incremental** (PyO3 function-by-function, profile-guided, reversible) swap of one module — not a rewrite. These disciplines are also just *good engine design*, so the optionality is nearly free.

### Throughput expectations
Optimized multiprocessed Python ≈ **10⁵–3×10⁵ steps/s**, comfortably above the ~10⁵ viability line. Balatro tolerates a slow env because it is a **low-frequency decision game** (~300 choices/run, so 10⁵ steps/s ≈ ~10⁶ complete games/hour; a 10⁹-step run ≈ 1–3 h on a pod). Optimization ladder, held in reserve, climbed only when a run demands it:
1. Clean engine + `__slots__` + int-encoded cards + precomputed hand/score tables → ~20–60k SPS/core.
2. **Multiprocess vectorization across vCPUs + shared memory + async double-buffer → ~10⁵–5×10⁵ SPS** (main lever, zero clarity cost).
3. Numba-JIT only the pure-numeric kernels (poker detection, chip sums); jokers stay plain Python.
4. PyPy for the env process (IPC to the JAX agent).
5. Surgically port the scoring hot-path to a C/Cython/Rust extension — last resort, not a rewrite.

---

## 3. Architecture

Layered modules with one hard boundary — the engine "island":

```
balatro_rl/
  engine/              ◀── PURE game logic. The ISLAND. Python now, Rust-portable later.
    state.py           #   State = plain data (int arrays + scalars + index refs), POD
    cards.py           #   card ↔ int encoding (rank, suit, enhancement, edition, seal)
    rng.py             #   explicit portable PRNG (splitmix64/PCG), threaded through step
    hands.py           #   poker-hand detection + base chip/mult tables   ← hot path
    scoring.py         #   pipeline: played → held → jokers → global       ← hot path
    jokers/            #   one clear fn per joker + exhaustive registry/dispatch
    consumables/       #   tarot / planet / spectral
    shop.py  blinds.py #   shop gen + economy; antes/blinds/boss effects
    engine.py          #   BalatroEngine: reset · legal_actions · step(state, action, rng)
  envs/                ◀── RL-facing. Consumes engine State only — never its internals.
    obs.py             #   encode_obs(State) → entity tokens + global vector
    actions.py         #   action space, id↔move, legal-action mask
    rewards/           #   PLUGGABLE objectives (win_ante8 · pure_score · max_depth · shaped)
    balatro_env.py     #   single-game env wrapping the engine
    vec_env.py         #   multiprocess workers + shared mem + async double-buffer
  agent/               ◀── JAX
    encoder.py heads.py networks.py   #   entity transformer · candidate-scoring π · two-hot V
    ppo.py             #   jit PPO: GAE, clip, action-masking, symlog targets
  train/   config.py curriculum.py train.py     #   rollout → learn → log/ckpt
  eval/                #   balatrobot_bridge · parity · baselines · evaluate
  infra/runpod/        #   Dockerfile, launch, checkpoint sync
  tests/golden/        #   recorded transition corpus = the Rust-port acceptance test
```

### Data flow — one async training step
```
 W worker procs (CPU)                          GPU (JAX)
 ┌──────────────────┐   obs+mask (B,·)    ┌────────────────────────┐
 │ engine.step ×B/W │ ──── shared mem ───▶│ encoder → π (masked)    │
 │ encode_obs/legal │                     │         → V (two-hot)   │
 │ reward           │ ◀─── actions (B) ───│ sample action           │
 └──────────────────┘                     └────────────────────────┘
        └ double-buffer: step batch t+1 while GPU runs t ┘
                 │
        rollout buffer ─▶ PPO update (jit: GAE · clip · symlog · mask) ─▶ params
                              │ periodic ▼
                  checkpoint · log · eval(sim + balatrobot) · parity test
```

### Decisions → structural commitments
| Decision | Structural commitment |
|---|---|
| Python engine, Rust-portable | `engine/` island: POD `State`, explicit `rng`, narrow `reset/legal_actions/step` seam |
| Objective is a research variable | `envs/rewards/` — swappable objective modules behind one interface, chosen by config |
| Start Tier-0, expand later | `config.engine.tier` gates which jokers/consumables/mechanics load |
| JAX agent + PPO | `agent/` is the only JAX code; everything else is plain Python |
| Python throughput lever | `vec_env.py`: multiprocess + shared memory + double-buffering |
| balatrobot for eval | `eval/` drives the real game; `tests/golden/` + `parity.py` keep the sim honest |
| RunPod | `infra/runpod/` containerized, checkpointable, resumable |

### Error handling (baked in)
- Illegal actions can't reach the engine (policy is masked; env also asserts legality → hard error in tests, penalty in train).
- Worker crash → supervisor reseeds & restarts that env.
- Deep-Endless score overflow to `inf` → symlog + clean episode termination guard.
- balatrobot disconnect → eval bridge reconnects with backoff.

---

## 4. Observation / Action / Reward contract

Engine-language-agnostic: consumes `State`, emits candidates + mask. The Python→Rust port never touches this.

### 4.1 Observation — entity tokens + a histogram for the hidden deck
Heterogeneous, variable-length state → a **set of typed tokens** (padded to caps, masked):

| Token group | v0 cap | Per-token features (illustrative) |
|---|---|---|
| **Hand cards** | ≤8 | rank(13), suit(4)+wild, enhancement(9), edition(5), seal(5), face-down, selected, in-best-hand |
| **Jokers** | ≤5 (cap 8) | type-id embed(~150), edition(5), stickers(3), sell$, scaling counters (symlog), **slot-position embed** |
| **Consumables** | ≤2 (cap 4) | type-id embed(~52), negative-flag |
| **Shop items** | ≤ slots | kind(7), type-id embed, cost(symlog), edition(5), affordable-flag |
| **Deck histogram** | 1 group | counts by rank(13)/suit(4)/enhancement(9)/edition/seal + cards-remaining — *the hidden deck* |
| **Hand-type levels** | 12 | level(symlog), times-played |
| **Global token** | 1 | money, ante, blind, hands/discards left, required & score(symlog), **chips/required ratio**, reroll$, slots, deck-id(15), stake(8), boss-effect(~28), vouchers(32), phase |

Non-obvious choices: jokers carry a **position embedding** (order changes scoring); the unseen deck is summarized as a **histogram** (turns "hidden cards" into "known distribution" — POMDP without an RNN); all exponential scalars are **symlog-encoded**.

### 4.2 Action — one masked candidate space across phases
The env emits a **legal-action mask** every step; the policy scores only legal candidates.

| Phase | Candidate actions |
|---|---|
| **Blind select** | play-blind · skip-blind (→ take tag) |
| **Playing** | play-subset (≤56 of choose-≤5-of-8) · discard-subset (≤256 bitmasks) |
| **Shop** | buy-item×slots · sell-joker×slots · sell/use-consumable×slots · **reorder-jokers** · reroll · open-pack×slots · leave |
| **Pack open** | pick-card×offered (· skip) |
| **Targeting (sub-phase)** | toggle-target-i · confirm (entered when using a card-targeting consumable) |

**v0:** a flat masked discrete space (~400–500 ids), exploiting the 8-card cap so the combinatorial play/discard choice is *enumerated and masked* (the proven `cassiusfive/balatro-gym` trick) — no autoregressive machinery needed. Shop/joker/consumable/pack actions are **candidate-scored** from their entity embeddings.

**Parameterized actions** (tarots, joker reorder, pack pick) decompose into the game's natural "click sequence" — a short chain of simple masked choices:
- Card-targeting tarots (Magician/Empress/Death/etc.) → enter the **targeting sub-phase**: select consumable → toggle ≤N target cards → confirm. Death's ordered pair = pick source → pick dest.
- Joker reorder → pick joker → pick destination slot (order changes score, so this is a real lever).
- Booster pack → **pack-open sub-phase**: pick ≤k of n.

**Upgrade path** (deferred, no env-contract change): swap the play head to an autoregressive/SAINT set-head if larger hands (hand-size mods) or richer targeting are needed.

### 4.3 Reward — pluggable objective × shared exponential-safe substrate
**Layer 1 — the swappable objective** (`envs/rewards/`, chosen by config — the thing we A/B):
- `win_ante8` — sparse milestone reward building to clearing Ante 8.
- `pure_score` — Δ symlog(cumulative score); chases the exponential tail.
- `max_depth` — proportional to antes/blinds survived.

**Layer 2 — a shared substrate every objective opts into** (so comparisons are apples-to-apples):
- **Potential-based shaping** `F = γΦ(s′) − Φ(s)`, Φ = weighted sum of *bounded* dense signals — chiefly **log(chips_scored / required)**, plus money, hand-levels, joker count. Potential-based ⇒ **policy-invariant** (speeds learning without changing the optimum or inviting reward-hacking).
- **Exponential handling, cross-cutting:** symlog-transform every score-derived quantity (obs *and* reward); the **value head is distributional** (two-hot over symexp bins, §5); reward-normalize + clip as a safety net. (This concern appears in three places — obs, reward, value head — and missing any one diverges training.)

---

## 5. Agent network (JAX)

```
 tokens: cards · jokers · consumables · shop · pack · deck-hist · global · [CLS]
        │  per-type projection  +  type-segment & joker-position embeddings
        ▼
 ┌──────────────────────────────────────────────┐
 │  Transformer trunk  (≈3 layers, d≈256, 4 heads)│  self-attention over all tokens (padding-masked)
 └──────────────────────────────────────────────┘
     │ per-token embeddings                  │ [CLS] / pooled state embedding
     ▼                                       ▼
  candidate scorers  (thin MLPs)          value head
   play/discard · buy · sell · use ·      symlog two-hot over K bins → V
   toggle-target · reorder · singletons
     │ concat → legal mask (−∞) → softmax → π
     ▼
   sample masked action
```

- **Encoder — entity transformer:** per-type input projections (joker/consumable type-ids → embedding tables of ~150/~52 + numeric features), a learned **type-segment** embedding, and a **position embedding on jokers only** (hand/shop/consumables are sets). Full self-attention lets jokers attend to hand cards (synergy) and shop items attend to current jokers. Start **small** (~2–4M params, d≈256, 3 layers) — the forward pass is the shared throughput ceiling (~1–5M SPS), so compactness keeps us near it.
- **Policy head — candidate scoring + masking:** the trunk reasons; thin per-kind MLP scorers read the relevant token embedding + [CLS] summary → logits; legal mask sets illegal entries to −∞; one softmax → π. Masked logits get zero gradient (Huang & Ontañón 2020) — a valid policy gradient — and the same mask machinery covers every phase/sub-phase.
- **Value head — symlog two-hot:** predicts a categorical distribution over K≈255 bins in symlog space; value = `symexp(Σ pₖ·binₖ)`; target return is symlog-encoded and two-hot across its two nearest bins; loss is cross-entropy (DreamerV3-style). This is the single thing keeping the critic from collapsing on 10-orders-of-magnitude returns.
- **PPO update (jit'd):** rewards are symlog-shaped in the env → log-bounded returns → clean GAE(λ) + advantage normalization for the policy loss; value loss uses two-hot CE; entropy bonus over the **masked** distribution (explore only among legal actions); Adam + grad-clip; a few epochs over minibatches via `scan`; env steps outside the jit (Python `vec_env`). **γ ≈ 0.999** deliberately — long horizon, delayed engine-building payoff.
- **Built-in accelerators (hooks, not v0-critical):** BC warm-start from a scripted expert (greedy-best-hand + simple shop heuristics, JackPotts-style); ante/stake curriculum.

*Hyperparameters above are starting points, to be tuned. Illustrative PPO config: B≈2048–4096 envs, rollout T≈32–128, 2–4 epochs, λ≈0.95, clip≈0.2, entropy≈0.01, LR≈2.5e-4 (annealed), grad-clip≈0.5, K≈255 value bins.*

---

## 6. Engine tiers + eval / parity / baselines

### 6.1 Tiers — each a playable, parity-tested slice
| Tier | Scope added | Unlocks strategically | Ships when |
|---|---|---|---|
| **0 · MVP** | full **scoring pipeline** (hand detection, base tables, order-of-ops); base 52 deck (no mods); antes 1–8 + score-multiplier bosses; 12 hand types + **planets**; **~30 curated jokers** (spanning +mult/×mult/chips/economy/retrigger/scaling); basic shop (buy/sell/reroll + economy); play/discard + joker reorder; pluggable reward | end-to-end trainable agent that engine-builds and clears early antes | scoring parity ≥99.9% on corpus |
| **1 · Content** | remaining common/uncommon jokers (→~100+); **booster packs** (pick-k sub-phase); **vouchers**; tags; **debuff boss blinds** | pack decisions, voucher economy, boss adaptation | parity holds with packs + bosses |
| **2 · Card mods** | **enhancements / editions / seals**; **card-targeting tarots** (targeting sub-phase); spectrals | the combinatorial "engine" explosion | parity on enhanced-card scoring |
| **3 · Full** | rare + legendary jokers; 15 decks; 8 stakes + stickers; **endless mode**; finisher bosses | full fidelity + deep-score chase | full-game parity |

The scoring pipeline is in **Tier 0 from day one** — it's the hard-to-replicate core; later tiers add *content*, not new scoring math. Reduced scope is a **validated subset**, not a fidelity gamble.

### 6.2 balatrobot — two distinct jobs
**(1) Parity testing** (keep the sim honest): replay identical `(seed, action-sequence)` through the Python engine and the real game; assert matching scores/states. Build a **golden corpus** from random/heuristic/edge-case play; run in CI and before every tier ships; target ≥99.9% (JackPotts hit 99.89%).
- **Honest caveat on randomness:** deterministic scoring must match exactly; probabilistic effects (Lucky 1-in-5, Glass 1-in-4) can't be asserted per-trial without replicating Balatro's exact RNG stream → test them **statistically** (distribution over many trials) plus a **force-outcome test mode** for deterministic unit tests.

**(2) Agent evaluation** (measure real skill): periodically run the *trained* agent on the real game (low frequency — slow) for ground-truth win-rate/ante/score and to catch sim-to-real divergence (the guard against over-optimizing a sim inaccuracy).

### 6.3 Baselines & metrics
| Baseline | Purpose |
|---|---|
| Masked-random | absolute floor |
| Scripted heuristic (best-hand + simple shop, JackPotts-style) | strong reference (~7% real-game win) |
| BC-of-heuristic | sanity-check the net can imitate before it innovates |

**Tracked:** win-rate (clear Ante 8), mean/max ante, score (log: mean/median/max), reward, episode length; training health — SPS, value loss, entropy, explained variance, KL. **Headline experiment:** hold agent + engine fixed, swap `rewards/{win_ante8, pure_score, max_depth}`, compare curves + real-game outcomes.

---

## 7. Observability, debug & replay (+ infra)

For an RL project, observability *is* the debugger.

### 7.1 Minimal infra (deliberately thin)
`Dockerfile` + config-driven entrypoint (`python -m balatro_rl.train --config configs/tier0_shaped.yaml`); **checkpoint = engine-tier + params + optimizer + RNG state** to a mounted volume / HF / S3 so a preempted pod resumes exactly. Everything seeded (env RNG + JAX PRNG keys). Iterate by editing a config and relaunching.

### 7.2 Dashboard
**Trackio** (HF, lightweight, local-first, wandb-API-compatible) — clean, no account, runs on the pod and syncs to a shareable Space; W&B optional later. Panels:
- **Curves:** reward, win-rate, mean/max ante, score (log), episode length, entropy, value-loss, KL, explained-variance, **env SPS + GPU util** (instantly see if env-bound).
- **Distributions:** ante-reached histogram, score distribution, **action-type mix**, **joker popularity**, hand-type usage.
- **Objective overlay:** the same panels with `win_ante8` / `pure_score` / `max_depth` superimposed.
- **Surfaced replays:** links to the viewer for best / worst / median episodes each eval.

### 7.3 RL debugging kit
- **Single-process `--debug` mode:** 1 env, no multiprocessing → `pdb` straight into engine/agent.
- **Assertion-rich engine:** invariants fail fast in debug, compiled-out in train.
- **Parity-as-debugger:** replay the exact `(seed, actions)` against the real game → "sim bug" vs "agent quirk."
- **Overfit-one-seed smoke test:** can the agent memorize a single fixed game? If not, the learning loop is broken.
- **Reward decomposition logged per term** (progress / money / hand-levels / milestone) → see what it's paid for.
- **Mask audits:** π never on illegal actions; a legal action always exists.

### 7.4 Replay viewer — the centerpiece
Because the engine is a **pure deterministic function of `(seed, actions)`**, a replay is just that tiny tuple — re-run it to reconstruct every state, and overlay the agent's logged `(π, value, reward)` to see what it was thinking.

```
┌──────────────────────────────────────────────────────────────┐
│ Run #1287 · seed 4f2a · reward=shaped · Ante 5/8 · ◀ ▷ step142 │ ← scrubber
├──────────────────────────── BOARD ───────────────────────────┤
│ Jokers: [Blueprint][Baron][Mime][+Mult][Bull]   $34  ✋4 ♻2   │
│ Hand:   7♥ K♠ K♦ 3♣ A♥ 9♠ Q♦ 2♣        (selected: K♠ K♦)      │
│ Blind:  Big · need 11,200 · scored 8,400                      │
├─────────────────────── SCORING (last play) ──────────────────┤
│ Pair of Kings  base 10×2                                      │
│   +chips K10+K10 → 30   ·   Baron ×1.5×1.5 +Mult4 → mult 13.5 │
│   = 30 × 13.5 = 405                                           │
├────────────────────── AGENT (this decision) ─────────────────┤
│ V≈9,800   reward +0.21 (prog +0.18, money +0.03)             │
│ π:  play[K♠K♦] 0.62 ████████  play[K♠K♦A♥] 0.21 ███          │
│     discard[3♣2♣] 0.10 █   …                                  │
└──────────────────────────────────────────────────────────────┘
```

Shows **board** + **score breakdown** (verify scoring visually) + **agent's mind** (value, reward decomposition, action distribution) so a bad move is obvious. **v0: a Gradio app** (fastest "simple but clean," Python-native, embeddable) with a timeline slider; custom HTML/JS is the "prettier later" option; a tiny ASCII/TUI version is an ultra-fast in-loop debug tool. Doubles as the **parity debugger** (sim vs real score side-by-side) and the **objective-comparison lens**.

### 7.5 The one sequencing principle (we iterate fast otherwise)
Build the spine **engine → env → random/heuristic agent → replay viewer → dashboard *first***, before the PPO learning loop — so the moment training starts, you can already *see* what's happening instead of staring at a loss curve. Then add learning, then climb engine tiers behind parity gates. Detailed milestones intentionally omitted.

---

## 8. Open questions / risks to revisit
- **Exact RNG-stream replication** for per-trial parity on probabilistic effects — start with statistical + force-outcome testing; only replicate Balatro's RNG if needed.
- **Hand sizes > 8** (hand-size mods) vs the flat-enumerated play action — Tier-0 caps at 8; upgrade to an autoregressive/SAINT play head when needed.
- **Python agent↔env boundary in JAX** (env steps outside jit) — confirm async double-buffering hides transfer at target batch sizes; measure SPS early to know if/when to climb the optimization ladder.
- **Boss-blind effects** that alter scoring/draw — phased across tiers; ensure the obs always exposes the active boss effect and the mask reflects its restrictions.
- **Curriculum & BC warm-start** — hooks present; tune only if cold-start is slow.

---

## 9. Reference points (open-source prior art)
- `coder/balatrobot` — real-game API (JSON-RPC), our eval/parity bridge.
- `evanofslack/balatro-rs` — fast Rust engine + Gym + action masking (skeleton if we ever port).
- `jarmstrong158/Balatron` — PPO over the real game, 814-dim state, ~6% win (reference encoding).
- `DrLatBC/JackPotts` — heuristic + MCTS, 99.89% scoring fidelity (baseline + scoring oracle).
- `cassiusfive/balatro-gym` — flat 312-action trick + reward shaping + BC warm-start (reference design).
- Immolate / balatrolator / EFHIII — RNG/shop generation and scoring references.

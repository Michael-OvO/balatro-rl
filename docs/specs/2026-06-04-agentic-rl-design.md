# Agentic-RL Track — Design Spec (v0)

- **Date:** 2026-06-04
- **Status:** Design approved (brainstorming complete) → ready for implementation planning
- **Goal:** Add an **agentic-RL** policy track — an LLM that plays Balatro by reasoning over a multi-turn `think → act → observe` loop, fine-tuned with RL from verifiable rewards (RLVR) — *beside* the existing JAX-PPO track, reusing the engine/env/reward/eval/replay spine unchanged.

---

## 1. Goal & scope

Train a 7–14B LLM, via **multi-turn GRPO**, to **win Balatro Ante 8** (Stage 1 of the staged objective). The reward stays pluggable; the LLM agent is "just another policy" to the existing eval/replay infrastructure.

**Staged objective (user intent):** first maximize Ante-8 win rate; *then*, in a follow-on, push score as high as possible in **Endless**. v0 covers **Stage 1 only**.

**In scope (v0):**
- A new `balatro_rl/llm/` package: state→text serialization, legal-action menu + parser, multi-turn rolling-summary context manager, an `LLMAgent`, a gym-like text env, and a thin verl adapter for GRPO.
- A **frozen-model baseline** that runs on the existing eval/replay harness with no trainer dependency (the cheap go/no-go gate).
- A **minimal** `ExperimentConfig` exposing only the knobs needed to launch a run.

**Out of scope / deferred (YAGNI — see §10):**
- **Endless mode** (continue past Ante 8, escalating blinds, bignum scoring). Separate spec; the engine does not support it today (`engine.py` wins at the Ante-8 boss; `economy.py`/`scoring.py` flag Endless as not-yet-implemented).
- The full composable-reward / credit-assignment / multi-algorithm *framework*. v0 reuses existing reward presets + verl's GRPO defaults; the richer config surfaces are an **extension path** (§5), built only when a sweep demands them.
- Any change to `engine/`. The engine island is untouched.

**Success criteria (v0):**
- **M1 gate:** a frozen LLM agent produces valid full-game trajectories and a measured Ante-8 win rate + ante-depth distribution vs the Random/Greedy/PPO baselines, with readable chain-of-thought replays.
- **M2 target:** a GRPO-trained LoRA that beats the frozen baseline on Ante-8 win rate.

---

## 2. Key decisions & rationale

| Decision | Choice | Why |
|---|---|---|
| Policy class | **LLM** (7–14B instruct/reasoning) + **LoRA** | Already knows poker/joker semantics — bypasses the tabula-rasa "blind-1 wall" the JAX track fights via `HandQuality`. 7–14B + LoRA fits 1×80GB. |
| Loop shape | **Full multi-turn** (persistent reasoning thread) | User choice. Richest agentic behavior + best CoT observability. Cost is controlled by the rolling-summary context manager (§3). |
| Framework | **PyTorch** + **verl** (GRPO) + **vLLM** rollouts | The mature agentic-RL stack lives here. Overrides the JAX preference in the original design doc — but only for the agent half; the engine/env stay framework-agnostic Python. |
| Algorithm | **GRPO** (critic-free) | No value network: advantages are group-relative over G rollouts of the same seed. Sidesteps Balatro's exponential-reward value-regression problem (group ranking is ~scale-free); the two-hot value head is unneeded here. |
| Architecture | **Clean text-env boundary + verl adapter** (option C) | Mirrors the repo's proven "engine island + pluggable agent" discipline. The frozen baseline + eval + replay run on the boundary with zero trainer dependency; the trainer is swappable. |
| Compute | **1×80GB (A100/H100)** | Sweet spot for 7–14B LoRA + reasonable group size G + multi-turn context. |
| Objective | **Win Ante 8** (Stage 1); Endless deferred | Stage 1 runs on the ready engine; Endless needs orthogonal engine work. |

### Why this is additive, not a rewrite
The engine exposes `reset / legal_actions / step` over plain-data state, and `descriptions.py` already renders faithful effect text. An LLM agent needs exactly those four things (step, legal moves, readable observation, scalar reward) — all present. The new package consumes them as `RandomAgent`/`GreedyAgent` do. ~70% of the project (engine/env/reward/eval/replay) is reused; ~30% (the JAX agent + PPO loop) is paralleled, not replaced — the JAX track remains as a baseline to beat, which matches the repo's "comparing routes is a first-class experiment" ethos.

---

## 3. Architecture

One new package beside `agent/`, consuming the engine and existing `envs/` unchanged:

```
balatro_rl/llm/
  serialize.py      # GameState -> compact text observation (reuses engine/descriptions.py)
  actions_text.py   # legal-action menu builder + action parser/validator (wraps envs/actions.py)
  context.py        # rolling-summary multi-turn context manager (the loop-B core)
  policy_client.py  # Policy backend: FrozenEndpointPolicy | training-time policy
  agent.py          # LLMAgent.act(state, mask) -> action_id  (trainer-agnostic, multi-turn)
  text_env.py       # BalatroTextEnv: gym-like text wrapper over the existing BalatroEnv
  reward_adapter.py # bridges envs/rewards.py to the trainer (mostly passthrough)         [M2]
  verl_env.py       # thin verl multi-turn-rollout adapter + GRPO config                  [M2]
  train.py          # entrypoint: verl GRPO run (LoRA, vLLM, group size G, KL)            [M2]
  baseline.py       # frozen-model eval entrypoint (reuses runner.py + eval)              [M1]
```

### Component contracts (each isolated, single-purpose, testable)

- **`serialize.py`** — `serialize_state(state) -> str`. Pure function. Renders the `GameState` as compact, readable text: phase, ante/blind, chips vs target, hands/discards left, money, the hand, jokers (name + **effect text from `descriptions.py`** + edition), consumables, shop offers w/ prices, packs, vouchers, active boss effect. Token-budget-aware. *Depends on:* engine state + `descriptions.py`.

- **`actions_text.py`** —
  - `legal_menu(state) -> list[Option]`: turns `legal_actions(state)` into a readable menu; each option carries its flat `action_id` (via existing `encode_action`).
  - `parse_action(model_output, state) -> action_id | Error`: maps the model's choice back to a validated flat id; illegal/unparseable → structured error.
  - **Action format:** discrete actions (shop / pack / voucher / use) → **numbered menu pick** (legal by construction). PLAY / DISCARD / USE_TARGET → **structured card-index call** (e.g. `{"action":"play","cards":[0,3,5]}`) validated against `legal_actions`, avoiding the 218-subset enumeration. This is `legal_mask` reborn as a menu + validator.

- **`context.py`** — the heart of the multi-turn loop. Within a token budget, maintains: (1) a static system prompt (rules primer + objective + action format), (2) a running compact summary (the model's carried plan + engine-derived run stats), (3) a sliding window of the last N raw turns; older turns fold into the summary. Guarantees bounded context across the ~300-turn game. v0 summary strategy: **deterministic** (engine-derived stats + a model-written plan slot); a model-summarizer is a later swap behind the same interface.

- **`policy_client.py`** — `Policy.generate(messages) -> text`. `FrozenEndpointPolicy` (OpenAI-compatible vLLM/API endpoint) powers M1 baseline + eval; the training policy is managed by verl's rollout worker at M2. Same interface both sides.

- **`agent.py`** — `LLMAgent` implementing the **same `act(state, mask) -> action_id`** signature as the baseline agents, plus per-episode conversation state. Builds the prompt (system + `context` + `serialize` + `legal_menu`), calls the `Policy`, parses the action, updates `context`.

- **`text_env.py`** — `BalatroTextEnv`: gym-like `reset(seed)` / `step()` returning text obs + reward + menu, internally wrapping the **existing `BalatroEnv`** (reusing its reward, masking, and "no-legal-move → terminal loss" handling).

**Why the `act(state, mask) -> action_id` signature matters:** a `Trajectory` is `(seed, [action_ids])` and the engine is deterministic, so `runner.py`, the replay viewer, the parity corpus, and the eval harness all work with the LLM agent **unchanged**. The LLM is "just another policy" downstream.

---

## 4. Data flow

### Frozen baseline (M1) — runs on the boundary, no trainer
```
runner.run_episode(BalatroTextEnv, LLMAgent(FrozenEndpointPolicy), seed)
  loop until done:
    prompt = system + context.render() + serialize(state) + legal_menu(state)
    reply  = policy.generate(prompt)        # thought + action
    act_id = parse_action(reply, state)      # validated; retry/penalty if illegal
    obs, r, done = env.step(act_id)
    context.update(reply, obs)
  -> Trajectory(seed, action_ids, ...)  + per-turn CoT
```

### GRPO training step (M2)
```
verl driver (1×80GB)
 ├─ for each seed in the batch, sample G rollouts via vLLM (temp>0)   # same left column as above
 ├─ GRPO advantage = group-normalize(returns of the G rollouts on that seed)
 │   (+ dense per-turn reward -> per-decision advantages; no value net)
 ├─ token-mask to assistant/action tokens only; LoRA update with KL-to-ref
 └─ Trackio: return, win-rate, KL, reward parts, turns/game, tokens/game (cost watch)
```
Same seed across the G rollouts gives the group baseline; rollouts differ only by sampled actions at temperature > 0 (engine RNG is seeded and deterministic).

---

## 5. Tunability — minimal now, seams for later

**v0 (build this):** reward = existing `envs.rewards.make_reward(name)` presets, unchanged; algorithm = verl's default **GRPO**; a thin `ExperimentConfig` (dataclass → YAML) exposing only:
```
ExperimentConfig:
  model:   name, lora{rank, alpha, dropout}, max_ctx
  reward:  name                         # existing preset: "shaped" | "hand_quality" | "win_ante8" | ...
  algo:    group_size G, kl_coef, lr    # verl GRPO defaults otherwise
  rollout: temperature, max_turns, max_tokens_per_turn, seeds_per_batch
  context: budget_tokens, window_n
  curriculum: req_scale, boss_rate      # reuse the existing env curriculum knobs
```
Default run: the `shaped` preset (which already bakes in the +1 clear / +10 win milestone bonuses, per `rewards.py`) + GRPO + group-relative dense credit.

**Extension path (do NOT build until a sweep needs it):** three orthogonal config surfaces, enabled by keeping the env/agent/serializer/reward-*content* code **agnostic to the algorithm**:
1. **`RewardConfig`** — composable reward content: `components: [(name, weight)]` + `transform: symlog|none`. Today's presets become defaults; weights become swept knobs. `rewards.py` is already organized as addable terms (`_shaped_potential`, tier ladder, milestones), so this is a clean extraction.
2. **`CreditConfig`** — how reward becomes signal, independent of content: `granularity: token|turn|episode`, `mode: dense|outcome_only|reward_to_go`, `gamma`, `advantage: group_relative|rloo|gae` (GAE needs a value critic — not used by v0's critic-free GRPO), `normalize`.
3. **`AlgoConfig`** — `name: grpo|ppo|rloo|...` + hyperparams, mapped onto verl (expose, don't reimplement).

**Invariant that keeps this real:** `text_env.py`, `agent.py`, `serialize.py`, and reward *content* never reference the algorithm. Only `verl_env.py` / `train.py` read `AlgoConfig` + `CreditConfig`. If algorithm names appear in env/reward code, the boundary has leaked.

---

## 6. Eval & observability
- **Eval:** `LLMAgent` drops into `runner.run_episode` + the existing eval harness. Report **Ante-8 win rate** and **ante-depth distribution**, head-to-head vs Random/Greedy/PPO.
- **Replay:** unchanged (`Trajectory` = seed + action_ids → engine determinism). **New:** attach per-turn CoT so the replay viewer shows *why*, atop the existing board + score-breakdown view.
- **Trackio:** training curves (return, win rate, KL, per-component reward) **plus cost telemetry** (turns/game, tokens/game) — the dial that catches runaway rollout cost under full multi-turn.

## 7. Error handling
- Illegal / unparseable action → structured error back to the model, bounded retries (≈2), then a safe legal fallback or terminal penalty. Reuses the env's existing "no legal move → terminal loss."
- Context overflow → bounded by `context.py`; hard truncation is the backstop.
- vLLM / rollout failure → that rollout is dropped from its GRPO group (degrade, don't crash the batch).

## 8. Testing
- **Unit:** serializer (golden text for known states); action parser (round-trip `legal_actions ↔ menu ↔ parse` + illegal rejection); `context.py` (bounded length).
- **Integration:** `LLMAgent` plays a full game via a **stub policy** (canned actions) → valid `Trajectory`; then a frozen-model smoke run.
- **Reuse:** the golden parity corpus still passes (engine untouched).

## 9. Build sequence (milestones)
- **M1 — Boundary + frozen baseline (cheap gate):** `serialize`, `actions_text`, `context`, `policy_client(frozen)`, `agent`, `text_env`, `baseline`. **Deliverable:** measured frozen win rate + ante-depth vs baselines, with CoT replays. Go/no-go on training.
- **M2 — GRPO training:** `reward_adapter`, `verl_env`, `train.py`, the minimal `ExperimentConfig`. LoRA GRPO on 1×80GB; default `shaped` preset (clear/win bonuses included), group-relative dense. **Deliverable:** trained LoRA beating the frozen baseline on Ante-8 win rate.
- **M3 — Endless:** separate spec (engine extension: continue past Ante 8, escalating blinds, bignum scoring + the score-maximization objective).

## 10. Non-goals (v0 / YAGNI)
- Endless mode and bignum scoring (Stage 2, separate spec).
- The composable-reward / credit-assignment / multi-algorithm framework (extension path only).
- Full fine-tuning (LoRA only at v0).
- Any `engine/` change.
- Replacing or retiring the JAX-PPO track (it remains a baseline).

## 11. Open questions (resolve at planning)
- Exact base model within 7–14B (instruct vs reasoning; e.g. a Qwen-family model with a thinking mode). Decide at M1 against the frozen baseline.
- verl version + its multi-turn rollout API specifics (pin during M2 planning).
- Context token budget + window N (tune empirically once the serializer's real token cost is measured at M1).

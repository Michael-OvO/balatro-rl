# Balatro RL

Training an agent **from scratch with reinforcement learning** to play [Balatro](https://balatrowiki.org/) and score as high as possible.

> **Status:** implemented & iterating. The full Python engine, maskable-PPO agent, agentic-LLM track, replay viewer, and Trackio dashboards are built, and the agent trains on the full acquisition game. **E7 (Phase 1):** a GPU-vectorizable pure-**JAX** *core* engine is shipped and parity-gated against the Python engine (PR #32) — see the E7 docs below.

## The shape of the project

- **Engines:** the **pure-Python** engine (`balatro_rl/engine/`) implements the full game (jokers / shop / consumables / packs / vouchers / bosses / economy) and is the reference implementation, parity oracle, replay/eval renderer, and **default** trainer env. **E7** adds a second, GPU-vectorizable **pure-JAX** engine (`balatro_rl/engine_jax/`) that trainers opt into (`engine="jax"`); **Phase 1 (core loop — no jokers/shop/bosses) is shipped and parity-gated**, and Phases 2–4 grow it toward full coverage. Both are written behind a clean, plain-data seam with an explicit RNG and a golden parity corpus.
- **Agent:** **JAX + maskable PPO** — entity-transformer encoder, candidate-scoring policy head, symlog two-hot (distributional) value head to survive Balatro's exponential (10²–10¹²⁺) scores. The trainer runs on either engine — Python (full game, default) or the JAX core engine (GPU-accelerated, opt-in via `TrainConfig.engine="jax"`).
- **Fidelity:** the real game via [`balatrobot`](https://github.com/coder/balatrobot) is the parity oracle and the ground-truth agent evaluator.
- **Objective is a research variable:** reward is pluggable (`win_ante8` / `pure_score` / `max_depth`); comparing routes is a first-class experiment.
- **Compute:** RunPod (containerized, checkpointable, resumable).
- **Built to be seen:** observability first — a Trackio dashboard and a clean **replay viewer** (board + score breakdown + the agent's action distribution / value / reward) because in RL, observability *is* the debugger.

## Build approach

Build the spine **engine → env → random/heuristic agent → replay viewer → dashboard first**, then the PPO learning loop, then climb engine tiers (Tier 0 MVP → full game) behind parity gates. We iterate fast.

## Docs

- [Design spec](docs/specs/2026-06-01-balatro-rl-design.md) — full architecture, observation/action/reward contract, agent network, engine tiers, eval/parity, and tooling.
- [E7 — JAX engine design](docs/specs/2026-06-06-jax-engine-refactor-design.md) + [Phase 0+1 plan](docs/plans/2026-06-06-jax-engine-phase01-plan.md) — the GPU-native vectorized engine (Phase 1 core loop shipped, parity-gated; Phases 2–4 add jokers → shop → bosses).

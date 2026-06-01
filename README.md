# Balatro RL

Training an agent **from scratch with reinforcement learning** to play [Balatro](https://balatrowiki.org/) and score as high as possible.

> **Status:** design phase complete. Implementation planning next. No code yet.

## The shape of the project

- **Engine:** pure **Python**, written behind a clean, Rust-portable seam (isolated module, plain-data state, explicit RNG, golden parity corpus) so a compiled port later is a surgical swap, not a rewrite.
- **Agent:** **JAX + maskable PPO** — entity-transformer encoder, candidate-scoring policy head, symlog two-hot (distributional) value head to survive Balatro's exponential (10²–10¹²⁺) scores.
- **Fidelity:** the real game via [`balatrobot`](https://github.com/coder/balatrobot) is the parity oracle and the ground-truth agent evaluator.
- **Objective is a research variable:** reward is pluggable (`win_ante8` / `pure_score` / `max_depth`); comparing routes is a first-class experiment.
- **Compute:** RunPod (containerized, checkpointable, resumable).
- **Built to be seen:** observability first — a Trackio dashboard and a clean **replay viewer** (board + score breakdown + the agent's action distribution / value / reward) because in RL, observability *is* the debugger.

## Build approach

Build the spine **engine → env → random/heuristic agent → replay viewer → dashboard first**, then the PPO learning loop, then climb engine tiers (Tier 0 MVP → full game) behind parity gates. We iterate fast.

## Docs

- [Design spec](docs/specs/2026-06-01-balatro-rl-design.md) — full architecture, observation/action/reward contract, agent network, engine tiers, eval/parity, and tooling.

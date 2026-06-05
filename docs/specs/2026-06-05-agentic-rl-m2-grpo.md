# Agentic-RL M2 — Multi-turn GRPO training — Design + Runbook

- **Date:** 2026-06-05
- **Status:** Design approved (continues the E6 agentic-RL track); building now. verl-agent API grounded against current docs; verl-specific code is **pod-only** (cannot run/test on a CPU dev box) and must be verified against the installed verl-agent version.
- **Depends on:** M1 (`docs/specs/2026-06-04-agentic-rl-design.md`, shipped in PRs #21/#25) — the `balatro_rl/llm/` boundary (serialize / menu / parse / context / LLMAgent / baseline).

## 1. Goal

Train the LLM agent with **multi-turn GRPO** (verl-agent) + **LoRA** so it beats the frozen M1 baseline (Qwen3-8B = win_rate 0.000, mean_final_ante 1.00). The M1 result is a weak floor: a frozen non-thinking 8B plays legal-but-weak (3-card hands, no discards, no poker reasoning) and never clears blind 1.

**The learning-signal problem (drives the whole design):** GRPO learns from **reward variance within a group** of rollouts on the same seed. If every rollout loses at ante 1 with near-identical reward, the group-relative advantage ≈ 0 and nothing is learned. Two mitigations, both adopted:
1. **Curriculum:** start at low `req_scale` (trivial blind targets) so *some* rollouts clear blinds → reward variance → gradient signal; ramp `req_scale` 0.1 → 1.0 as the rolling clear-rate rises (mirrors the E5 PPO curriculum).
2. **Dense reward:** use the existing `shaped` reward (per-turn potential shaping + clear/win bonuses), not sparse win/loss. With `gigpo`'s step-advantages this gives per-decision credit across the ~300-turn horizon.

## 2. Why verl-agent (not base verl / TRL)

`verl-agent` (langfengq/verl-agent) extends verl with a **step-independent multi-turn rollout** + a custom-environment interface (`EnvironmentManagerBase`) that maps 1:1 onto the M1 boundary. Base verl's Agent Loop is tool-calling-oriented; TRL's multi-turn is newer. verl-agent gives us, out of the box: `gigpo`/`grpo`/`rloo` estimators, LoRA, vLLM/SGLang rollouts, `use_invalid_action_penalty`, and a `success_evaluator` hook that reads `info["won"]` — all of which we want.

**The adapter is thin** because M1 already did the hard part:
- verl-agent `projection_f(text) -> (action, valid)` ≡ M1 `parse_action(reply, state) -> ParseResult`.
- verl-agent `{"text": ...}` observation ≡ M1 `serialize_state(state) + render_menu(build_menu(state))`.
- per-step `rewards` ≡ M1 `envs.rewards.make_reward("shaped")`.
- `info["won"]`/`info["ante"]` ≡ already surfaced by `BalatroEnv.step`.

## 3. Architecture

```
balatro_rl/llm/
  text_env.py     # BalatroTextEnv: single-game text env (reset->text obs; step(action_text)->(text,reward,done,info)).
                  #   verl-AGNOSTIC, fully unit-tested. Wraps BalatroEnv + serialize + render_menu + parse_action.
                  #   Curriculum-aware: reset(seed, req_scale). Returns info{won, ante, cleared, is_action_valid}.
  config.py       # ExperimentConfig (model/lora, reward, algo, rollout, context, curriculum) -> verl-agent Hydra overrides.
  verl_env.py     # BalatroEnvManager(EnvironmentManagerBase): vectorizes N BalatroTextEnv; reset/step/build_text_obs/
                  #   success_evaluator + projection_f via parse_action. POD-ONLY (imports verl-agent's agent_system).
  curriculum.py   # ReqScaleCurriculum: rolling clear-rate -> req_scale ramp (shared by text_env tests + env manager).
  train_grpo.py   # entrypoint: compose the verl-agent config from ExperimentConfig + launch. POD-ONLY.
configs/
  balatro_grpo.yaml   # verl-agent config: gigpo + LoRA + curriculum + Balatro env. POD-ONLY (run target).
docs/
  RUNPOD_M2.md        # how to install verl-agent + launch on the pod.
```

**Locally testable (TDD, like M1):** `text_env.py`, `config.py`, `curriculum.py`.
**Pod-only (grounded against docs, verified on the pod):** `verl_env.py`, `train_grpo.py`, `configs/balatro_grpo.yaml`.

## 4. BalatroTextEnv contract (the substrate)

```python
class BalatroTextEnv:
    def __init__(self, reward_name="shaped", enable_bosses=False): ...
    def reset(self, seed: int, req_scale: float = 1.0) -> tuple[str, dict]:
        # -> (observation_text, info). observation_text = serialize_state + "\n\n" + render_menu.
    def step(self, action_text: str) -> tuple[str, float, bool, dict]:
        # parse_action(action_text, state) -> validated action_id (or invalid);
        # on invalid: no engine step, reward=0, info["is_action_valid"]=False (verl-agent applies the penalty);
        # on valid: BalatroEnv.step(action_id); reward from `shaped`; info{won, ante, cleared, is_action_valid=True}.
        # -> (next_observation_text, reward, done, info)
```
This is the single seam every later layer builds on, and it is exercised end-to-end by a scripted-stub test (reusing M1's `ScriptedStubPolicy` pattern) — proving a full multi-turn game runs through the text interface with curriculum `req_scale`.

## 5. BalatroEnvManager (verl-agent, pod-only)

Subclasses `agent_system.environments.base.EnvironmentManagerBase`:
- `reset(kwargs)`: per-env `BalatroTextEnv.reset(seed_i, req_scale=curriculum.current)`; returns `{"text": [...], "image": None, "anchor": [...]}, infos`.
- `step(text_actions)`: `projection_f` = `parse_action` per env → (action_id, valid); step each env; return `({"text":[...]}, rewards, dones, infos)` with `info["is_action_valid"]` set so `use_invalid_action_penalty` fires on parse failures.
- `success_evaluator(...)`: returns `{"success_rate": mean(info["won"])}` — feeds the curriculum ramp and the deploy metric.
- Registered as `env_name: "balatro"` for `make_envs(config)`.

## 6. Config (the run)

`ExperimentConfig` defaults (overridable), emitted as verl-agent Hydra overrides:
- **algorithm:** `adv_estimator=gigpo` (multi-turn step+episode advantages; `grpo` selectable), `gamma=0.99`, `use_kl_loss=True`, `kl_loss_coef=0.01`.
- **model:** Qwen3-8B (or 14B), `lora_rank=32`, `lora_alpha=16`, `target_modules=all-linear`.
- **rollout:** `name=vllm`, `n=8` (group size; `env.rollout.n` matches), `temperature=1.0`, `--no-thinking` analog via `+data.apply_chat_template_kwargs.enable_thinking=False`, `max_response_length` sized for brief reasoning + JSON.
- **env:** `env_name=balatro`, `max_steps` ≈ 350 (a full run), `history_length` per the M1 context window.
- **curriculum:** `req_scale` 0.1 → 1.0 ramp on rolling clear-rate (start trivial for signal).
- **trainer:** LoRA on 1×80GB; `save_freq` periodic; vLLM ≈ 0.6 gpu_memory_utilization.

## 7. Build sequence

1. `text_env.py` + tests (full multi-turn game via stub, curriculum req_scale). Locally tested.
2. `curriculum.py` + tests (clear-rate → req_scale ramp; reuse E5 curriculum logic). Locally tested.
3. `config.py` + tests (ExperimentConfig → Hydra-override list; defaults reproduce §6). Locally tested.
4. `verl_env.py` (BalatroEnvManager + projection). Pod-only; import-guarded so the package imports without verl-agent installed.
5. `train_grpo.py` + `configs/balatro_grpo.yaml` + `docs/RUNPOD_M2.md`. Pod-only.
6. Full suite green (the verl-agnostic additions); commit + PR.

## 8. Non-goals / risks
- **Untested verl integration:** `verl_env.py`/`train_grpo.py`/the YAML cannot run on a CPU box and the verl-agent API may differ by version — they are written against current docs and **must be verified on the pod** (first launch is a smoke test, not a trusted run). Import-guard verl-agent so the rest of `balatro_rl` (and CI) is unaffected.
- **No bignum scoring / Endless** (still M3).
- **The replay/req_scale determinism gap** (known, separate) does not block training (it affects `replay()`, not the live rollout).

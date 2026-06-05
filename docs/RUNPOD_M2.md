# RunPod runbook — E6 M2: multi-turn GRPO/GiGPO on Balatro (verl-agent)

POD-ONLY. The `balatro_rl/llm/` M2 code (`text_env`, `curriculum`, `config`, `verl_env`,
`train_grpo`) is written against verl-agent's current docs but **cannot be run/tested on a CPU
box**. Treat the first launch as a smoke test, not a trusted run. Design: `docs/specs/2026-06-05-agentic-rl-m2-grpo.md`.

## 0. Prereqs
- 1×80GB GPU (A100/H100). LoRA + an 8B fits with vLLM at ~0.6 gpu_memory_utilization.
- The E5 checkpoint and this repo are on the persistent network volume (`/workspace`).

## 1. Install
```bash
# this repo (engine + the LLM boundary the env adapter imports)
cd ~/balatro-e6 && pip install -e '.[llm]'
# verl-agent (multi-turn agentic RL on top of verl). The [vllm] extra brings a consistent
# torch 2.8 + vLLM; we DO NOT install the [gpu] (flash-attn) extra — this pod has no nvcc, and
# HF transformers auto-falls-back to SDPA attention for the actor, while vLLM uses its own.
git clone https://github.com/langfengq/verl-agent ~/verl-agent
cd ~/verl-agent && pip install -e '.[vllm]'
```
> Note: this upgrades torch (pod ships 2.4.1) to the `[vllm]`-pinned build. If a wheel/CUDA
> mismatch appears, pin a compatible `vllm` within verl-agent's `vllm>=0.8.5,<=0.11.0` range.

## 2. Register the `balatro` env into verl-agent
All the adapter code (vec env, projection, manager, factory) lives in this repo
(`balatro_rl/llm/verl_env.py`, installed via `pip install -e '.[llm]'`) and is verified against
verl-agent's real `EnvironmentManagerBase(envs, projection_f, config)` contract. The only pod-side
edit is a **3-line hook** in `agent_system/environments/env_manager.py`'s `make_envs(config)` —
add it as the first branch:
```python
    if "balatro" in config.env.env_name.lower():
        from balatro_rl.llm.verl_env import make_balatro_envs
        return make_balatro_envs(config)   # -> (train_manager, val_manager)
```
That's it — no symlink, no manager edits. `make_balatro_envs` builds the train/val
`BalatroEnvManager`s (each group of `rollout.n` rollouts shares a seed; curriculum ramps req_scale).

## 3. Placeholder dataset
verl-agent still expects train/val parquet even when the env drives the task; a minimal stub suffices:
```bash
python - <<'PY'
import os, pandas as pd
os.makedirs(os.path.expanduser("~/data/balatro"), exist_ok=True)
row = {"data_source": "balatro", "prompt": [{"role": "user", "content": "play"}],
       "ability": "balatro", "reward_model": {"style": "rule", "ground_truth": ""},
       "extra_info": {"index": 0}}
for split in ("train", "val"):
    pd.DataFrame([row] * 16).to_parquet(os.path.expanduser(f"~/data/balatro/{split}.parquet"))
PY
```

## 4. Launch
Print the exact command from the config (no guessing Hydra overrides):
```bash
python -m balatro_rl.llm.train_grpo --model Qwen/Qwen3-8B --print-only
```
Then launch on the pod (or pass `--launch`). Equivalent direct form using the YAML:
```bash
cd ~/verl-agent
python -m verl.trainer.main_ppo --config-path ~/balatro-e6/configs --config-name balatro_grpo
```

## 5. Smoke test before trusting it
- Set `trainer.total_epochs=1`, `data.train_batch_size=2`, `env.rollout.n=2` for a fast pass.
- Confirm: rollouts step the Balatro env (text obs flowing), `is_action_valid` toggling, reward
  variance within a group (NOT all-zero — that's the M1 signal problem; if so, lower
  `req_scale_start`), and `success_rate` logging.
- Then scale back up to the §4 config.

## 6. Resume / artifacts
- Checkpoints: `trainer.default_local_dir=/workspace/sweep_out/e6_m2` (persistent volume).
- The deploy metric is `success_rate` (win-rate) from `BalatroEnvManager.success_evaluator`.
- Curriculum `req_scale` ramps automatically from the rolling clear-rate (see `curriculum.py`).

## 7. GPU co-existence with E5 PPO
E5 (JAX PPO) is CPU-bound (~9GB GPU); M2 vLLM is GPU-bound. They co-exist on one 80GB card
(set `gpu_memory_utilization` so vLLM leaves PPO its ~9GB), exactly as in the E5+E6 brief.

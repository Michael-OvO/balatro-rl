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
# verl-agent (multi-turn agentic RL on top of verl)
git clone https://github.com/langfengq/verl-agent ~/verl-agent
cd ~/verl-agent && pip install -e . && pip install vllm
```

## 2. Register the `balatro` env into verl-agent
verl-agent resolves `env.env_name` through its env registry / `make_envs`. Wire ours in:
```bash
# expose BalatroEnvManager to verl-agent's environment package
ln -s ~/balatro-e6/balatro_rl/llm/verl_env.py \
      ~/verl-agent/agent_system/environments/balatro_env.py
```
Then add a registry entry so `env_name: balatro` resolves to `BalatroEnvManager` (in
`agent_system/environments/env_manager.py`'s `make_envs`):
```python
# inside make_envs(config): map config.env.env_name == "balatro"
if config.env.env_name == "balatro":
    from agent_system.environments.balatro_env import BalatroEnvManager
    b = config.env.balatro
    make = lambda: BalatroEnvManager(
        n=config.env.rollout.n, reward_name=b.reward_name,
        enable_bosses=b.enable_bosses, req_scale_start=b.req_scale_start,
        req_scale_end=b.req_scale_end, seed=config.env.seed)
    return make(), make()   # (train_envs, val_envs)
```
> Verify `EnvironmentManagerBase`'s constructor + the `make_envs` return contract against the
> installed verl-agent version; adapt `BalatroEnvManager.__init__`/`reset`/`step` shapes if they differ.

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

# RunPod runbook — E6 M2: multi-turn GiGPO/GRPO on Balatro (verl-agent)

**VALIDATED end-to-end on a RunPod H100 (2026-06-06):** vLLM loaded, the LLM rolled out on the
Balatro env via the `make_balatro_envs` hook, GiGPO computed advantages, and a LoRA train step
completed (GPU ~86% / 40 GB, 0 errors, `success_rate=0.000` on the 0.5B smoke — expected; the
smoke proves wiring, not play quality). `scripts/balatro_grpo.sh` is the exact validated launcher.
Design: `docs/specs/2026-06-05-agentic-rl-m2-grpo.md`.

## 0. Prereqs
- 1×80 GB GPU (H100/A100), driver supporting CUDA 12.8 (torch 2.8 cu128). Disk: an 8B model is
  ~16 GB — keep room on `/` (root, ephemeral) or point HF cache at the network volume.
- The E5 checkpoint + this repo's working copy live under `~` / the persistent volume (`/workspace`).

## 1. Install
```bash
# this repo (engine + the LLM boundary the env adapter imports)
cd ~/balatro-e6 && pip install -e '.[llm]'
# verl-agent (multi-turn agentic RL on verl). The [vllm] extra brings torch 2.8 + vLLM 0.11.
git clone https://github.com/langfengq/verl-agent ~/verl-agent
cd ~/verl-agent && pip install -e '.[vllm]'
```
**flash-attn is REQUIRED** (verl's actor hard-imports `flash_attn.bert_padding` at module load —
SDPA fallback is NOT enough). There's no `nvcc` to compile it, so install the matching **prebuilt
wheel** (auto-resolves torch/cu/cp + cxx11-abi from the installed torch):
```bash
python3 - <<'PY'
import json, urllib.request, subprocess, torch
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
rel = sum((json.load(urllib.request.urlopen(
    f"https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=30&page={p}")) for p in (1,2,3)), [])
w = [a["browser_download_url"] for r in rel for a in r.get("assets", [])
     if "torch2.8" in a["name"] and "cp311" in a["name"] and f"abi{abi}" in a["name"] and a["name"].endswith(".whl")]
assert w, "no matching flash-attn wheel"; subprocess.run(["pip","install","--no-deps",w[0]], check=True)
PY
```
> torch is upgraded (pod ships 2.4.1 → `[vllm]` pins 2.8). vLLM uses the **FLASH_ATTN** backend
> (set via env var at launch); **xformers is NOT installed** (the XFORMERS backend will assert).

## 2. Register the `balatro` env into verl-agent
All adapter code (vec env, projection, manager, factory) is in this repo
(`balatro_rl/llm/verl_env.py`, installed via `pip install -e '.[llm]'`) and matches verl-agent's
real `EnvironmentManagerBase(envs, projection_f, config)` contract. The only pod-side edit is a
**3-line hook** — add it as the first branch of `make_envs(config)` in
`agent_system/environments/env_manager.py`:
```python
    if "balatro" in config.env.env_name.lower():
        from balatro_rl.llm.verl_env import make_balatro_envs
        return make_balatro_envs(config)   # -> (train_manager, val_manager)
```

## 3. Placeholder dataset
verl-agent still expects train/val parquet even though the env drives the task:
```bash
python - <<'PY'
import os, pandas as pd
os.makedirs(os.path.expanduser("~/data/balatro"), exist_ok=True)
row = {"data_source": "balatro", "prompt": [{"role": "user", "content": "play"}],
       "ability": "balatro", "reward_model": {"style": "rule", "ground_truth": ""},
       "extra_info": {"index": 0}}
for split, n in (("train", 64), ("val", 16)):   # train rows MUST be >= train_batch_size or
    pd.DataFrame([row] * n).to_parquet(os.path.expanduser(f"~/data/balatro/{split}.parquet"))
PY                                                # verl errors "Train dataloader is empty!"
```

## 4. Launch (the validated recipe)
Canonical launcher — defaults are the proven smoke config; env vars scale it up:
```bash
bash ~/balatro-e6/scripts/balatro_grpo.sh                                   # smoke (Qwen2.5-0.5B, fast)
MODEL=Qwen/Qwen3-8B LORA=32 TRAIN_BS=16 GROUP=8 MAX_STEPS=150 REQ_START=0.1 EPOCHS=8 \
  bash ~/balatro-e6/scripts/balatro_grpo.sh                                 # real first run (~hours; scale up later)
```
Equivalent programmatic form (prints/launches the same bare overrides on verl's `ppo_trainer` base):
```bash
python -m balatro_rl.llm.train_grpo --model Qwen/Qwen3-8B --print-only      # or --launch
```
> Launch is **bare overrides** on verl's default `ppo_trainer` config — NOT `--config-name`
> (the base already has the env/gigpo/algorithm sections). Key validated settings:
> `actor_rollout_ref.rollout.n=1` (GRPO group = `env.rollout.n`), the `*_micro_batch_size_per_gpu`
> fields (verl errors without them), `use_invalid_action_penalty=True`, `+env.balatro.*`.

## 5. Operational gotchas (cost us real time — read before relaunching)
- **`pkill -f` self-matches your shell.** `pkill -9 -f verl.trainer.main_ppo` kills the SSH shell
  running it (its argv contains that string). Use the bracket trick: `pkill -9 -f 'verl.trainer.[m]ain_ppo'`.
- **tmux is not installed.** Detach with `setsid`: `( setsid bash scripts/balatro_grpo.sh </dev/null >~/m2.log 2>&1 & )`.
- **Kill stale Ray/workers between runs** or a new run reuses a stale cluster:
  `pkill -9 -f '[r]aylet'; pkill -9 -f '[r]ay::'; ray stop --force`.
- RunPod's exposed-TCP SSH rate-limits rapid connections; the `ssh.runpod.io` proxy needs `-tt`.

## 6. What "working" looks like (from the validated run)
- GPU memory climbs ~0.5 GB → a few GB (vLLM weights + KV) → util spikes during the train step.
- Log shows `Capturing CUDA graphs`, `gigpo`, `rollout`, then `Training Progress: N/total`.
- `success_rate` (win-rate) from `BalatroEnvManager.success_evaluator` is the deploy metric;
  curriculum `req_scale` ramps from the rolling clear-rate (`curriculum.py`). Start
  `req_scale_start` low so early rollouts clear blinds → reward variance → signal.

## 7. Live dashboards (opt-in — both default to console)
Install once on the pod: `pip install trackio wandb`.
- **PPO sweep → Trackio (HuggingFace Space):** `huggingface-cli login` (or `HF_TOKEN`), then
  `TRACKIO_SPACE=<hf-user>/balatro-e5 bash scripts/ppo_sweep.sh` — every run streams to one Space,
  each a distinct named run (`BALATRO_RUN_NAME`); a browser URL that survives the pod. Unset → console.
- **8B GiGPO → Weights & Biases:** `export WANDB_API_KEY=...`, then
  `LOGGER='[console,wandb]' MODEL=Qwen/Qwen3-8B … bash scripts/balatro_grpo.sh` — verl logs
  reward / success_rate / GPU-util to wandb.
- Console is always live: `tail -f /workspace/grpo_8b.log`, `tail -f /workspace/<sweep>/run_*/train.log`.

## 8. The throughput reality — UPDATED for the E7 JAX core engine (2026-06-07)

**The old bottleneck (measured 2026-06-06):** every track funnelled through the pure-Python,
object-based engine. A single E5 PPO run was serial-Python (`SyncVectorEnv` loops `BalatroEnv`),
~72 s/update at 64 envs × 128 steps ≈ **~114 env-steps/s** with **zero GPU**; the 8B agentic run was
rollout-bound (GPU ~0–20 %, peaks ~87 %) waiting on CPU env-steps. No flag fixed it — the env had to
move onto the GPU.

**The fix (E7):** a GPU-native, branchless, `vmap`-able JAX core engine (`balatro_rl/engine_jax/`),
proven bit-for-bit equal to the Python oracle on the core loop and dropped under PPO via
`JaxVectorEnv` (a `SyncVectorEnv`-compatible drop-in; flip with `TrainConfig.engine="jax"`).

**Phase-1 results (the go/no-go):**
- **Parity ✅** — `tests/engine_jax/test_core_parity_gate.py`: **1000** (seed × random-legal-action)
  rollouts to termination agree with the Python engine on every within-blind transition
  (scalars + ordered hand slots + the full core observation + the shaped reward) — **12,147
  within-blind transitions + 1,914 blind boundaries, zero mismatches**. (A 200-rollout subset runs by
  default in CI; the 1000-rollout gate is `@pytest.mark.slow`, opt in with `BALATRO_RUN_SLOW=1`.)
- **Throughput** (`scripts/bench_jax_engine.py`, 200-step `lax.scan` over `batched_step`): on an
  **M4 CPU** ~**85k env-steps/s @ 1k envs** and ~**90k @ 10k** (few cores → wall-time ~linear in
  batch, throughput ≈ flat); on a **higher-core CPU** the 10k batch scales further (~190k observed) —
  the absolute numbers are host-dependent. Either way it's **~700–1600× the old ~114 env-steps/s
  Python pace, CPU-to-CPU** — the JAX win is the branchless `vmap`+`jit` engine, not the hardware.
  On a **GPU** the `vmap` lanes run in parallel, so wall-time stays ~flat as the batch grows →
  env-steps/s scales with batch size and the device saturates; the CPU ~700×+ is the floor, not the ceiling.
- **Learning ✅ (smoke)** — `tests/agent/test_jax_engine_smoke.py`: PPO trains end-to-end on the JAX env
  with finite losses (a full learning curve is the next run).

**Still to run on a CUDA box** (criterion #2's GPU number — the pod was shut down to halt spend):
`BENCH_SIZES=1000,10000,50000 python scripts/bench_jax_engine.py` to record env-steps/s **and**
`nvidia-smi` GPU utilization (target ≥ ~80 % at ≥10k envs). The script auto-prints `nvidia-smi` when a
CUDA device is present. Install `jaxlib-cuda` on the box first (the pod historically shipped CPU jaxlib).

**Scope note:** Phase 1 is the CORE loop only (deal → play/discard → score → blind/ante → win/lose).
Jokers/shop/consumables/vouchers/bosses are Phase 2+; the Python engine remains the oracle and the home
of those features. The agentic-LLM track plugs into the same batched engine from Phase 1 onward.

**Legacy 8B/PPO-sweep guidance (still valid for the Python-engine path):** right-size or it runs for
weeks — a GiGPO step (128 eps × 350 turns) is >30 min → 600 steps ≈ 2 weeks; use `EPOCHS≈8` (~32 steps)
+ `MAX_STEPS≈150`, and ~150 warm-started PPO updates (not 2000). Run Python-engine PPO as a parallel
sweep (`scripts/ppo_sweep.sh`); the JAX engine makes that sweep obsolete once the GPU number is confirmed.

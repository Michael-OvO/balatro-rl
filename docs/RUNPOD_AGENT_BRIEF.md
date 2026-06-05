# RunPod H100 Agent Brief — Balatro RL, co-located training

**You are a fresh agent running ON a RunPod H100 pod, with no prior chat context. This file is
your complete brief.** Goal: run BOTH Balatro RL tracks on this one card, cleanly partitioned —
the deep-RL PPO training and the agentic-RL LLM baseline eval.

---

## 0. Context (everything you need to know)

This repo trains agents to play **Balatro** (a poker-roguelike). Two policy tracks share one
engine/env/reward/eval/replay spine:

- **E5 — deep-RL:** a JAX/Flax **maskable-PPO** card-aware actor-critic (~4.7M params). **Fully
  built and ready on `master`.** Entrypoint: `python -m balatro_rl.agent.retrain`.
- **E6 — agentic-RL:** an **LLM plays Balatro via a text interface** ("just another policy"). Only
  **M1 = the frozen-baseline EVAL** exists — *no training yet* (GRPO/LoRA is M2, not built). Its
  code is on **branch `e6-agentic-rl-m1` / PR #21**, NOT on master. Entrypoint:
  `python -m balatro_rl.llm.baseline` (needs a served LLM endpoint).

**This box:** H100 80GB · **256 CPU cores** · ~30GB container disk (`/`) · 9.4GB `/workspace` volume.

### The one fact that drives everything: PPO is CPU-bound, not GPU-bound
The PPO model is tiny (4.7M params); its limiter is **Python env-stepping on CPU**. Consequences:
1. **The 256 cores are the real asset** — set `num_envs` high.
2. **The H100 sits ~idle during PPO** — expected, NOT a bug. (Don't "fix" it.)
3. **That idle GPU is why we can co-locate an LLM on the same card.**

### Clean memory partition (this is "efficient and clean")
| Job | GPU memory | How |
|---|---|---|
| E5 PPO (JAX) | cap ~12GB | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15` + `XLA_PYTHON_CLIENT_PREALLOCATE=false` (it really uses ~4GB) |
| E6 LLM (vLLM) | ~48GB | `--gpu-memory-utilization 0.6` |
Together ~0.75 of 80GB, ~20GB margin. **Never run two heavy GPU jobs** — the partition works only
because PPO is tiny.

---

## 1. STEP ONE — launch the E5 PPO run (priority; it's the long job)

```bash
cd ~ && git clone https://github.com/Michael-OvO/balatro-rl 2>/dev/null; cd balatro-rl
git checkout master && git pull
pip install -r requirements-cuda.txt && pip install -e .
python -c "import jax; print(jax.devices())"          # MUST list a CudaDevice (H100), not CpuDevice
```
If `jax.devices()` shows CPU, the wrong JAX is installed — reinstall `jax[cuda12]` (it's in
`requirements-cuda.txt`); do not proceed on CPU.

**Preflight smoke (~1 min — proves the whole stack trains/evals/checkpoints/records):**
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m balatro_rl.agent.retrain --smoke
```
Expect: `(GPU)`, a few updates with eval lines, `checkpoint @ update 2`, two recorded replays,
`DONE`. If that passes, launch the real run:

```bash
tmux new -s ppo            # so it survives disconnects
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 XLA_PYTHON_CLIENT_PREALLOCATE=false
export BALATRO_EPISODE_DIR=/workspace/sweep_out
export BALATRO_NUM_ENVS=192 BALATRO_NUM_MB=24    # 192*64/24 = 512 minibatch (use the 256 cores)
export BALATRO_UPDATES=6000 BALATRO_EARLY_STOP=8 BALATRO_CHECKPOINT_EVERY=50
# optional live dashboard (needs an HF token): huggingface-cli login
#   export BALATRO_TRACKIO_SPACE=<your-hf-user>/balatro-e5
python -m balatro_rl.agent.retrain 2>&1 | tee /workspace/ppo.log
# detach with Ctrl-b then d
```

**What "healthy" looks like** (watch `tail -f /workspace/ppo.log`):
- `train/req_scale` and `train/boss_rate` ramp **0.2 → 1.0** as the agent starts clearing.
- `eval @ N | blinds … ante … max …` climbing over time (the deploy-target metric).
- `BALATRO_EARLY_STOP=8` ends the run when eval plateaus for 8 evals at full difficulty.
- Checkpoints land at `/workspace/sweep_out/retrain_e5_ckpt.msgpack` every 50 updates.
- Crash recovery: relaunch with `BALATRO_RESUME=/workspace/sweep_out/retrain_e5_ckpt.msgpack`.

**Tuning the cores:** after ~2 min, check the per-update time. If CPU isn't saturated (`htop`),
raise `BALATRO_NUM_ENVS` (and `BALATRO_NUM_MB` to keep `num_envs*64/NUM_MB` ≈ 512 and evenly
divisible). The PPO GPU staying near-idle is correct — don't chase GPU utilization here.

---

## 2. STEP TWO — the E6 M1 LLM baseline (co-located on the same card)

The E6 code isn't on master. Bring it in (use a SECOND clone/worktree so the PPO run's checkout
is undisturbed):
```bash
cd ~ && git clone https://github.com/Michael-OvO/balatro-rl balatro-e6 2>/dev/null; cd balatro-e6
git fetch origin && git checkout e6-agentic-rl-m1 && pip install -e . && pip install vllm
```

**Serve an LLM** (⚠️ CONFIRM the model with the user first — see §3):
```bash
export HF_HOME=/workspace/hf       # keep weights off the small root disk
tmux new -s vllm
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000 \
     --gpu-memory-utilization 0.6 --max-model-len 8192
# detach (Ctrl-b d); wait until it logs "Uvicorn running on ... :8000"
```

**Run the baseline** (once vLLM is up):
```bash
python -m balatro_rl.llm.baseline \
     --model Qwen/Qwen2.5-3B-Instruct --base-url http://localhost:8000/v1 --seeds 0-31
```
This reports the **Ante-8 win rate + ante-depth distribution** vs Random/Greedy/PPO, with
chain-of-thought replays — the agentic-RL "should we invest in this track?" go/no-go number.

---

## 3. ⚠️ Confirm with the user before spending on these

- **Which LLM model?** Disk is the constraint (`/` ≈ 30GB after torch+vllm leaves ~6GB; `/workspace`
  is 9.4GB). **Qwen2.5-3B-Instruct (~6GB)** fits comfortably and is the safe default, but a 3B may
  play Balatro weakly. **A 7B (~15GB)** is the E6 design's intent (better poker knowledge) but is
  *tight* on 30GB — may need a bigger disk/volume. Ask which they want.
- **PR #21 vs branch?** Running E6 from the `e6-agentic-rl-m1` branch is fine; merging #21 to master
  is the alternative. Don't merge an open PR without the user's OK.
- **Don't download a model or start a long run without confirming** — these cost money/time.

---

## 4. Report back

- **E5:** final eval (mean blinds cleared / ante / win rate), whether it early-stopped, and the
  artifacts: `/workspace/sweep_out/retrain_e5_params.msgpack` + `retrain_e5_seed{4,7}.episode.json`.
- **E6:** the baseline's printed win-rate / ante-depth table.
- **Card usage:** confirm `nvidia-smi` shows vLLM holding the bulk and PPO ~4GB (the partition working).

## 5. Gotchas
- PPO GPU near-idle = expected (CPU-bound). The CPU (`htop`) should be busy, not the GPU.
- `jax.devices()` shows CPU → wrong jax; install `jax[cuda12]` from `requirements-cuda.txt`.
- `num_envs*num_steps` (num_steps=64) must divide evenly by `NUM_MB`, else the loader silently drops rows.
- A 7B on the 30GB root is tight after torch+vllm; prefer 3B, or set `HF_HOME=/workspace/hf` and verify free space first (`df -h`).
- Start the PPO run first (priority); vLLM measures free GPU memory at startup, so launching it after PPO (which uses ~4GB) is fine.

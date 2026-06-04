# Running the E5 retrain on a GPU (RunPod)

The agent trains with maskable PPO in JAX. Profiling puts **~89% of the wall-clock in the PPO
backward pass** (GPU-accelerable) and only ~7% in the Python env stepping (CPU-only). So a single
modern GPU gives roughly **5–8×** over this Mac's CPU. The env stays on CPU regardless — that's
the small slice — so a huge GPU is wasted; a mid GPU (A40 / L40S / 4090 / A100) is the sweet spot.

## 1. Pick a pod
- Any CUDA-12 image works (e.g. RunPod's `runpod/pytorch:2.x-cuda12.x` or a bare `nvidia/cuda:12-runtime`).
- 1 GPU, ~8 vCPU (the env stepping is CPU-bound, so vCPUs matter for rollout throughput), 16–24 GB RAM.

## 2. Install
```bash
git clone <this repo> && cd balatro-rl
pip install -r requirements-cuda.txt      # CUDA jaxlib + plugin (bundled CUDA libs; needs only the NVIDIA driver)
pip install -e .                          # the balatro_rl package
python -c "import jax; print(jax.devices())"   # must list a CudaDevice, NOT CpuDevice
```
If `jax.devices()` shows only CPU, the CUDA plugin didn't load — check `nvidia-smi` works in the pod
and that `requirements-cuda.txt` (not the plain CPU `jax`) was installed.

## 3. Smoke test (do this first — ~1 min)
Proves the full E5 stack trains end-to-end on the GPU before committing to the long run:
```bash
python -m balatro_rl.agent.retrain --smoke
```
Expect: `JAX devices: [CudaDevice(id=0)] (GPU)`, a handful of updates with eval lines, two recorded
replays, and `[retrain] DONE`. No crash = the obs/action/network contract is intact on the GPU.

## 4. Full run
```bash
# defaults: d_model 256, 64 envs, 2000 updates, curriculum floor 0.2, boss curriculum on
nohup python -m balatro_rl.agent.retrain > retrain.log 2>&1 &
tail -f retrain.log
```
Tune via env vars (no code edit):
```bash
BALATRO_DMODEL=384 BALATRO_NUM_ENVS=128 BALATRO_UPDATES=3000 \
  python -m balatro_rl.agent.retrain
```
- `BALATRO_DMODEL`  network width (default 256)
- `BALATRO_NUM_ENVS` parallel envs (default 64 — raise if the GPU is underused and you have vCPUs)
- `BALATRO_UPDATES`  PPO updates (default 2000)
- `BALATRO_EPISODE_DIR` where params + replay JSONs are written (default `/tmp/sweep_out`)

## 5. What "good" looks like
Watch the `eval @ N | blinds ... ante ... max ...` lines (greedy eval on the **real** deploy game:
full req_scale, full bosses). `train/req_scale` and `train/boss_rate` should ramp 0.2 → 1.0 as the
agent starts clearing (the boss curriculum fades bosses in with the score bar — the fix for the
plateau we saw earlier). The previous baseline plateaued around 1.5 blinds; E5 gives the agent the
acquisition tools (Planets to level hands, packs, vouchers) it was blind to, so it should climb past
that as it learns to *build a deck*, not just play hands.

## 6. Bring the results back
`retrain_e5_params.msgpack` + `retrain_e5_seed{4,7}.episode.json` land in `BALATRO_EPISODE_DIR`.
Copy them down (`runpodctl send` / `scp`) and load the params into `ActorCritic(action_dim=NUM_ACTIONS,
d_model=<same D_MODEL>)`; drop the episode JSONs into the viewer's episode dir to watch the run.

#!/bin/bash
# E5 PPO parallel sweep — turn a high-core box into a fleet of warm-started runs, then pick the best.
#
# Why a sweep (not one big run): the rollout (SyncVectorEnv) steps envs in a serial Python loop, so a
# SINGLE run uses only a handful of cores and more cores can't speed it up. Data-parallelism across
# processes is how throughput "scales with vCPUs" — hence OMP_NUM_THREADS=5 PER process, so N runs
# share the box without thread oversubscription.
#
# Each run warm-starts from BASE_CKPT (params only; a fresh optimizer picks up the swept LR), writes
# to its OWN dir (so runs never clobber each other's checkpoints), and varies seed x lr. d_model is
# FIXED at 256 — a warm-start must shape-match the checkpoint. Pick the winner by the highest
# eval/mean_blinds_cleared across runs (scripts/ppo_sweep_pick.sh, or grep the per-run train.log).
#
#   BASE_CKPT=/workspace/sweep_out/retrain_e5_ckpt_base.msgpack ROOT=/workspace/sweep_out/sweep \
#     bash scripts/ppo_sweep.sh
set -u
BASE_CKPT=${BASE_CKPT:-/workspace/sweep_out/retrain_e5_ckpt_base.msgpack}
ROOT=${ROOT:-/workspace/sweep_out/sweep}
UPDATES=${UPDATES:-2000}
OMP=${OMP:-5}
SEEDS=${SEEDS:-"0 1 2 3"}
LRS=${LRS:-"1e-4 3e-4 6e-4"}

if [ ! -f "$BASE_CKPT" ]; then echo "FATAL: BASE_CKPT not found: $BASE_CKPT" >&2; exit 1; fi
mkdir -p "$ROOT"
i=0
for seed in $SEEDS; do
  for lr in $LRS; do
    out="$ROOT/run_$(printf '%02d' $i)_s${seed}_lr${lr}"
    mkdir -p "$out"
    ( setsid env OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP \
        BALATRO_RESUME="$BASE_CKPT" BALATRO_EPISODE_DIR="$out" \
        BALATRO_SEED=$seed BALATRO_LR=$lr BALATRO_DMODEL=256 \
        BALATRO_UPDATES=$UPDATES BALATRO_CHECKPOINT_EVERY=50 BALATRO_EARLY_STOP=8 \
        python3 -m balatro_rl.agent.retrain </dev/null >"$out/train.log" 2>&1 & )
    echo "launched run $i: seed=$seed lr=$lr -> $out"
    i=$((i + 1))
  done
done
echo "launched $i runs under $ROOT (OMP=$OMP each). Pick best by eval/mean_blinds_cleared."

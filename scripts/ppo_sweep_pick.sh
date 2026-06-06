#!/bin/bash
# Rank E5 PPO sweep runs by their best eval/mean_blinds_cleared (parsed from each run's train.log).
# The winner's trained params are at <run_dir>/retrain_e5_params.msgpack (written when that run ends).
#   ROOT=/workspace/sweep_out/sweep bash scripts/ppo_sweep_pick.sh
set -u
ROOT=${ROOT:-/workspace/sweep_out/sweep}
printf "%-8s  %-10s  %s\n" "BEST" "STATUS" "RUN"
for d in "$ROOT"/run_*; do
  [ -d "$d" ] || continue
  log="$d/train.log"
  best=$(grep -aoE "mean_blinds_cleared[\":= ]+[0-9.]+" "$log" 2>/dev/null | grep -oE "[0-9.]+$" | sort -g | tail -1)
  if grep -qa "\[retrain\] DONE" "$log" 2>/dev/null; then status=done
  elif pgrep -f "BALATRO_EPISODE_DIR=$d" >/dev/null 2>&1; then status=running
  else status="stopped"; fi
  printf "%-8s  %-10s  %s\n" "${best:-NA}" "$status" "$(basename "$d")"
done | { read -r h; echo "$h"; sort -gr -k1; }
echo "(highest BEST = strongest agent; params -> <run>/retrain_e5_params.msgpack)"

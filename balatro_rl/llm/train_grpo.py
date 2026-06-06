"""M2 training entrypoint — compose the verl-agent launch command from an ExperimentConfig.

POD-ONLY to actually launch (needs verl-agent + vLLM + a GPU). On any box it can PRINT the
exact override command so you don't hand-assemble Hydra overrides:

    python -m balatro_rl.llm.train_grpo --model Qwen/Qwen3-8B --print-only
    python -m balatro_rl.llm.train_grpo --model Qwen/Qwen3-8B --launch   # on the pod

Registering the `balatro` env into verl-agent and preparing the placeholder parquet are
one-time pod setup steps — see docs/RUNPOD_M2.md. This entrypoint only builds + (optionally)
runs the training command; it does not perform that registration.
"""
from __future__ import annotations

import argparse
import shlex

from .config import ExperimentConfig

# verl-agent's training entrypoint. Launch is BARE overrides on its default ppo_trainer config
# (validated on the pod 2026-06-06) — NOT a custom --config-name; the base config already carries
# the env/gigpo/algorithm sections. See scripts/balatro_grpo.sh + docs/RUNPOD_M2.md.
VERL_AGENT_MAIN = "verl.trainer.main_ppo"
# vLLM attention backend: flash_attn is installed on the pod, xformers is not.
VLLM_ATTENTION_BACKEND = "FLASH_ATTN"


def build_command(cfg: ExperimentConfig) -> list[str]:
    """argv for `python -m verl.trainer.main_ppo <overrides>` (bare overrides on verl's default
    ppo_trainer config). Set VLLM_ATTENTION_BACKEND=FLASH_ATTN in the environment when launching
    (main() does this); scripts/balatro_grpo.sh is the canonical, validated launcher."""
    return ["python", "-m", VERL_AGENT_MAIN, *cfg.to_overrides()]


def _cfg_from_args(args) -> ExperimentConfig:
    return ExperimentConfig(
        model_path=args.model, adv_estimator=args.adv_estimator, group_size=args.group_size,
        lora_rank=args.lora_rank, req_scale_start=args.req_scale_start,
        req_scale_end=args.req_scale_end, reward_name=args.reward_name,
        experiment_name=args.experiment_name,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compose/launch M2 verl-agent GRPO for Balatro.")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adv-estimator", default="gigpo", choices=["gigpo", "grpo", "rloo"])
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--reward-name", default="shaped")
    ap.add_argument("--req-scale-start", type=float, default=0.1)
    ap.add_argument("--req-scale-end", type=float, default=1.0)
    ap.add_argument("--experiment-name", default="gigpo_qwen3_8b_lora")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--print-only", action="store_true", help="print the launch command and exit")
    g.add_argument("--launch", action="store_true", help="exec the launch command (pod only)")
    args = ap.parse_args()

    cmd = build_command(_cfg_from_args(args))
    print(f"VLLM_ATTENTION_BACKEND={VLLM_ATTENTION_BACKEND} \\")
    print(" ".join(shlex.quote(c) for c in cmd))
    if args.launch:
        import os
        import subprocess
        env = {**os.environ, "VLLM_ATTENTION_BACKEND": VLLM_ATTENTION_BACKEND}
        raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()

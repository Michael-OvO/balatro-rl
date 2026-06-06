#!/bin/bash
# E6 M2 — multi-turn GiGPO/GRPO on Balatro (verl-agent). THE VALIDATED LAUNCH RECIPE.
#
# This exact override set ran end-to-end on a RunPod H100 (2026-06-06): vLLM loaded, the
# LLM rolled out on the Balatro env via the BalatroEnvManager hook, GiGPO computed advantages,
# and a LoRA train step completed (GPU ~86% / 40 GB, 0 errors). See docs/RUNPOD_M2.md for setup.
#
# Hard-won settings (each fixed a real failure the smoke test surfaced — do not "simplify" away):
#   - actor_rollout_ref.rollout.n=1            : verl+env keeps actor n=1; GRPO group = env.rollout.n
#   - VLLM_ATTENTION_BACKEND=FLASH_ATTN        : flash_attn is installed; xformers is NOT
#   - *.log_prob_micro_batch_size_per_gpu      : required (verl errors if neither *_size nor *_per_gpu set)
#   - bare overrides on verl's default ppo_trainer config (NO --config-name; the base has env/gigpo)
#   - +env.balatro.* (the '+' adds our custom subtree, accepted by Hydra)
#
# Defaults below are the SMOKE config (tiny model, fast). To scale to the real run, override on the
# CLI (or edit): MODEL=Qwen/Qwen3-8B LORA=32 TRAIN_BS=16 GROUP=8 MAX_STEPS=350 REQ_START=0.1 EPOCHS=150
set -x
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}   # real run: Qwen/Qwen3-8B
LORA=${LORA:-16}                             # real run: 32
TRAIN_BS=${TRAIN_BS:-2}                       # task-groups/step; real run: 16
VAL_BS=${VAL_BS:-2}
GROUP=${GROUP:-2}                             # GRPO group size (env.rollout.n); real run: 8
MAX_STEPS=${MAX_STEPS:-20}                     # per-episode cap; real run: ~350 (a full game)
REQ_START=${REQ_START:-0.05}                  # curriculum floor (low -> reward variance); real run: 0.1
EPOCHS=${EPOCHS:-1}
GPUS=${GPUS:-1}
DATA=${DATA:-$HOME/data/balatro}

cd ~/verl-agent
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  algorithm.gamma=0.99 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_std_norm \
  algorithm.use_kl_in_reward=False \
  data.train_files=$DATA/train.parquet \
  data.val_files=$DATA/val.parquet \
  data.train_batch_size=$TRAIN_BS \
  data.val_batch_size=$VAL_BS \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.lora_rank=$LORA \
  actor_rollout_ref.model.lora_alpha=$LORA \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BS \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  env.env_name=balatro \
  env.seed=0 \
  env.max_steps=$MAX_STEPS \
  env.rollout.n=$GROUP \
  env.resources_per_worker.num_cpus=0.1 \
  +env.balatro.reward_name=shaped \
  +env.balatro.enable_bosses=False \
  +env.balatro.req_scale_start=$REQ_START \
  +env.balatro.req_scale_end=1.0 \
  trainer.critic_warmup=0 \
  trainer.logger=[console] \
  trainer.project_name=balatro_e6_m2 \
  trainer.experiment_name=gigpo_lora \
  trainer.n_gpus_per_node=$GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=$EPOCHS \
  trainer.val_before_train=False "$@"

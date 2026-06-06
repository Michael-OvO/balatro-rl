"""ExperimentConfig: the knob surface for M2 multi-turn GRPO/GiGPO, emitted as verl-agent Hydra
overrides. The override set is the recipe VALIDATED on the pod (2026-06-06): vLLM + GiGPO + LoRA
rolled out on the Balatro env and completed a train step. See scripts/balatro_grpo.sh (the exact
launch) and docs/RUNPOD_M2.md.

Hard-won, load-bearing details (each fixed a real launch failure — see scripts/balatro_grpo.sh):
  - actor_rollout_ref.rollout.n == 1 ALWAYS; the GRPO group size lives on env.rollout.n (verl+env).
  - launch is bare overrides on verl's default ppo_trainer config (NO --config-name); the base
    config already carries the env/gigpo/algorithm sections.
  - *.log_prob_micro_batch_size_per_gpu are required (verl errors if unset).
  - vLLM attention backend = FLASH_ATTN (set via env var by the launcher; xformers is not installed).

Kept verl-agnostic + pure so it is unit-tested on a CPU box.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    # --- model / LoRA ---
    model_path: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    lora_alpha: int = 16
    # --- algorithm (GRPO family) ---
    adv_estimator: str = "gigpo"        # multi-turn step+episode advantages; "grpo"/"rloo" also valid
    gamma: float = 0.99
    gigpo_step_advantage_w: float = 1.0
    gigpo_mode: str = "mean_std_norm"
    kl_loss_coef: float = 0.01
    group_size: int = 8                 # GRPO group size == env.rollout.n (actor rollout.n stays 1)
    actor_lr: float = 1e-6
    invalid_action_penalty_coef: float = 0.1
    # --- rollout (vLLM) ---
    temperature: float = 1.0
    max_prompt_length: int = 4096
    max_response_length: int = 512      # brief reasoning + the JSON action
    enable_thinking: bool = False       # Qwen3 <think> off so the action fits in max_response_length
    gpu_memory_utilization: float = 0.6
    tensor_model_parallel_size: int = 1
    micro_batch_size_per_gpu: int = 2   # actor / rollout / ref *_micro_batch_size_per_gpu
    # --- env / curriculum ---
    env_name: str = "balatro"
    reward_name: str = "shaped"         # dense per-turn signal (pairs with gigpo step-advantages)
    enable_bosses: bool = False
    max_steps: int = 350                # a full ~300-decision run + headroom
    seed: int = 0
    req_scale_start: float = 0.1        # curriculum: trivial blinds first -> reward variance
    req_scale_end: float = 1.0
    # --- data / trainer ---
    train_files: str = "~/data/balatro/train.parquet"
    val_files: str = "~/data/balatro/val.parquet"
    train_batch_size: int = 16          # task-groups per step
    val_batch_size: int = 8
    n_gpus_per_node: int = 1
    nnodes: int = 1
    total_epochs: int = 150
    save_freq: int = 20
    project_name: str = "balatro-e6-m2"
    experiment_name: str = "gigpo_qwen3_8b_lora"

    def to_overrides(self) -> list[str]:
        """Hydra override strings for `python -m verl.trainer.main_ppo` (bare overrides on verl's
        default ppo_trainer config — NOT a custom --config-name). Launch with
        VLLM_ATTENTION_BACKEND=FLASH_ATTN (see scripts/balatro_grpo.sh)."""
        mb = self.micro_batch_size_per_gpu
        return [
            f"algorithm.adv_estimator={self.adv_estimator}",
            f"algorithm.gamma={self.gamma}",
            f"algorithm.gigpo.step_advantage_w={self.gigpo_step_advantage_w}",
            f"algorithm.gigpo.mode={self.gigpo_mode}",
            "algorithm.use_kl_in_reward=False",
            f"data.train_files={self.train_files}",
            f"data.val_files={self.val_files}",
            f"data.train_batch_size={self.train_batch_size}",
            f"data.val_batch_size={self.val_batch_size}",
            f"data.max_prompt_length={self.max_prompt_length}",
            f"data.max_response_length={self.max_response_length}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            "data.return_raw_chat=True",
            f"+data.apply_chat_template_kwargs.enable_thinking={self.enable_thinking}",
            f"actor_rollout_ref.model.path={self.model_path}",
            f"actor_rollout_ref.model.lora_rank={self.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={self.lora_alpha}",
            "actor_rollout_ref.model.target_modules=all-linear",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.actor.optim.lr={self.actor_lr}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={self.train_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={mb}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "actor_rollout_ref.actor.use_invalid_action_penalty=True",
            f"actor_rollout_ref.actor.invalid_action_penalty_coef={self.invalid_action_penalty_coef}",
            "actor_rollout_ref.actor.fsdp_config.param_offload=False",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.tensor_model_parallel_size}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            "actor_rollout_ref.rollout.n=1",       # verl+env: actor n=1; GRPO group is env.rollout.n
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={mb}",
            "actor_rollout_ref.rollout.enforce_eager=False",
            "actor_rollout_ref.rollout.free_cache_engine=False",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={mb}",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"env.env_name={self.env_name}",
            f"env.seed={self.seed}",
            f"env.max_steps={self.max_steps}",
            f"env.rollout.n={self.group_size}",    # the GRPO group size
            "env.resources_per_worker.num_cpus=0.1",
            f"+env.balatro.reward_name={self.reward_name}",
            f"+env.balatro.enable_bosses={self.enable_bosses}",
            f"+env.balatro.req_scale_start={self.req_scale_start}",
            f"+env.balatro.req_scale_end={self.req_scale_end}",
            "trainer.critic_warmup=0",
            f"trainer.n_gpus_per_node={self.n_gpus_per_node}",
            f"trainer.nnodes={self.nnodes}",
            f"trainer.save_freq={self.save_freq}",
            f"trainer.total_epochs={self.total_epochs}",
            "trainer.val_before_train=False",
            f"trainer.project_name={self.project_name}",
            f"trainer.experiment_name={self.experiment_name}",
        ]

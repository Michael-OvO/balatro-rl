"""ExperimentConfig: the minimal knob surface for M2 multi-turn GRPO, emitted as verl-agent
Hydra overrides. Defaults reproduce the M2 spec (gigpo + LoRA + curriculum + Balatro env).

Kept verl-AGNOSTIC and pure (no verl import) so it is unit-tested on a CPU box; train_grpo.py
(pod-only) feeds to_overrides() into verl-agent's Hydra `compose`. The group size is shared by
the rollout and the env (verl-agent requires actor_rollout_ref.rollout.n == env.rollout.n).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    # --- model / LoRA ---
    model_path: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32                 # >0 enables LoRA
    lora_alpha: int = 16
    # --- algorithm (GRPO family) ---
    adv_estimator: str = "gigpo"        # multi-turn step+episode advantages; "grpo"/"rloo" also valid
    gamma: float = 0.99
    kl_loss_coef: float = 0.01
    group_size: int = 8                 # rollouts per seed/group (rollout.n == env.rollout.n)
    actor_lr: float = 1e-6
    # --- rollout ---
    rollout_backend: str = "vllm"       # "vllm" | "sglang"
    temperature: float = 1.0
    max_prompt_length: int = 4096
    max_response_length: int = 512      # brief reasoning + the JSON action
    enable_thinking: bool = False       # Qwen3 <think> off so the action fits in max_response_length
    gpu_memory_utilization: float = 0.6
    # --- env / curriculum ---
    env_name: str = "balatro"
    reward_name: str = "shaped"         # dense per-turn signal (pairs with gigpo step-advantages)
    max_steps: int = 350                # a full ~300-decision run + headroom
    history_length: int = 8             # turns of raw history kept (matches the M1 context window)
    req_scale_start: float = 0.1        # curriculum: trivial blinds first -> reward variance
    req_scale_end: float = 1.0
    # --- trainer ---
    train_batch_size: int = 16          # task-groups per step (x group_size rollouts each)
    val_batch_size: int = 8             # task-groups for validation (group_n=1)
    total_epochs: int = 150
    save_freq: int = 20
    project_name: str = "balatro-e6-m2"
    experiment_name: str = "gigpo_qwen3_8b_lora"

    def to_overrides(self) -> list[str]:
        """Hydra override strings for verl-agent's ppo_trainer config."""
        return [
            f"algorithm.adv_estimator={self.adv_estimator}",
            f"algorithm.gamma={self.gamma}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={self.kl_loss_coef}",
            f"actor_rollout_ref.actor.optim.lr={self.actor_lr}",
            f"actor_rollout_ref.model.path={self.model_path}",
            f"actor_rollout_ref.model.lora_rank={self.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={self.lora_alpha}",
            "actor_rollout_ref.model.target_modules=all-linear",
            f"actor_rollout_ref.rollout.name={self.rollout_backend}",
            f"actor_rollout_ref.rollout.n={self.group_size}",
            f"actor_rollout_ref.rollout.temperature={self.temperature}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.gpu_memory_utilization}",
            f"+data.apply_chat_template_kwargs.enable_thinking={self.enable_thinking}",
            f"data.train_batch_size={self.train_batch_size}",
            f"data.val_batch_size={self.val_batch_size}",
            f"data.max_prompt_length={self.max_prompt_length}",
            f"data.max_response_length={self.max_response_length}",
            "data.return_raw_chat=True",
            f"env.env_name={self.env_name}",
            f"env.max_steps={self.max_steps}",
            f"env.history_length={self.history_length}",
            f"env.rollout.n={self.group_size}",
            f"+env.balatro.reward_name={self.reward_name}",
            f"+env.balatro.req_scale_start={self.req_scale_start}",
            f"+env.balatro.req_scale_end={self.req_scale_end}",
            f"trainer.total_epochs={self.total_epochs}",
            f"trainer.save_freq={self.save_freq}",
            f"trainer.project_name={self.project_name}",
            f"trainer.experiment_name={self.experiment_name}",
        ]

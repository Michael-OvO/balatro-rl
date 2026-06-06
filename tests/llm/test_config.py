from balatro_rl.llm.config import ExperimentConfig


def test_defaults_reproduce_the_m2_spec():
    cfg = ExperimentConfig()
    ov = cfg.to_overrides()
    joined = "\n".join(ov)
    assert "algorithm.adv_estimator=gigpo" in joined
    assert "actor_rollout_ref.model.lora_rank=32" in joined
    assert "actor_rollout_ref.model.path=Qwen/Qwen3-8B" in joined
    assert "+data.apply_chat_template_kwargs.enable_thinking=False" in joined
    assert "env.env_name=balatro" in joined
    assert "+env.balatro.req_scale_start=0.1" in joined


def test_group_size_goes_on_env_rollout_not_actor():
    # verl+env (validated on the pod): actor_rollout_ref.rollout.n is ALWAYS 1; the GRPO group
    # size lives on env.rollout.n. (Setting actor rollout.n>1 trips a verl assertion.)
    cfg = ExperimentConfig(group_size=16)
    ov = cfg.to_overrides()
    assert "actor_rollout_ref.rollout.n=1" in ov
    assert "env.rollout.n=16" in ov


def test_emits_validated_required_fields():
    # Fields the smoke test proved are required (verl errors without the micro-batch ones; the
    # penalty is what the is_action_valid plumbing feeds).
    joined = "\n".join(ExperimentConfig().to_overrides())
    for key in ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=",
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=",
                "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=",
                "actor_rollout_ref.actor.use_invalid_action_penalty=True",
                "algorithm.gigpo.step_advantage_w=", "data.train_files="):
        assert key in joined, key


def test_overrides_are_flat_strings():
    ov = ExperimentConfig().to_overrides()
    assert ov and all(isinstance(o, str) and "=" in o for o in ov)


def test_grpo_estimator_selectable():
    assert "algorithm.adv_estimator=grpo" in ExperimentConfig(adv_estimator="grpo").to_overrides()

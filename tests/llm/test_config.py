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


def test_group_size_is_consistent_across_rollout_and_env():
    # verl-agent requires actor_rollout_ref.rollout.n == env.rollout.n (the GRPO group size).
    cfg = ExperimentConfig(group_size=16)
    ov = cfg.to_overrides()
    assert "actor_rollout_ref.rollout.n=16" in ov
    assert "env.rollout.n=16" in ov


def test_overrides_are_flat_strings():
    ov = ExperimentConfig().to_overrides()
    assert ov and all(isinstance(o, str) and "=" in o for o in ov)


def test_grpo_estimator_selectable():
    assert "algorithm.adv_estimator=grpo" in ExperimentConfig(adv_estimator="grpo").to_overrides()

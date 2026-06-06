from balatro_rl.llm.config import ExperimentConfig
from balatro_rl.llm.train_grpo import build_command


def test_build_command_wraps_overrides_for_verl_agent():
    cmd = build_command(ExperimentConfig())
    assert cmd[:3] == ["python", "-m", "verl.trainer.main_ppo"]
    joined = " ".join(cmd)
    assert "algorithm.adv_estimator=gigpo" in joined
    assert "actor_rollout_ref.model.path=Qwen/Qwen3-8B" in joined
    assert "env.env_name=balatro" in joined


def test_build_command_is_bare_overrides_not_config_name():
    # Validated on the pod: bare overrides on verl's default ppo_trainer config (NO --config-name;
    # the base already has env/gigpo). It must enable the invalid-action penalty + the required
    # micro-batch fields, which the earlier --config-name form silently dropped.
    joined = " ".join(build_command(ExperimentConfig()))
    assert "--config-name" not in joined and "--config-path" not in joined
    assert "actor_rollout_ref.actor.use_invalid_action_penalty=True" in joined
    assert "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=" in joined


def test_build_command_reflects_config_overrides():
    cmd = build_command(ExperimentConfig(adv_estimator="grpo", group_size=4))
    joined = " ".join(cmd)
    assert "algorithm.adv_estimator=grpo" in joined
    # verl+env: actor rollout.n is ALWAYS 1; the GRPO group size is env.rollout.n.
    assert "actor_rollout_ref.rollout.n=1" in joined and "env.rollout.n=4" in joined

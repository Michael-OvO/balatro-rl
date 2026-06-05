from balatro_rl.llm.config import ExperimentConfig
from balatro_rl.llm.train_grpo import build_command


def test_build_command_wraps_overrides_for_verl_agent():
    cmd = build_command(ExperimentConfig())
    assert cmd[:3] == ["python", "-m", "verl.trainer.main_ppo"]
    joined = " ".join(cmd)
    assert "algorithm.adv_estimator=gigpo" in joined
    assert "actor_rollout_ref.model.path=Qwen/Qwen3-8B" in joined
    assert "env.env_name=balatro" in joined


def test_build_command_reflects_config_overrides():
    cmd = build_command(ExperimentConfig(adv_estimator="grpo", group_size=4))
    joined = " ".join(cmd)
    assert "algorithm.adv_estimator=grpo" in joined
    assert "actor_rollout_ref.rollout.n=4" in joined and "env.rollout.n=4" in joined

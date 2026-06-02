import numpy as np
from balatro_rl.agent.run import run_training
from balatro_rl.agent.train import TrainConfig
from balatro_rl.agent.metrics_logger import NullLogger


def test_run_training_with_null_logger():
    cfg = TrainConfig(num_envs=4, num_steps=16, num_updates=2, d_model=32,
                      num_minibatches=2, update_epochs=1, reward_name="max_depth", seed=0)
    cfg.eval_interval = 1
    cfg.eval_seeds = [0, 1]
    logger = NullLogger()
    result = run_training(cfg, logger=logger)
    assert len(result.losses) == 2
    assert len(result.eval_history) == 2
    assert any("eval/win_rate" in d for _s, d in logger.history)

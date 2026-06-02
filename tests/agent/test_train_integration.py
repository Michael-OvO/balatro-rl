import numpy as np
from balatro_rl.agent.train import train, TrainConfig


def test_short_training_run_is_stable():
    # A slightly larger short run: must complete, stay finite, and not NaN out.
    cfg = TrainConfig(num_envs=8, num_steps=32, num_updates=5, d_model=32,
                      num_minibatches=2, update_epochs=2, reward_name="shaped", seed=3)
    res = train(cfg)
    assert len(res.losses) == 5
    totals = [t for (t, *_rest) in res.losses]
    assert all(np.isfinite(t) for t in totals)
    assert all(np.isfinite(r) for r in res.mean_returns)

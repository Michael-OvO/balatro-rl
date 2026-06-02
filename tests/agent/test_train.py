import numpy as np
from balatro_rl.agent.train import train, TrainConfig


def _tiny_cfg(**over):
    cfg = TrainConfig(num_envs=4, num_steps=16, num_updates=3, d_model=32,
                      num_minibatches=2, update_epochs=2, reward_name="max_depth", seed=0)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_train_runs_and_losses_finite():
    result = train(_tiny_cfg())
    assert len(result.losses) == 3                       # one per update
    for (total, pg, vl, ent) in result.losses:
        assert np.isfinite(total) and np.isfinite(pg) and np.isfinite(vl) and np.isfinite(ent)


def test_train_params_have_no_nans():
    result = train(_tiny_cfg())
    import jax
    leaves = jax.tree_util.tree_leaves(result.params)
    assert all(np.all(np.isfinite(np.asarray(l))) for l in leaves)


def test_train_deterministic_for_seed():
    a = train(_tiny_cfg(seed=7))
    b = train(_tiny_cfg(seed=7))
    assert np.allclose(a.losses[0][0], b.losses[0][0], atol=1e-4)   # same first-update total loss


def test_train_tracks_episode_returns():
    result = train(_tiny_cfg(num_updates=4))
    assert len(result.mean_returns) == 4                  # a finite scalar per update
    assert all(np.isfinite(r) for r in result.mean_returns)

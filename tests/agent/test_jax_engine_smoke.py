"""Smoke test: JAX engine wired into PPO train loop (Task 1.9).

Runs two PPO updates with JaxVectorEnv (engine="jax") and asserts:
- train() completes without error and returns a TrainResult with params.
- All losses are finite (no NaN/inf).
- The correct number of updates ran.
- Param leaves are all finite arrays.

The python-path regression is covered by the existing tests/agent/test_train.py.
"""
import numpy as np
import jax
import pytest

from balatro_rl.agent.train import train, TrainConfig


def _jax_smoke_cfg(**overrides):
    cfg = TrainConfig(
        engine="jax",
        num_envs=16,
        num_steps=16,
        num_updates=2,
        d_model=32,
        num_minibatches=2,
        update_epochs=2,
        reward_name="shaped",   # JaxVectorEnv only supports "shaped"
        enable_bosses=False,    # JaxVectorEnv raises if True
        eval_interval=0,        # disable eval (keeps test fast)
        seed=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_jax_engine_train_runs():
    """train() with engine='jax' completes and returns a TrainResult with params."""
    result = train(_jax_smoke_cfg())
    assert result.params is not None, "train() must return params"


def test_jax_engine_losses_finite():
    """All PPO losses are finite (not NaN / inf) after two updates."""
    result = train(_jax_smoke_cfg())
    assert len(result.losses) == 2, f"expected 2 updates, got {len(result.losses)}"
    for i, (total, pg, vl, ent) in enumerate(result.losses):
        assert np.isfinite(total), f"update {i}: total loss is not finite ({total})"
        assert np.isfinite(pg),    f"update {i}: pg loss is not finite ({pg})"
        assert np.isfinite(vl),    f"update {i}: vl loss is not finite ({vl})"
        assert np.isfinite(ent),   f"update {i}: ent loss is not finite ({ent})"


def test_jax_engine_params_finite():
    """All parameter leaves are finite arrays (no NaN / inf from JAX env steps)."""
    result = train(_jax_smoke_cfg())
    leaves = jax.tree_util.tree_leaves(result.params)
    assert len(leaves) > 0, "params tree must be non-empty"
    for leaf in leaves:
        arr = np.asarray(leaf)
        assert np.all(np.isfinite(arr)), f"param leaf has non-finite values: {arr}"


def test_jax_engine_mean_returns_finite():
    """Mean per-step reward is recorded and finite for every update."""
    result = train(_jax_smoke_cfg())
    assert len(result.mean_returns) == 2
    for r in result.mean_returns:
        assert np.isfinite(r), f"mean_return is not finite: {r}"


def test_jax_engine_enable_bosses_raises():
    """Confirm that engine='jax' with enable_bosses=True raises ValueError."""
    with pytest.raises(ValueError, match="enable_bosses"):
        train(_jax_smoke_cfg(enable_bosses=True))


def test_ppo_smoke_with_jokers():
    """PPO trains end-to-end with a fixed joker loadout (Joker + The Duo).

    TrainConfig is constructed directly (not via _jax_smoke_cfg's setattr) so a
    missing `joker_loadout` field fails loudly with TypeError (Task 2.9 TDD).
    """
    cfg = TrainConfig(
        engine="jax",
        num_envs=16,
        num_steps=16,
        num_updates=2,
        d_model=32,
        num_minibatches=2,
        update_epochs=2,
        reward_name="shaped",
        enable_bosses=False,
        eval_interval=0,
        seed=0,
        joker_loadout=[1, 131],   # Joker + The Duo for every env
    )
    result = train(cfg)
    assert result.params is not None, "train() must return params"
    assert len(result.losses) == 2, f"expected 2 updates, got {len(result.losses)}"
    for i, (total, pg, vl, ent) in enumerate(result.losses):
        assert np.isfinite(total), f"update {i}: total loss is not finite ({total})"
        assert np.isfinite(pg),    f"update {i}: pg loss is not finite ({pg})"
        assert np.isfinite(vl),    f"update {i}: vl loss is not finite ({vl})"
        assert np.isfinite(ent),   f"update {i}: ent loss is not finite ({ent})"


def test_jax_vec_env_joker_loadout_in_obs():
    """JaxVectorEnv with a loadout exposes it via the joker obs (non-zero types)."""
    from balatro_rl.envs.jax_vec_env import JaxVectorEnv

    venv = JaxVectorEnv(16, reward_name="shaped", base_seed=0,
                        joker_loadout=[1, 131])
    obs, masks = venv.reset()
    jt = np.asarray(obs["joker_types"])          # [N, MAX_JOKERS]
    assert jt.shape[0] == 16
    assert np.all(jt[:, 0] == 1),   "slot 0 must hold Joker (id 1) in every env"
    assert np.all(jt[:, 1] == 131), "slot 1 must hold The Duo (id 131) in every env"
    assert np.all(jt[:, 2:] == 0),  "remaining slots must be 0-padded"


def test_jax_vec_env_loadout_too_long_raises():
    """A loadout longer than MAX_JOKERS slots is rejected at construction."""
    from balatro_rl.envs.jax_vec_env import JaxVectorEnv
    from balatro_rl.envs.actions import MAX_JOKERS

    with pytest.raises(ValueError, match="slots"):
        JaxVectorEnv(4, reward_name="shaped", base_seed=0,
                     joker_loadout=list(range(1, MAX_JOKERS + 2)))

"""Tests for JaxVectorEnv — SyncVectorEnv-compatible JAX-native vectorized env.

Task 1.8 requirements:
  1. reset() returns (obs_dict, masks) with shapes matching SyncVectorEnv(N).
  2. The obs dict KEYS match SyncVectorEnv exactly.
  3. A 5-step random-legal rollout runs without error; returns correct shapes.
  4. Done envs auto-reset (the next obs is a fresh episode).
  5. set_req_scale() updates the in-state required_table.
  6. set_boss_rate() is a no-op and emits a warning.
  7. enable_bosses=True raises ValueError.
"""
from __future__ import annotations

import warnings
import numpy as np
import jax.numpy as jnp
import pytest

from balatro_rl.envs.jax_vec_env import JaxVectorEnv
from balatro_rl.envs.vec_env import SyncVectorEnv
from balatro_rl.envs.actions import NUM_ACTIONS
from balatro_rl.envs.obs import OBS_SHAPES


N = 8   # number of envs for all tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_legal_actions(mask: np.ndarray) -> np.ndarray:
    """For each env, pick the first legal action from the mask."""
    n = mask.shape[0]
    actions = np.zeros(n, dtype=np.int32)
    for i in range(n):
        legal = np.flatnonzero(mask[i])
        assert len(legal) > 0, f"Env {i} has no legal actions"
        actions[i] = int(legal[0])
    return actions


# ---------------------------------------------------------------------------
# 1 & 2. reset() shapes and obs keys match SyncVectorEnv
# ---------------------------------------------------------------------------

def test_reset_shapes_match_sync():
    """reset() returns (obs, masks) with same shapes and keys as SyncVectorEnv(N)."""
    jenv = JaxVectorEnv(num_envs=N)
    senv = SyncVectorEnv(num_envs=N)

    j_obs, j_mask = jenv.reset()
    s_obs, s_mask = senv.reset()

    # Mask shape and dtype
    assert j_mask.shape == (N, NUM_ACTIONS), (
        f"mask shape: got {j_mask.shape}, expected {(N, NUM_ACTIONS)}"
    )
    assert j_mask.dtype == bool or j_mask.dtype == np.bool_, (
        f"mask dtype: {j_mask.dtype}"
    )

    # Obs keys must match exactly
    assert set(j_obs.keys()) == set(s_obs.keys()), (
        f"obs key mismatch:\n  JaxVec={sorted(j_obs.keys())}\n  SyncVec={sorted(s_obs.keys())}"
    )

    # Obs shapes per key must match (leading dim N + per-obs shape from OBS_SHAPES)
    for k, per_env_shape in OBS_SHAPES.items():
        expected_shape = (N,) + per_env_shape
        j_shape = j_obs[k].shape
        s_shape = s_obs[k].shape
        assert j_shape == expected_shape, (
            f"obs['{k}'] shape: JaxVec={j_shape}, expected {expected_shape}"
        )
        assert j_shape == s_shape, (
            f"obs['{k}'] shape mismatch: JaxVec={j_shape}, SyncVec={s_shape}"
        )


def test_reset_masks_have_legal_actions():
    """Every env has at least one legal action after reset."""
    jenv = JaxVectorEnv(num_envs=N)
    _, mask = jenv.reset()
    mask_np = np.asarray(mask, dtype=bool)
    assert mask_np.any(axis=1).all(), (
        "Some envs have no legal actions after reset"
    )


# ---------------------------------------------------------------------------
# 3. 5-step random-legal rollout — shapes and no errors
# ---------------------------------------------------------------------------

def test_rollout_5_steps():
    """A 5-step rollout picking random legal actions runs without error."""
    jenv = JaxVectorEnv(num_envs=N)
    obs, mask = jenv.reset()

    for step_i in range(5):
        mask_np = np.asarray(mask, dtype=bool)
        actions = _pick_legal_actions(mask_np)
        obs, rewards, dones, infos, mask = jenv.step(actions)

        # Shapes
        assert rewards.shape == (N,), f"step {step_i}: rewards shape {rewards.shape}"
        assert rewards.dtype == np.float32, f"rewards dtype {rewards.dtype}"
        assert dones.shape == (N,), f"step {step_i}: dones shape {dones.shape}"

        mask_np = np.asarray(mask, dtype=bool)
        assert mask_np.shape == (N, NUM_ACTIONS), f"mask shape {mask_np.shape}"

        # Obs shapes
        for k, per_env_shape in OBS_SHAPES.items():
            expected = (N,) + per_env_shape
            assert obs[k].shape == expected, (
                f"step {step_i}, obs['{k}']: got {obs[k].shape}, expected {expected}"
            )

        # Infos list length and keys
        assert len(infos) == N, f"len(infos)={len(infos)}, expected {N}"
        for i, info in enumerate(infos):
            assert "cleared" in info, f"step {step_i}, env {i}: 'cleared' missing from info"
            assert "ante" in info, f"step {step_i}, env {i}: 'ante' missing from info"
            assert "score" in info, f"step {step_i}, env {i}: 'score' missing from info"
            assert "round_score" in info, f"step {step_i}, env {i}: 'round_score' missing"

        # After a step, every env still has legal actions (auto-reset ensures this)
        assert mask_np.any(axis=1).all(), (
            f"step {step_i}: some env has no legal actions (auto-reset should fix this)"
        )


# ---------------------------------------------------------------------------
# 4. Done envs auto-reset to fresh episode
# ---------------------------------------------------------------------------

def test_done_envs_autoreset():
    """After a done, the returned obs is a fresh episode (not garbage)."""
    # Drive envs until at least one done occurs, then check that obs is fresh.
    jenv = JaxVectorEnv(num_envs=N, base_seed=1)
    obs, mask = jenv.reset()

    found_done = False
    for _ in range(500):  # run up to 500 steps; Balatro games end within ~200 plays
        mask_np = np.asarray(mask, dtype=bool)
        actions = _pick_legal_actions(mask_np)
        obs, rewards, dones, infos, mask = jenv.step(actions)
        dones_np = np.asarray(dones, dtype=bool)
        mask_np = np.asarray(mask, dtype=bool)

        if dones_np.any():
            found_done = True
            # After auto-reset, every env (including done ones) must have legal actions.
            assert mask_np.any(axis=1).all(), (
                "A done env was not auto-reset — no legal actions available"
            )
            # The obs for a done env should correspond to a fresh episode:
            # the global feature vector's hands_left entry (index 3) should be
            # HANDS_PER_BLIND (4), not 0.  We check any done env.
            from balatro_rl.engine_jax.config import HANDS_PER_BLIND
            global_obs = np.asarray(obs["global"])   # [N, 24]
            for i in np.where(dones_np)[0]:
                hands_left_feat = float(global_obs[i, 3])
                assert hands_left_feat == float(HANDS_PER_BLIND), (
                    f"Done env {i} did not auto-reset: "
                    f"global[3] (hands_left) = {hands_left_feat}, "
                    f"expected {float(HANDS_PER_BLIND)}"
                )
            break

    assert found_done, (
        "No done env was observed in 500 steps — increase step budget or check step logic"
    )


# ---------------------------------------------------------------------------
# 5. set_req_scale() patches in-state required_table
# ---------------------------------------------------------------------------

def test_set_req_scale():
    """set_req_scale rebuilds the required_table and patches in-state envs."""
    from balatro_rl.engine_jax.curriculum import build_required_table

    jenv = JaxVectorEnv(num_envs=N, req_scale=1.0)
    jenv.reset()

    old_table = np.asarray(jenv.required_table)

    jenv.set_req_scale(0.5)
    new_table = np.asarray(jenv.required_table)

    expected_table = build_required_table(0.5)

    assert not np.array_equal(old_table, new_table), (
        "required_table did not change after set_req_scale(0.5)"
    )
    assert np.array_equal(new_table, expected_table), (
        "required_table does not match build_required_table(0.5)"
    )

    # The in-state required_table must also be patched.
    state_tables = np.asarray(jenv.state.required_table)  # [N, 9, 3]
    for i in range(N):
        assert np.array_equal(state_tables[i], expected_table), (
            f"Env {i}: in-state required_table not updated by set_req_scale"
        )


# ---------------------------------------------------------------------------
# 6. set_boss_rate() is a no-op and warns
# ---------------------------------------------------------------------------

def test_set_boss_rate_warns():
    """set_boss_rate(0.5) emits a warning and is a no-op."""
    jenv = JaxVectorEnv(num_envs=N)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jenv.set_boss_rate(0.5)
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert "no-op" in str(w[0].message).lower() or "boss" in str(w[0].message).lower()

    # Second call should NOT warn again (warned once only)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jenv.set_boss_rate(0.8)
        assert len(w) == 0, "Expected no second warning"


def test_set_boss_rate_zero_no_warn():
    """set_boss_rate(0.0) should not warn (rate=0 = bosses off = expected)."""
    jenv = JaxVectorEnv(num_envs=N)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jenv.set_boss_rate(0.0)
        assert len(w) == 0, f"Unexpected warning for rate=0: {w}"


# ---------------------------------------------------------------------------
# 7. enable_bosses=True raises ValueError
# ---------------------------------------------------------------------------

def test_enable_bosses_raises():
    """enable_bosses=True must raise ValueError."""
    with pytest.raises(ValueError, match="enable_bosses"):
        JaxVectorEnv(num_envs=N, enable_bosses=True)


# ---------------------------------------------------------------------------
# 8. Infos keys match what SyncVectorEnv produces (intersection check)
# ---------------------------------------------------------------------------

def test_infos_keys_superset_of_sync():
    """JaxVectorEnv infos contain at least the keys that train.py reads."""
    jenv = JaxVectorEnv(num_envs=N)
    _, mask = jenv.reset()
    mask_np = np.asarray(mask, dtype=bool)
    actions = _pick_legal_actions(mask_np)
    _, _, _, infos, _ = jenv.step(actions)

    required_keys = {"cleared", "ante", "score", "round_score"}
    for i, info in enumerate(infos):
        missing = required_keys - set(info.keys())
        assert not missing, f"Env {i} info missing keys: {missing}"


# ---------------------------------------------------------------------------
# 9. Interface parity with SyncVectorEnv (obs keys + mask shape)
# ---------------------------------------------------------------------------

def test_obs_keys_match_sync_vec():
    """Obs dict keys from JaxVectorEnv exactly match SyncVectorEnv."""
    jenv = JaxVectorEnv(num_envs=N)
    senv = SyncVectorEnv(num_envs=N)

    j_obs, _ = jenv.reset()
    s_obs, _ = senv.reset()

    assert set(j_obs.keys()) == set(s_obs.keys()), (
        f"Key mismatch:\n  extra in JaxVec: {set(j_obs.keys()) - set(s_obs.keys())}\n"
        f"  missing from JaxVec: {set(s_obs.keys()) - set(j_obs.keys())}"
    )

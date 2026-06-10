"""Parity test: encode_core + legal_mask_core vs Python envs/obs.py + envs/actions.py.

Drives both the Python oracle and the JAX engine with the same scripted action
(lowest-id PLAY or DISCARD), asserting at every PLAYING step:

  (a) encode_core(cs)[key] matches encode(gs)[key] for all keys in OBS_SHAPES,
      using allclose (atol=1e-5) for float32 arrays and exact equality for int32.

  (b) legal_mask_core(cs)[:436] matches legal_mask(gs)[:436] exactly.

Coverage:
  - 40 seeds × up to 30 steps each with the play-policy.
  - 40 seeds × up to 30 steps each with the discard-then-play policy.
  - Both PLAYING mid-blind and LOST terminal states are compared.
  - Blind-clear transitions (SHOP) are skipped like in test_step_parity.

Keys excluded from float comparison (zeroed in core, not zero in Python):
  - None: deck_rank_hist and deck_suit_hist ARE now computed from deck_ptr,
    so all keys are compared.

Keys that are always zero in Python (during a core play-only run, no shop):
  - joker_*, shop_*, boss_onehot, consum_*, pack_*, voucher_*, pending_consum.
  - These match trivially.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from balatro_rl.engine import engine
from balatro_rl.engine.state import Phase
from balatro_rl.envs.obs import OBS_SHAPES, encode
from balatro_rl.envs.actions import decode, legal_mask, PLAY_N, NUM_ACTIONS
from balatro_rl.engine_jax import step as J
from balatro_rl.engine_jax.config import MAX_HAND
from balatro_rl.engine_jax.obs import encode_core, legal_mask_core
from tests.engine_jax.parity_util import deck_from_python

# step now folds the full joker pipeline (Task 2.6), so an UNCOMPILED trace costs
# ~0.5s — jit once at module level so the per-step loop below runs in ~ms.
_JIT_STEP = jax.jit(J.step)

N_SEEDS = 40
MAX_STEPS = 30
SCALE = 0.2

# Keys compared with exact int equality (integer arrays)
_INT_KEYS = {
    "joker_types", "shop_types", "shop_consum", "consum_types",
    "pack_kind", "pack_size", "pack_item_joker", "pack_item_consum",
    "voucher_offer", "pending_consum",
}

# Keys compared with allclose (float32 arrays)
_FLOAT_KEYS = set(OBS_SHAPES.keys()) - _INT_KEYS

# Keys to SKIP in the comparison (cannot be faithfully computed in core OR
# intentionally diverge). Currently none — all keys are compared.
_SKIP_KEYS: set[str] = set()


def _sel_mask(idxs) -> np.ndarray:
    m = np.zeros(MAX_HAND, dtype=bool)
    for i in idxs:
        m[i] = True
    return m


def _lowest_play_id(mask) -> int | None:
    ids = np.nonzero(mask[:PLAY_N])[0]
    return int(ids[0]) if ids.size else None


def _lowest_discard_id(mask) -> int | None:
    ids = np.nonzero(mask[PLAY_N:2 * PLAY_N])[0]
    return int(ids[0] + PLAY_N) if ids.size else None


def _assert_obs_equal(cs, gs, step_info: str) -> None:
    """Assert encode_core(cs) matches encode(gs) for all OBS_SHAPES keys."""
    jax_obs = encode_core(cs)
    py_obs = encode(gs)

    assert set(jax_obs.keys()) == set(py_obs.keys()), (
        f"{step_info}: key mismatch — JAX has {set(jax_obs.keys())}, "
        f"Python has {set(py_obs.keys())}"
    )

    for key in OBS_SHAPES:
        if key in _SKIP_KEYS:
            continue

        jx_val = np.asarray(jax_obs[key])
        py_val = np.asarray(py_obs[key])

        assert jx_val.shape == py_val.shape, (
            f"{step_info}: key '{key}' shape mismatch: JAX={jx_val.shape}, Py={py_val.shape}"
        )

        if key in _INT_KEYS:
            if not np.array_equal(jx_val, py_val):
                raise AssertionError(
                    f"{step_info}: key '{key}' (int) mismatch:\n"
                    f"  JAX={jx_val}\n  Py ={py_val}"
                )
        else:
            if not np.allclose(jx_val, py_val, atol=1e-5, rtol=0.0):
                diff = np.abs(jx_val.astype(np.float64) - py_val.astype(np.float64))
                raise AssertionError(
                    f"{step_info}: key '{key}' (float) mismatch:\n"
                    f"  JAX={jx_val}\n  Py ={py_val}\n"
                    f"  max_err={diff.max():.2e} at idx {np.unravel_index(diff.argmax(), diff.shape)}"
                )


def _assert_legal_mask_equal(cs, gs, step_info: str) -> None:
    """Assert legal_mask_core(cs)[:436] == legal_mask(gs)[:436]."""
    jx_mask = np.asarray(legal_mask_core(cs), dtype=bool)
    py_mask = np.asarray(legal_mask(gs), dtype=bool)

    assert len(jx_mask) == NUM_ACTIONS, (
        f"{step_info}: JAX mask length {len(jx_mask)} != {NUM_ACTIONS}"
    )

    # PLAY range [0, 218)
    jx_play = jx_mask[:PLAY_N]
    py_play = py_mask[:PLAY_N]
    if not np.array_equal(jx_play, py_play):
        diffs = np.where(jx_play != py_play)[0]
        raise AssertionError(
            f"{step_info}: legal_mask PLAY range [0,218) mismatch at ids={diffs.tolist()}"
        )

    # DISCARD range [218, 436)
    jx_disc = jx_mask[PLAY_N:2 * PLAY_N]
    py_disc = py_mask[PLAY_N:2 * PLAY_N]
    if not np.array_equal(jx_disc, py_disc):
        diffs = np.where(jx_disc != py_disc)[0]
        raise AssertionError(
            f"{step_info}: legal_mask DISCARD range [218,436) mismatch at relative ids={diffs.tolist()}"
        )

    # Shop/pack range [436, 708) must be all-False in core.
    jx_shop = jx_mask[2 * PLAY_N:]
    if jx_shop.any():
        bad = np.where(jx_shop)[0] + 2 * PLAY_N
        raise AssertionError(
            f"{step_info}: legal_mask shop/pack range [436,708) has True at ids={bad.tolist()}"
        )


def _run_policy(pick_action_id, label: str) -> int:
    """Run scripted policy over N_SEEDS; return number of within-blind comparisons."""
    compared = 0
    for seed in range(N_SEEDS):
        gs = engine.reset(seed, SCALE, None, False)
        ranks, suits = deck_from_python(gs)
        cs = J.reset(ranks, suits, required=gs.required)

        # Compare initial state obs (before any step).
        info = f"seed={seed} step=0 (initial)"
        _assert_obs_equal(cs, gs, info)
        _assert_legal_mask_equal(cs, gs, info)
        compared += 1

        for step_i in range(1, MAX_STEPS + 1):
            mask = legal_mask(gs)
            aid = pick_action_id(gs, mask)
            if aid is None:
                break
            verb, idxs = decode(aid)
            sel = _sel_mask(idxs)

            gs2, _info = engine.step(gs, (verb, tuple(idxs)))
            cs2, sig = _JIT_STEP(cs, int(verb), sel)

            info = f"{label} seed={seed} step={step_i}"

            if gs2.phase in (Phase.SHOP, Phase.WON) or gs2.won:
                # Blind cleared: stop comparing (JAX clear half-state intentionally uncompared).
                assert bool(sig.cleared), f"{info}: JAX did not flag cleared on a Python clear"
                break
            elif gs2.phase == Phase.LOST:
                # Terminal LOST state: compare obs (Python encodes terminal states the same way).
                assert gs2.done
                _assert_obs_equal(cs2, gs2, info + " [LOST]")
                # Legal mask on a LOST state: Python legal_mask calls legal_actions which returns
                # [] on a done state, so all entries are False. JAX is_playing=False -> all False.
                _assert_legal_mask_equal(cs2, gs2, info + " [LOST]")
                compared += 1
                break
            else:
                _assert_obs_equal(cs2, gs2, info)
                _assert_legal_mask_equal(cs2, gs2, info)
                compared += 1
                gs, cs = gs2, cs2

    return compared


def test_obs_parity_play_policy():
    """Play-only policy: always take the lowest-index legal PLAY id."""
    compared = _run_policy(
        lambda gs, mask: _lowest_play_id(mask),
        label="play-policy",
    )
    assert compared > 0, "no states were compared"


def test_obs_parity_discard_policy():
    """Discard-first policy: DISCARD while discards remain, then PLAY.

    Exercises the obs at varying discards_left (3, 2, 1, 0) and the DISCARD
    legal-mask edge where discard_legal flips off when discards_left=0.
    """
    def pick(gs, mask):
        if gs.discards_left > 0:
            did = _lowest_discard_id(mask)
            if did is not None:
                return did
        return _lowest_play_id(mask)

    compared = _run_policy(pick, label="discard-policy")
    assert compared > 0, "no states were compared"


def test_obs_joker_keys_match_python():
    from balatro_rl.engine_jax.step import reset
    from balatro_rl.engine_jax.obs import encode_core
    from balatro_rl.envs.actions import MAX_JOKERS
    # Build a JAX state with a fixed loadout and check the joker obs keys.
    jk = np.zeros(MAX_JOKERS, np.int32); jk[0] = 1; jk[1] = 131  # Joker, The Duo
    ranks = [r for r in range(2,15) for _ in range(4)]; suits = [s for _ in range(2,15) for s in range(4)]
    st = reset(ranks, suits, required=300, jokers=jk)
    obs = encode_core(st)
    assert int(np.asarray(obs["joker_types"])[0]) == 1
    assert int(np.asarray(obs["joker_types"])[1]) == 131
    assert float(np.asarray(obs["joker_mask"])[0]) == 1.0 and float(np.asarray(obs["joker_mask"])[1]) == 1.0
    assert float(np.asarray(obs["joker_mask"])[2]) == 0.0
    assert float(np.asarray(obs["joker_counter"])[0]) == 0.0   # stateless -> symlog(0)=0
    assert float(np.asarray(obs["global"])[10]) == 2.0          # joker count
    assert np.all(np.asarray(obs["joker_types"])[2:] == 0)      # empty slots stay 0

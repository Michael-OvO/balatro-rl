"""Phase-1 go/no-go PARITY GATE (Task 1.10) + Phase-2 Gate B (Task 2.8).

The most thorough correctness test for the JAX core engine: drive BOTH the Python
oracle and the JAX engine in lockstep over many (seed x random-legal-action) rollouts
to termination, and assert the FULL within-blind transition agrees at every step:

  * scalars + sorted held-card multiset   (assert_states_equal)
  * ordered hand slot-by-slot             (assert_hand_slots_equal)
  * the full observation, every OBS key    (encode_core vs envs.obs.encode)
  * the shaped reward                       (shaped_core vs envs.rewards.Shaped)

Policy: a UNIFORMLY RANDOM legal action each step (seeded per rollout), which exercises
discards + varied plays far more than the scripted/greedy policies in the other tests.

Boundary handling reuses the Task-1.4 machinery (shop-skip + advance-scalar assert +
deck/hand/money re-sync); obs/reward are NOT compared across a blind boundary because
the deck reshuffle (RNG) and the economy (cashout money) are out of the JAX core's
scope — exactly as in tests 1.4/1.5/1.6. Everything WITHIN a blind is compared fully.

Any mismatch here is a real engine bug (debug against the Python oracle); never weaken
the assertion to pass.

GATE B (Task 2.8) generalizes the same harness to a FIXED JOKER LOADOUT: inject the
identical loadout into both engines at reset (Python: ``dataclasses.replace`` of the
post-reset ``jokers`` tuple; JAX: the 0-padded ``jokers=`` reset kwarg) and run the
same within-blind lockstep, additionally asserting the joker OBS keys. This proves the
Task-2.6 integration (held-card derivation, context assembly, obs, reward) end-to-end:
Gate A already proved the scoring kernel == ``score_play`` for any loadout/hand, so
any Gate-B divergence is an integration bug. The empty loadout ``[]`` runs the same
path as a Phase-1 regression. Loadouts stay in the float32-safe regime (at most ONE
Baron — see the stacked-x1.5 note in ``engine_jax/jokers.py``).

Runs a 200-rollout gate by default; a 1000-rollout variant is marked ``slow``.
"""
from __future__ import annotations

import dataclasses
import os
import random

import jax
import numpy as np
import pytest

from balatro_rl.engine import engine
from balatro_rl.engine.jokers.base import JokerState
from balatro_rl.engine.state import Phase
from balatro_rl.envs.actions import MAX_JOKERS, decode, legal_mask
from balatro_rl.envs.obs import OBS_SHAPES, encode
from balatro_rl.envs.rewards import Shaped
from balatro_rl.engine_jax import step as J
from balatro_rl.engine_jax.config import MAX_HAND, Phase as JPhase
from balatro_rl.engine_jax.obs import encode_core
from balatro_rl.engine_jax.rewards import shaped_core
from tests.engine_jax.parity_util import (
    assert_hand_slots_equal,
    assert_states_equal,
    build_required_table,
    deck_from_python,
    jax_core_fields,
    python_core_fields,
)
# Reuse the boundary machinery proven in Task 1.4.
from tests.engine_jax.test_progression_parity import (
    _assert_advance_scalars,
    _python_leave_shop,
    _resync_jax_from_python,
    _sel_mask,
)

# Cycle a spread of curriculum scales so rollouts span easy blinds (frequent clears ->
# many boundaries) through the real game (required=300 -> mostly short losses).
SCALES = (0.05, 0.2, 0.5, 1.0)
STEP_CAP = 400
OBS_ATOL = 1e-5
REWARD_ATOL = 1e-5

_oracle_reward = Shaped(gamma=0.999)

# step now folds the full joker pipeline (Task 2.6), so an UNCOMPILED trace costs
# ~0.5s — jit once at module level so the rollout loops below run in ~ms per step.
_JIT_STEP = jax.jit(J.step)

# The Phase-1 gate compares only the CORE observation keys. The remaining OBS keys
# (shop_*, boss_onehot, consum_*, pack_*, voucher_*, pending_consum) are
# Phase-3+ scope: the JAX core engine has no such state and zeros them. Python
# leaves them zero within a fresh blind (proven key-by-key in test_obs_parity), but
# its SHOP rolls e.g. a `voucher_offer` and `_advance_blind` carries it into the next
# blind — so across a boundary those fields legitimately diverge (the JAX core can't
# produce them), exactly like money/deck which the gate re-syncs from the oracle.
# These six keys ARE the complete core observation (global scalars, per-slot cards,
# hand mask, hand-type levels, and the deck histograms).
_OBS_CORE_KEYS = ("global", "hand", "hand_mask", "levels", "deck_rank_hist", "deck_suit_hist")

# Gate B additionally asserts the joker keys (filled by encode_core since Task 2.7).
# With a fixed loadout and a no-buy shop policy these are constant on both sides, so
# they are safe to compare within AND across blind boundaries.
_OBS_JOKER_KEYS = ("joker_types", "joker_counter", "joker_mask")


def _assert_obs_equal(cs, gs, where: str, keys=_OBS_CORE_KEYS) -> None:
    jx = encode_core(cs)
    py = encode(gs)
    assert set(jx.keys()) == set(py.keys()), f"{where}: obs key set mismatch"
    for key in keys:
        jv = np.asarray(jx[key])
        pv = np.asarray(py[key])
        assert jv.shape == pv.shape, f"{where}: obs '{key}' shape {jv.shape} != {pv.shape}"
        assert np.allclose(jv, pv, atol=OBS_ATOL, rtol=0.0), (
            f"{where}: obs '{key}' mismatch\nJAX={jv}\nPy ={pv}")


def _assert_reward_equal(prev_cs, cs, sig, prev_gs, gs, aid, info, where: str) -> None:
    jax_r = float(shaped_core(prev_cs, cs, sig.cleared, sig.won))
    py_r = float(_oracle_reward(prev_gs, aid, gs, info))
    assert abs(jax_r - py_r) < REWARD_ATOL, (
        f"{where}: reward mismatch JAX={jax_r} Py={py_r} (|d|={abs(jax_r - py_r):.2e})")


def _run_rollout(seed: int, scale: float, loadout=None) -> dict:
    """One lockstep rollout under a uniformly-random legal policy. Returns a summary
    {outcome, within, boundaries, obs_checks, reward_checks}. Raises on any mismatch.

    ``loadout=None`` is the Phase-1 gate (no jokers, core obs keys only). A list
    (possibly empty — the Gate-B regression case) injects the SAME fixed joker
    loadout into both engines at reset and additionally asserts the joker obs keys.
    """
    rng = random.Random(seed * 7919 + int(scale * 1000))
    req_table = build_required_table(scale)
    gs = engine.reset(seed, scale, None, False)
    jk = None
    obs_keys = _OBS_CORE_KEYS
    if loadout is not None:
        # Python oracle: replace the (empty) post-reset jokers tuple. None of the
        # in-scope jokers has an on_round_start hook, so this is identical to a
        # reset that started with the loadout. JAX: 0-padded int32[MAX_JOKERS].
        gs = dataclasses.replace(
            gs, jokers=tuple(JokerState(type=j) for j in loadout))
        jk = np.zeros(MAX_JOKERS, dtype=np.int32)
        jk[:len(loadout)] = loadout
        obs_keys = _OBS_CORE_KEYS + _OBS_JOKER_KEYS
    ranks, suits = deck_from_python(gs)
    cs = J.reset(ranks, suits, required=gs.required, required_table=req_table, jokers=jk)

    within = boundaries = obs_checks = reward_checks = 0

    for _ in range(STEP_CAP):
        # Compare the obs + legal-derived inputs on the CURRENT (pre-action) PLAYING state.
        _assert_obs_equal(cs, gs, f"seed={seed} scale={scale} pre-step", obs_keys)
        obs_checks += 1

        mask = legal_mask(gs)
        legal_ids = np.nonzero(mask[:436])[0]            # core action range only
        assert legal_ids.size > 0, "no legal core action while PLAYING"
        aid = int(rng.choice(legal_ids.tolist()))
        verb, idxs = decode(aid)
        sel = _sel_mask(idxs)

        prev_gs, prev_cs = gs, cs
        gs2, info = engine.step(gs, (verb, tuple(idxs)))
        cs2, sig = _JIT_STEP(cs, int(verb), sel)
        where = f"seed={seed} scale={scale} verb={int(verb)}"

        if gs2.won or gs2.phase == Phase.WON:
            assert bool(sig.won) and int(cs2.phase) == JPhase.WON and bool(cs2.done), where
            assert gs2.ante == int(cs2.ante) and int(cs2.blind_index) == gs2.blind_index == 2
            return _summary("won", within, boundaries, obs_checks, reward_checks)

        if gs2.phase == Phase.LOST:
            assert gs2.done and int(cs2.phase) == JPhase.LOST and bool(cs2.done), where
            assert not bool(sig.cleared)
            assert_states_equal(python_core_fields(gs2), jax_core_fields(cs2))
            assert_hand_slots_equal(gs2, cs2)
            _assert_obs_equal(cs2, gs2, where + " [LOST]", obs_keys)
            _assert_reward_equal(prev_cs, cs2, sig, prev_gs, gs2, aid, info, where + " [LOST]")
            within += 1; obs_checks += 1; reward_checks += 1
            return _summary("lost", within, boundaries, obs_checks, reward_checks)

        if gs2.phase == Phase.SHOP:
            # CLEAR (not win): JAX advanced this step; walk Python through the shop, then
            # assert in-scope advance scalars and re-sync the RNG/economy-divergent fields.
            assert bool(sig.cleared) and not bool(sig.won), where
            gs_next = _python_leave_shop(gs2)
            if gs_next.done:
                return _summary("lost", within, boundaries, obs_checks, reward_checks)
            _assert_advance_scalars(cs2, gs_next)
            cs2 = _resync_jax_from_python(cs2, gs_next)
            boundaries += 1
            gs, cs = gs_next, cs2
            continue

        # Still PLAYING: full within-blind parity, including obs + reward.
        assert not bool(sig.cleared), where
        assert_states_equal(python_core_fields(gs2), jax_core_fields(cs2))
        assert_hand_slots_equal(gs2, cs2)
        _assert_reward_equal(prev_cs, cs2, sig, prev_gs, gs2, aid, info, where)
        within += 1; reward_checks += 1
        gs, cs = gs2, cs2

    return _summary("cap", within, boundaries, obs_checks, reward_checks)


def _summary(outcome, within, boundaries, obs_checks, reward_checks) -> dict:
    return {"outcome": outcome, "within": within, "boundaries": boundaries,
            "obs_checks": obs_checks, "reward_checks": reward_checks}


def _run_gate(n_rollouts: int, base_seed: int = 0, loadout=None) -> dict:
    totals = {"won": 0, "lost": 0, "cap": 0,
              "within": 0, "boundaries": 0, "obs_checks": 0, "reward_checks": 0}
    for i in range(n_rollouts):
        scale = SCALES[i % len(SCALES)]
        s = _run_rollout(base_seed + i, scale, loadout)
        totals[s["outcome"]] += 1
        for k in ("within", "boundaries", "obs_checks", "reward_checks"):
            totals[k] += s[k]
    return totals


def test_core_parity_gate_200():
    """200 random-legal rollouts across 4 scales — full within-blind parity
    (state + ordered slots + obs + reward), boundary advance scalars, and terminals."""
    t = _run_gate(200)
    # Sanity: the gate must have actually exercised the engine broadly.
    assert t["within"] > 1000, f"too few within-blind transitions: {t}"
    assert t["boundaries"] > 0, f"no blind boundaries crossed: {t}"
    assert t["obs_checks"] > 1000 and t["reward_checks"] > 1000, f"thin obs/reward coverage: {t}"
    print(f"\nPARITY GATE (200 rollouts): {t}")


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("BALATRO_RUN_SLOW"),
                    reason="slow: set BALATRO_RUN_SLOW=1 to run the full 1000-rollout gate")
def test_core_parity_gate_1000():
    """The full Phase-1 parity gate: 1000 rollouts. Opt in with BALATRO_RUN_SLOW=1."""
    t = _run_gate(1000)
    assert t["within"] > 5000 and t["boundaries"] > 0, f"{t}"
    print(f"\nPARITY GATE (1000 rollouts): {t}")


# ---------------------------------------------------------------------------
# Gate B (Task 2.8): fixed-loadout episode parity
# ---------------------------------------------------------------------------
# One loadout per in-scope joker family + a high-interaction mix + the empty loadout
# (Phase-1 regression through the Gate-B path). All float32-safe: at most ONE Baron
# (stacked x1.5 from multi-Baron x many-Kings can exceed float32 parity; see the
# spec §4.3 / jokers.py Baron note).
_LOADOUTS = [
    [],                  # empty (Phase-1 regression through the gate)
    [1],                 # Joker (flat +mult, independent)
    [3],                 # Lusty Joker (suit on_score)
    [6],                 # Jolly Joker (contains-pair +mult, independent)
    [131],               # The Duo (pair xmult, independent)
    [78, 109],           # Photograph + Sock&Buskin (face xmult + face retrigger)
    [52, 3, 131, 109],   # Splash + suit + xmult + retrigger (high interaction)
    [72],                # Baron (held-card xmult; single Baron = float32-safe)
    [22, 23, 53],        # Banner + Mystic Summit + Blue Joker (context-linear)
]


def _run_loadout_episodes(loadout, n_rollouts: int, base_seed: int) -> dict:
    """Gate B: the Phase-1 within-blind harness with ``loadout`` injected into BOTH
    engines at reset; asserts joker obs keys in addition to the core keys, and keeps
    the Phase-1 boundary resync (the fixed loadout carries through `_advance_blind`
    / the no-buy shop unchanged on both sides)."""
    t = _run_gate(n_rollouts, base_seed=base_seed, loadout=list(loadout))
    # Sanity: the gate must have actually exercised the engine, incl. boundaries.
    assert t["within"] > 2 * n_rollouts, f"loadout={loadout}: too few within-blind transitions: {t}"
    assert t["boundaries"] > 0, f"loadout={loadout}: no blind boundaries crossed: {t}"
    assert t["obs_checks"] > 2 * n_rollouts and t["reward_checks"] > 2 * n_rollouts, (
        f"loadout={loadout}: thin obs/reward coverage: {t}")
    return t


@pytest.mark.parametrize(
    "loadout", _LOADOUTS,
    ids=lambda lo: "loadout-" + ("-".join(map(str, lo)) if lo else "empty"))
def test_episode_parity_with_loadout(loadout):
    """Gate B (CI): 50 random-legal rollouts per loadout across 4 scales — full
    within-blind parity (state + ordered slots + core/joker obs + reward)."""
    t = _run_loadout_episodes(loadout, n_rollouts=50, base_seed=0)
    print(f"\nGATE B (loadout={loadout}, 50 rollouts): {t}")


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("BALATRO_RUN_SLOW"),
                    reason="slow: set BALATRO_RUN_SLOW=1 to run the full Gate B")
def test_episode_parity_with_loadout_full():
    """The full Gate B: every loadout x 200 rollouts. Opt in with BALATRO_RUN_SLOW=1."""
    for loadout in _LOADOUTS:
        t = _run_loadout_episodes(loadout, n_rollouts=200, base_seed=1000)
        print(f"\nGATE B FULL (loadout={loadout}, 200 rollouts): {t}")

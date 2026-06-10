"""Within-blind PLAY/DISCARD parity (Task 1.3).

Drives BOTH the Python oracle and the JAX engine with the SAME flat action id,
asserting ``assert_states_equal`` after every step. Two scripted policies:

  * play-policy:    always take the lowest-index legal PLAY id.
  * discard-policy: take the lowest-index legal DISCARD id while discards remain,
                    else the lowest-index legal PLAY id (exercises discard+refill).

The flat id is decoded to a Python ``(Verb, idxs)`` for the oracle and to a JAX
``(int verb, sel_mask bool[8])`` for the JAX engine. Because the SAME idxs drive
both engines, any slot-order divergence in the JAX refill would desync the NEXT
action — so a 30-step rollout passing is strong evidence the slot order matches
Python's ``_draw`` exactly.

Blind boundaries are out of scope here (Task 1.4): when the Python step CLEARS the
blind (lands in SHOP / WON) we break WITHOUT asserting (the JAX clear half-state is
intentionally uncompared). The LOSS terminal IS compared (refilled hand + LOST).
"""
import jax
import jax.numpy as jnp
import numpy as np

from balatro_rl.engine import engine
from balatro_rl.engine.state import Phase
from balatro_rl.envs.actions import decode, legal_mask, MAX_JOKERS, PLAY_N, _SUBSETS
from balatro_rl.engine_jax import step as J
from balatro_rl.engine_jax.config import MAX_HAND
from tests.engine_jax.parity_util import (
    assert_hand_slots_equal,
    assert_states_equal,
    deck_from_python,
    jax_core_fields,
    python_core_fields,
)

# step now folds the full joker pipeline (Task 2.6), so an UNCOMPILED trace costs
# ~0.5s — jit once at module level so the per-step loop below runs in ~ms.
_JIT_STEP = jax.jit(J.step)

N_SEEDS = 50
MAX_STEPS = 30
SCALE = 0.2  # low blind targets so a blind clears within a few hands (boundary -> break)


def _sel_mask(idxs) -> np.ndarray:
    """Build a bool[8] slot-selection mask from a tuple of hand indices."""
    m = np.zeros(MAX_HAND, dtype=bool)
    for i in idxs:
        m[i] = True
    return m


def _lowest_play_id(mask) -> int | None:
    """Lowest-index legal PLAY flat id, or None if none legal."""
    ids = np.nonzero(mask[:PLAY_N])[0]
    return int(ids[0]) if ids.size else None


def _lowest_discard_id(mask) -> int | None:
    """Lowest-index legal DISCARD flat id (range [PLAY_N, 2*PLAY_N)), or None."""
    ids = np.nonzero(mask[PLAY_N:2 * PLAY_N])[0]
    return int(ids[0] + PLAY_N) if ids.size else None


def _run_policy(pick_action_id):
    """Run a scripted policy over N_SEEDS, returning the count of compared
    within-blind transitions (sanity: must be > 0). ``pick_action_id(gs, mask)``
    returns the chosen flat id (or None to stop this seed)."""
    compared = 0
    for seed in range(N_SEEDS):
        gs = engine.reset(seed, SCALE, None, False)
        ranks, suits = deck_from_python(gs)
        cs = J.reset(ranks, suits, required=gs.required)

        for _ in range(MAX_STEPS):
            mask = legal_mask(gs)
            aid = pick_action_id(gs, mask)
            if aid is None:
                break
            verb, idxs = decode(aid)
            sel = _sel_mask(idxs)

            gs2, _info = engine.step(gs, (verb, tuple(idxs)))
            cs2, sig = _JIT_STEP(cs, int(verb), sel)

            if gs2.phase in (Phase.SHOP, Phase.WON) or gs2.won:
                # Blind cleared: JAX clear half-state is intentionally uncompared
                # here (Task 1.4 covers blind boundaries), but the `cleared` signal
                # must already agree with the oracle.
                assert bool(sig.cleared), "JAX did not flag cleared on a Python clear"
                break
            elif gs2.phase == Phase.LOST:
                assert gs2.done
                assert not bool(sig.cleared)
                assert_states_equal(python_core_fields(gs2), jax_core_fields(cs2))
                assert_hand_slots_equal(gs2, cs2)
                compared += 1
                break
            else:
                assert not bool(sig.cleared)
                assert_states_equal(python_core_fields(gs2), jax_core_fields(cs2))
                assert_hand_slots_equal(gs2, cs2)
                compared += 1
                gs, cs = gs2, cs2
    return compared


def test_play_policy_parity():
    """Always PLAY the lowest-index legal subset (a PLAY early on)."""
    compared = _run_policy(lambda gs, mask: _lowest_play_id(mask))
    assert compared > 0, "no within-blind PLAY transitions were compared"


def test_discard_policy_parity():
    """DISCARD the lowest-index legal subset while discards remain, else PLAY.

    Exercises discard+refill slot-order parity, then plays to (eventually) clear
    or lose once discards are exhausted.
    """
    def pick(gs, mask):
        if gs.discards_left > 0:
            did = _lowest_discard_id(mask)
            if did is not None:
                return did
        return _lowest_play_id(mask)

    compared = _run_policy(pick)
    assert compared > 0, "no within-blind transitions were compared"


def test_step_uses_jokers_loadout():
    # Deterministic deck: first 8 cards are the opening hand. Make slots 0,1 a pair of Aces.
    ranks = [14, 14] + [r for r in range(2, 15) for _ in range(4)][:50]
    suits = [0, 1] + [s for _ in range(2, 15) for s in range(4)][:50]
    jk = np.zeros(MAX_JOKERS, np.int32); jk[0] = 1  # JOKER +4 mult
    st = J.reset(ranks, suits, required=10**9, jokers=jk)  # huge required -> never clears
    # PLAY the pair of Aces: subset {0,1}.
    aid = _SUBSETS.index((0, 1))
    verb, sel = J.decode_action(jnp.int32(aid))
    ns, sig = _JIT_STEP(st, verb, sel)
    # PAIR base 10c/2m; aces 11+11 -> 32c; Joker +4 -> mult 6 -> 192.
    assert int(sig.score) == 192

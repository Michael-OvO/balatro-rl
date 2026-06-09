"""Full-episode blind/ante progression parity (Task 1.4).

Drives BOTH the Python oracle and the JAX engine with the SAME flat action id over
whole episodes that cross blind boundaries, asserting parity at every within-blind
transition, at the WIN / LOSS terminals, and (the new bit) at each blind ADVANCE.

The oracle clears a blind into a SHOP phase, then advances on shop-leave. The JAX
engine collapses that to a direct advance (money / shop are out of its scope). So at
every clear the harness:

  1. lets JAX advance in the single step that cleared,
  2. walks the Python state through the shop *buying nothing* (issue LEAVE_SHOP, and
     defensively SKIP_PACK if a pack ever opens) until Python re-reaches PLAYING,
  3. asserts JAX's freshly-advanced scalars match the Python next-blind scalars
     (ante / blind_index / round_score=0 / required / hands_left / discards_left /
     hand_size / phase / done) — EXCLUDING money + the RNG-divergent deck/hand,
  4. re-syncs JAX's deck/hand/money from the Python next-blind state so the following
     within-blind segment is byte-comparable again.

Greedy policy: among legal PLAY ids, take the one whose decoded subset SCORES
HIGHEST (evaluated via the oracle's ``score_play`` on the chosen cards). This
actually clears blinds (vs. the lowest-id policy in test_step_parity), exercising
the boundary handling.

PART C (win path) runs the same driver at a tiny scale where every required floors
to 1, so every play clears and both engines race to the ante-8 boss WIN.
"""
import jax
import numpy as np

from balatro_rl.engine import engine
from balatro_rl.engine.engine import Verb
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.state import Phase
from balatro_rl.envs.actions import PLAY_N, decode, legal_mask
from balatro_rl.engine_jax import step as J
from balatro_rl.engine_jax.config import MAX_HAND, Phase as JPhase
from tests.engine_jax.parity_util import (
    assert_hand_slots_equal,
    assert_states_equal,
    build_required_table,
    deck_from_python,
    jax_core_fields,
    python_core_fields,
)

# step now folds the full joker pipeline (Task 2.6), so an UNCOMPILED trace costs
# ~0.5s — jit once at module level so the per-step loop below runs in ~ms.
_JIT_STEP = jax.jit(J.step)

# Scales:
#   0.2  -> crosses some boundaries, mostly ends in LOSS (boundary handling exercise).
#   1e-9 -> every blind's required floors to 1, so every play clears -> race to WIN.
SCALE = 0.2
WIN_SCALE = 1e-9


def _sel_mask(idxs) -> np.ndarray:
    """Build a bool[8] slot-selection mask from a tuple of hand indices."""
    m = np.zeros(MAX_HAND, dtype=bool)
    for i in idxs:
        m[i] = True
    return m


def _best_play_id(gs, mask):
    """Highest-scoring legal PLAY flat id (evaluated via the oracle ``score_play``
    on the decoded cards), or None if no PLAY is legal. Ties resolve to the lowest id.
    """
    best_id, best_score = None, -1
    for aid in np.nonzero(mask[:PLAY_N])[0]:
        aid = int(aid)
        _verb, idxs = decode(aid)
        sel = [gs.hand[i] for i in idxs]
        res = score_play(sel)
        if res.score > best_score:
            best_score, best_id = res.score, aid
    return best_id


def _python_leave_shop(gs):
    """Advance the Python state through the shop / any pack buying NOTHING, until it
    re-reaches PLAYING (or terminates). Returns the next PLAYING (or terminal) state.

    With the greedy PLAY-only policy the only intermediate phase is SHOP, left via
    LEAVE_SHOP. OPEN_PACK never arises here (we never issue OPEN), but we handle it
    defensively with SKIP_PACK so the catch-up loop is robust to any future change.
    """
    guard = 0
    while not gs.done and gs.phase != Phase.PLAYING:
        guard += 1
        assert guard < 20, f"shop catch-up did not terminate (phase={Phase(gs.phase).name})"
        if gs.phase == Phase.SHOP:
            gs, _info = engine.step(gs, (Verb.LEAVE_SHOP, 0))
        elif gs.phase == Phase.OPEN_PACK:
            gs, _info = engine.step(gs, (Verb.SKIP_PACK, 0))
        else:
            raise AssertionError(
                f"unexpected phase during shop catch-up: {Phase(gs.phase).name}")
    return gs


def _resync_jax_from_python(cs, gs_next):
    """Rebuild the JAX state's RNG-divergent + out-of-scope fields from the Python
    next-blind state so the next within-blind segment is byte-comparable.

    The JAX advance reshuffled with its own PRNG (for standalone PPO), and money / shop
    are out of JAX scope, so deck/hand/deck_ptr/hand_mask/money diverge from Python at
    a boundary. We overwrite exactly those from ``gs_next`` (everything else — ante,
    blind_index, required, hands_left, ... — was just asserted equal, so it stays).
    """
    import jax.numpy as jnp

    r, s = deck_from_python(gs_next)  # full 52-card draw order (hand + deck)
    hand_rank = np.zeros(MAX_HAND, dtype=np.int8)
    hand_suit = np.zeros(MAX_HAND, dtype=np.int8)
    hand_mask = np.zeros(MAX_HAND, dtype=bool)
    for i, c in enumerate(gs_next.hand):
        hand_rank[i] = int(c.rank)
        hand_suit[i] = int(c.suit)
        hand_mask[i] = True

    return cs._replace(
        deck_rank=jnp.asarray(r, dtype=jnp.int8),
        deck_suit=jnp.asarray(s, dtype=jnp.int8),
        deck_ptr=jnp.array(MAX_HAND, dtype=jnp.int32),
        hand_rank=jnp.asarray(hand_rank, dtype=jnp.int8),
        hand_suit=jnp.asarray(hand_suit, dtype=jnp.int8),
        hand_mask=jnp.asarray(hand_mask, dtype=bool),
        money=jnp.array(int(gs_next.money), dtype=jnp.int32),
    )


def _assert_advance_scalars(cs2, gs_next):
    """Assert the JAX advance produced the SAME in-scope scalars Python reaches after
    leaving the shop (EXCLUDING money + the RNG-divergent deck/hand)."""
    assert int(cs2.ante) == gs_next.ante, (
        f"advance ante: JAX={int(cs2.ante)} Python={gs_next.ante}")
    assert int(cs2.blind_index) == gs_next.blind_index, (
        f"advance blind: JAX={int(cs2.blind_index)} Python={gs_next.blind_index}")
    assert int(cs2.round_score) == 0 == gs_next.round_score, (
        f"advance round_score: JAX={int(cs2.round_score)} Python={gs_next.round_score}")
    assert int(cs2.required) == gs_next.required, (
        f"advance required: JAX={int(cs2.required)} Python={gs_next.required}")
    assert int(cs2.hands_left) == gs_next.hands_left, (
        f"advance hands_left: JAX={int(cs2.hands_left)} Python={gs_next.hands_left}")
    assert int(cs2.discards_left) == gs_next.discards_left, (
        f"advance discards_left: JAX={int(cs2.discards_left)} Python={gs_next.discards_left}")
    assert int(cs2.hand_size) == gs_next.hand_size, (
        f"advance hand_size: JAX={int(cs2.hand_size)} Python={gs_next.hand_size}")
    assert int(cs2.phase) == JPhase.PLAYING, f"advance phase: JAX={int(cs2.phase)}"
    assert not bool(cs2.done), "advance should not be done"


def _run_episode(seed, scale, step_cap):
    """Drive one greedy episode in lockstep across both engines. Returns a small
    summary dict: {outcome: 'won'|'lost'|'cap', boundaries: int, within: int}."""
    req_table = build_required_table(scale)
    gs = engine.reset(seed, scale, None, False)
    ranks, suits = deck_from_python(gs)
    cs = J.reset(ranks, suits, required=gs.required, required_table=req_table)

    boundaries = 0   # blind advances crossed
    within = 0       # within-blind transitions compared

    for _ in range(step_cap):
        mask = legal_mask(gs)
        aid = _best_play_id(gs, mask)
        assert aid is not None, "no legal PLAY while PLAYING (unexpected)"
        verb, idxs = decode(aid)
        sel = _sel_mask(idxs)

        gs2, _info = engine.step(gs, (verb, tuple(idxs)))
        cs2, sig = _JIT_STEP(cs, int(verb), sel)

        if gs2.won or gs2.phase == Phase.WON:
            # WIN: cleared the ante-8 boss. JAX must flag won + reach the WON terminal.
            assert bool(sig.won), "JAX did not flag won on a Python win"
            assert int(cs2.phase) == JPhase.WON, f"JAX phase {int(cs2.phase)} != WON"
            assert bool(cs2.done), "JAX win not done"
            assert gs2.ante == int(cs2.ante), (
                f"win ante: Python={gs2.ante} JAX={int(cs2.ante)}")
            assert int(cs2.blind_index) == gs2.blind_index == 2
            return {"outcome": "won", "boundaries": boundaries, "within": within}

        if gs2.phase == Phase.LOST:
            # LOSS: refilled hand + LOST terminal must match byte-for-byte.
            assert gs2.done
            assert int(cs2.phase) == JPhase.LOST, f"JAX phase {int(cs2.phase)} != LOST"
            assert bool(cs2.done), "JAX loss not done"
            assert not bool(sig.cleared)
            assert_states_equal(python_core_fields(gs2), jax_core_fields(cs2))
            assert_hand_slots_equal(gs2, cs2)
            within += 1
            return {"outcome": "lost", "boundaries": boundaries, "within": within}

        if gs2.phase == Phase.SHOP:
            # CLEAR (not a win): JAX already advanced this step; Python is in the shop.
            assert bool(sig.cleared), "JAX did not flag cleared on a Python clear"
            assert not bool(sig.won)
            gs_next = _python_leave_shop(gs2)
            if gs_next.done:
                # A shop-leave terminal shouldn't happen with bosses off, but stay safe.
                assert gs_next.phase != Phase.WON or bool(cs2.won)
                return {"outcome": "lost", "boundaries": boundaries, "within": within}
            _assert_advance_scalars(cs2, gs_next)
            cs2 = _resync_jax_from_python(cs2, gs_next)
            boundaries += 1
            gs, cs = gs_next, cs2
            continue

        # Still PLAYING (non-clearing play).
        assert not bool(sig.cleared)
        assert_states_equal(python_core_fields(gs2), jax_core_fields(cs2))
        assert_hand_slots_equal(gs2, cs2)
        within += 1
        gs, cs = gs2, cs2

    return {"outcome": "cap", "boundaries": boundaries, "within": within}


# ---------------------------------------------------------------------------
# PART B: full-episode parity across boundaries (scale 0.2)
# ---------------------------------------------------------------------------

def test_progression_parity_scale_02():
    """50 greedy episodes at scale 0.2: parity holds within blinds, at LOSS, and at
    every blind ADVANCE; at least some boundaries are crossed."""
    total_boundaries = 0
    total_within = 0
    for seed in range(50):
        summary = _run_episode(seed, SCALE, step_cap=300)
        total_boundaries += summary["boundaries"]
        total_within += summary["within"]
    assert total_within > 0, "no within-blind transitions were compared"
    assert total_boundaries > 0, "no blind boundaries were crossed (advance untested)"


# ---------------------------------------------------------------------------
# PART C: explicit WIN path (tiny scale -> every play clears -> race to ante 8)
# ---------------------------------------------------------------------------

def test_win_path_tiny_scale():
    """At WIN_SCALE every blind's required floors to 1, so each play clears and both
    engines climb the full ante ladder to the ante-8 boss WIN. Exercises the win
    branch + the harness shop-skip across 23 boundaries."""
    for seed in range(3):
        summary = _run_episode(seed, WIN_SCALE, step_cap=60)
        assert summary["outcome"] == "won", (
            f"seed {seed}: expected WIN, got {summary['outcome']} "
            f"(boundaries={summary['boundaries']})")
        # 8 antes x 3 blinds = 24 clears to win, climbing (1,0) -> ... -> (8,2). The
        # 24th clear (ante 8, blind 2) IS the win; the first 23 are advances to a next
        # blind. So 24 clears, 23 of which are advances (the 24th clear is the win).
        assert summary["boundaries"] == 23, (
            f"seed {seed}: expected 23 advances before the win, got {summary['boundaries']}")

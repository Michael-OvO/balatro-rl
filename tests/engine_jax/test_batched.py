"""Task 1.7 / 1.8: batched env (vmap) + flat-action adapter + reward/done contract.

Tests:
  1. decode_action matches Python decode on [0, 436).
  2. batched_reset over N=1024 distinct keys produces CoreState with leading dim 1024.
  3. vmapped encode_core over batched state produces obs dict with leading dim 1024.
  4. jit(batched_step) over N=1024 envs runs; outputs have leading dim 1024.
  5. vmap consistency: batched_step lane i == step_with_action(states[i], action_ids[i])
     for single-card PLAY, a DISCARD id, and a multi-card PLAY id; reward/done compared.
  6. auto-reset: step an env to done -> returned state is fresh episode while
     signals show terminal.
  7. Reward uses the TERMINAL state (not the fresh-reset state): a LOST transition
     must produce a reward that differs from a fresh-episode reward.
"""
import numpy as np
import jax
import jax.numpy as jnp

from balatro_rl.envs.actions import _SUBSETS, decode as py_decode, PLAY_N, MAX_JOKERS
from balatro_rl.engine_jax import step as J
from balatro_rl.engine_jax.obs import encode_core, legal_mask_core
from balatro_rl.engine_jax.config import Phase, Verb, HANDS_PER_BLIND, DISCARDS_PER_BLIND, MAX_HAND
from balatro_rl.engine_jax.rewards import shaped_core
from tests.engine_jax.parity_util import build_required_table


def _zero_jokers(n: int) -> jnp.ndarray:
    """Return a zeros int32[n, MAX_JOKERS] joker loadout (empty, Phase-1 compatible)."""
    return jnp.zeros((n, MAX_JOKERS), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_required_table():
    return jnp.asarray(build_required_table(1.0), dtype=jnp.int32)


def _make_keys(n: int, seed: int = 42):
    """Return n distinct JAX PRNGKeys derived from seed."""
    base = jax.random.PRNGKey(seed)
    return jax.random.split(base, n)


# ---------------------------------------------------------------------------
# 1. decode_action matches Python decode on [0, 436)
# ---------------------------------------------------------------------------

def test_decode_action_matches_python_play():
    """PLAY ids [0, 218): same verb and slot set as Python decode."""
    for action_id in range(PLAY_N):
        py_verb, py_idxs = py_decode(action_id)
        jax_verb, jax_sel = J.decode_action(jnp.int32(action_id))

        assert int(jax_verb) == int(py_verb), (
            f"action_id={action_id}: verb mismatch: JAX={int(jax_verb)}, Python={int(py_verb)}"
        )
        # sel_mask -> set of True indices
        jax_slots = frozenset(int(i) for i in range(MAX_HAND) if bool(jax_sel[i]))
        py_slots = frozenset(py_idxs)
        assert jax_slots == py_slots, (
            f"action_id={action_id}: slot mismatch: JAX={sorted(jax_slots)}, Python={sorted(py_slots)}"
        )


def test_decode_action_matches_python_discard():
    """DISCARD ids [218, 436): same verb and slot set as Python decode."""
    for action_id in range(PLAY_N, 2 * PLAY_N):
        py_verb, py_idxs = py_decode(action_id)
        jax_verb, jax_sel = J.decode_action(jnp.int32(action_id))

        assert int(jax_verb) == int(py_verb), (
            f"action_id={action_id}: verb mismatch: JAX={int(jax_verb)}, Python={int(py_verb)}"
        )
        jax_slots = frozenset(int(i) for i in range(MAX_HAND) if bool(jax_sel[i]))
        py_slots = frozenset(py_idxs)
        assert jax_slots == py_slots, (
            f"action_id={action_id}: slot mismatch: JAX={sorted(jax_slots)}, Python={sorted(py_slots)}"
        )


def test_decode_action_shop_ids_noop():
    """Shop/pack ids [436, 708): decode returns verb=PLAY, sel_mask=all-False (no-op)."""
    for action_id in range(2 * PLAY_N, J.NUM_ACTIONS_JAX):
        jax_verb, jax_sel = J.decode_action(jnp.int32(action_id))
        assert int(jax_verb) == Verb.PLAY, f"action_id={action_id}: expected PLAY no-op"
        assert not jnp.any(jax_sel), f"action_id={action_id}: expected all-False sel_mask"


def test_decode_action_is_jittable():
    """decode_action must be JIT-compilable."""
    jit_decode = jax.jit(J.decode_action)
    verb, sel = jit_decode(jnp.int32(5))
    assert int(verb) == Verb.PLAY
    verb2, sel2 = jit_decode(jnp.int32(220))
    assert int(verb2) == Verb.DISCARD


# ---------------------------------------------------------------------------
# 2. reset_jax: standalone JAX reset
# ---------------------------------------------------------------------------

def test_reset_jax_shapes():
    """reset_jax returns a CoreState with correct field shapes."""
    key = jax.random.PRNGKey(7)
    req_table = _default_required_table()
    state = J.reset_jax(key, req_table)

    assert state.deck_rank.shape == (52,)
    assert state.hand_rank.shape == (8,)
    assert state.hand_mask.shape == (8,)
    assert state.levels.shape == (12,)
    assert state.required_table.shape == (9, 3)
    assert state.rng.shape == (2,)


def test_reset_jax_initial_values():
    """reset_jax produces correct initial scalars: ante=1, blind=0, etc."""
    key = jax.random.PRNGKey(99)
    req_table = _default_required_table()
    state = J.reset_jax(key, req_table)

    assert int(state.ante) == 1
    assert int(state.blind_index) == 0
    assert int(state.round_score) == 0
    assert int(state.hands_left) == HANDS_PER_BLIND
    assert int(state.discards_left) == DISCARDS_PER_BLIND
    assert int(state.hand_size) == MAX_HAND
    assert int(state.phase) == Phase.PLAYING
    assert not bool(state.done)
    assert not bool(state.won)
    # All 8 hand slots occupied at reset
    assert jnp.all(state.hand_mask)
    # levels all-ones
    assert jnp.all(state.levels == 1)


def test_reset_jax_52_unique_cards():
    """reset_jax deck contains exactly 52 cards with 4 suits × 13 ranks each."""
    key = jax.random.PRNGKey(0)
    req_table = _default_required_table()
    state = J.reset_jax(key, req_table)

    ranks = np.asarray(state.deck_rank)
    suits = np.asarray(state.deck_suit)
    cards = list(zip(ranks.tolist(), suits.tolist()))
    assert len(cards) == 52
    assert len(set(cards)) == 52, "Duplicate cards in deck"
    # ranks should be in 2..14, suits in 0..3
    assert all(2 <= r <= 14 for r, _ in cards)
    assert all(0 <= s <= 3 for _, s in cards)


def test_reset_jax_different_keys_give_different_decks():
    """Different keys produce different shuffles."""
    req_table = _default_required_table()
    s1 = J.reset_jax(jax.random.PRNGKey(1), req_table)
    s2 = J.reset_jax(jax.random.PRNGKey(2), req_table)
    # Very unlikely that two random shuffles of 52 are identical
    assert not jnp.array_equal(s1.deck_rank, s2.deck_rank) or \
           not jnp.array_equal(s1.deck_suit, s2.deck_suit)


def test_reset_jax_required_set_from_table():
    """reset_jax sets required = required_table[1, 0] (ante=1, small blind)."""
    key = jax.random.PRNGKey(0)
    req_table = _default_required_table()
    state = J.reset_jax(key, req_table)
    expected_required = int(req_table[1, 0])
    assert int(state.required) == expected_required


# ---------------------------------------------------------------------------
# 3. batched_reset
# ---------------------------------------------------------------------------

N = 1024

def test_batched_reset_shape():
    """batched_reset over N=1024 keys -> CoreState with leading dim 1024."""
    keys = _make_keys(N)
    req_table = _default_required_table()
    states = J.batched_reset(keys, req_table, _zero_jokers(N))

    assert states.deck_rank.shape == (N, 52)
    assert states.hand_rank.shape == (N, MAX_HAND)
    assert states.hand_mask.shape == (N, MAX_HAND)
    assert states.ante.shape == (N,)
    assert states.required_table.shape == (N, 9, 3)
    assert states.rng.shape == (N, 2)


def test_batched_reset_jit():
    """batched_reset is JIT-compilable."""
    keys = _make_keys(N)
    req_table = _default_required_table()
    jit_reset = jax.jit(J.batched_reset)
    states = jit_reset(keys, req_table, _zero_jokers(N))
    assert states.ante.shape == (N,)


def test_reset_jax_nonzero_loadout_stored_verbatim():
    """reset_jax stores a non-zero joker loadout verbatim; batched_reset maps
    per-env loadouts (in_axes=(0, None, 0) is per-env, not broadcast)."""
    req_table = _default_required_table()

    # Single env: non-zero loadout stored verbatim.
    loadout = jnp.array([1, 131, 0, 0, 0, 0], dtype=jnp.int32)
    assert loadout.shape == (MAX_JOKERS,)
    state = J.reset_jax(jax.random.PRNGKey(3), req_table, loadout)
    assert jnp.array_equal(state.jokers, loadout)

    # Batched: per-env DISTINCT loadouts each stored at the matching env index.
    n = 4
    loadouts = jnp.stack([
        jnp.array([1, 131, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.array([2, 0, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.array([0, 0, 0, 0, 0, 7], dtype=jnp.int32),
        jnp.array([3, 4, 5, 0, 0, 0], dtype=jnp.int32),
    ])
    states = J.batched_reset(_make_keys(n), req_table, loadouts)
    assert states.jokers.shape == (n, MAX_JOKERS)
    assert jnp.array_equal(states.jokers, loadouts)


def test_batched_reset_obs_shape():
    """vmapped encode_core over batched state -> obs dict each value leading dim N."""
    keys = _make_keys(N)
    req_table = _default_required_table()
    states = J.batched_reset(keys, req_table, _zero_jokers(N))
    batch_encode = jax.vmap(encode_core)
    obs = batch_encode(states)
    for k, v in obs.items():
        assert v.shape[0] == N, f"obs['{k}'] has leading dim {v.shape[0]}, expected {N}"


# ---------------------------------------------------------------------------
# 4. step_with_action — new signature: (final_state, reward, done, signals)
# ---------------------------------------------------------------------------

def test_step_with_action_basic():
    """step_with_action returns (CoreState, reward, done, StepSignals) for a legal action."""
    key = jax.random.PRNGKey(5)
    req_table = _default_required_table()
    state = J.reset_jax(key, req_table)
    # action 0 = PLAY subset (0,) — single card; always legal at fresh start
    final_state, reward, done, signals = J.step_with_action(state, jnp.int32(0))
    assert isinstance(final_state, J.CoreState)
    assert isinstance(signals, J.StepSignals)
    assert reward.shape == ()          # scalar float32
    assert reward.dtype == jnp.float32
    assert done.shape == ()            # scalar bool
    assert done.dtype == jnp.bool_


def test_step_with_action_jit():
    """step_with_action is JIT-compilable with new 4-tuple return."""
    key = jax.random.PRNGKey(3)
    req_table = _default_required_table()
    state = J.reset_jax(key, req_table)
    jit_step = jax.jit(J.step_with_action)
    final_state, reward, done, signals = jit_step(state, jnp.int32(0))
    assert final_state.ante.shape == ()
    assert reward.shape == ()
    assert done.shape == ()


# ---------------------------------------------------------------------------
# 5. batched_step — new signature: (final_states, rewards, dones, signals)
# ---------------------------------------------------------------------------

def test_batched_step_shape():
    """jit(batched_step) over N envs: outputs have leading dim N."""
    keys = _make_keys(N)
    req_table = _default_required_table()
    states = J.batched_reset(keys, req_table, _zero_jokers(N))

    # Use action 0 (PLAY first card) for all envs
    action_ids = jnp.zeros(N, dtype=jnp.int32)

    jit_batched_step = jax.jit(J.batched_step)
    next_states, rewards, dones, signals = jit_batched_step(states, action_ids)

    assert next_states.ante.shape == (N,)
    assert next_states.deck_rank.shape == (N, 52)
    assert rewards.shape == (N,)
    assert rewards.dtype == jnp.float32
    assert dones.shape == (N,)
    assert signals.cleared.shape == (N,)
    assert signals.score.shape == (N,)


def test_batched_step_vmap_consistency():
    """batched_step lane i == step_with_action(states[i], action_ids[i]).

    Covers:
      - A single-card PLAY id (0)
      - A DISCARD id (PLAY_N + 0 = 218)
      - A multi-card PLAY id (e.g. subset index for (0,1) = 8)
      - Checks reward and done outputs match between batched and single-env.
    """
    N_CHECK = 8
    keys = _make_keys(N_CHECK, seed=7)
    req_table = _default_required_table()
    states = J.batched_reset(keys, req_table, _zero_jokers(N_CHECK))

    # Mix of action types:
    #   lanes 0..1:  single-card PLAY  (ids 0, 1)
    #   lanes 2..3:  DISCARD ids       (ids PLAY_N, PLAY_N+1)
    #   lanes 4..5:  multi-card PLAY   (ids for 2-card subsets; first 2-card subset
    #                                   in _SUBSETS is (0,1) at index 8)
    #   lanes 6..7:  single-card PLAY  (ids 6, 7)
    _first_2card = next(i for i, s in enumerate(_SUBSETS) if len(s) == 2)
    action_ids = jnp.array([
        0, 1,
        PLAY_N, PLAY_N + 1,
        _first_2card, _first_2card + 1,
        6, 7,
    ], dtype=jnp.int32)

    batch_next, batch_rewards, batch_dones, batch_sigs = J.batched_step(states, action_ids)

    for i in range(N_CHECK):
        lane_state = jax.tree_util.tree_map(lambda x: x[i], states)
        single_next, single_reward, single_done, single_sigs = J.step_with_action(
            lane_state, action_ids[i])

        # Compare state scalars
        for field in ("ante", "blind_index", "round_score", "hands_left",
                      "discards_left", "phase", "done"):
            batch_val = int(getattr(batch_next, field)[i])
            single_val = int(getattr(single_next, field))
            assert batch_val == single_val, (
                f"Lane {i}, field '{field}': batched={batch_val}, single={single_val}"
            )

        # Compare hand mask
        batch_mask = np.asarray(batch_next.hand_mask[i], dtype=bool)
        single_mask = np.asarray(single_next.hand_mask, dtype=bool)
        assert np.array_equal(batch_mask, single_mask), f"Lane {i} hand_mask mismatch"

        # Compare signals
        for sig_field in ("cleared", "won", "hand_type", "score"):
            b_val = int(getattr(batch_sigs, sig_field)[i])
            s_val = int(getattr(single_sigs, sig_field))
            assert b_val == s_val, (
                f"Lane {i}, signal '{sig_field}': batched={b_val}, single={s_val}"
            )

        # Compare reward and done (new in Task 1.8)
        b_reward = float(batch_rewards[i])
        s_reward = float(single_reward)
        assert abs(b_reward - s_reward) < 1e-5, (
            f"Lane {i}, reward: batched={b_reward:.6f}, single={s_reward:.6f}"
        )
        b_done = bool(batch_dones[i])
        s_done = bool(single_done)
        assert b_done == s_done, (
            f"Lane {i}, done: batched={b_done}, single={s_done}"
        )


# ---------------------------------------------------------------------------
# 6. auto-reset semantics
# ---------------------------------------------------------------------------

def _force_to_done(state, req_table):
    """Drive a single-env state to done by exhausting hands_left with PLAY actions.

    We loop up to 200 steps; a LOST terminal will have done=True.
    If the blind clears before loss, keep stepping until done.
    Returns (final_state, last_signals) where final_state.done is True.
    Calls J.step directly (no auto-reset) so we can observe the terminal.
    """
    max_iters = 200
    cur = state
    last_sigs = None
    for _ in range(max_iters):
        if bool(cur.done):
            break
        # Pick lowest legal PLAY action
        mask = np.asarray(legal_mask_core(cur), dtype=bool)
        play_ids = np.nonzero(mask[:PLAY_N])[0]
        if len(play_ids) == 0:
            break
        aid = int(play_ids[0])
        verb, sel = J.decode_action(jnp.int32(aid))
        cur, last_sigs = J.step(cur, verb, sel)
    return cur, last_sigs


def test_auto_reset_returns_fresh_state():
    """step_with_action on a done state auto-resets: returned state is fresh episode."""
    key = jax.random.PRNGKey(17)
    req_table = _default_required_table()

    state = J.reset_jax(key, req_table)
    done_state, _ = _force_to_done(state, req_table)
    assert bool(done_state.done), "Expected a done state"

    # Now call step_with_action on done_state with any action:
    # the auto-reset should give us a fresh state despite done=True input.
    final_state, reward, done_flag, signals = J.step_with_action(done_state, jnp.int32(0))

    # The returned STATE should be a fresh episode
    assert int(final_state.ante) == 1, f"Expected ante=1 after auto-reset, got {int(final_state.ante)}"
    assert int(final_state.round_score) == 0
    assert int(final_state.hands_left) == HANDS_PER_BLIND
    assert int(final_state.discards_left) == DISCARDS_PER_BLIND
    assert int(final_state.phase) == Phase.PLAYING
    assert not bool(final_state.done)
    assert jnp.all(final_state.hand_mask), "Fresh state should have full hand"


def test_auto_reset_signals_reflect_terminal():
    """step_with_action on a LOST transition: done==True and reward differs from fresh.

    Contract: the reward is computed on the TERMINAL next_state (before auto-reset).
    A LOST episode terminal must have done=True, and the reward (computed at the
    terminal ante/score) must not equal a fresh-episode no-op reward (which would
    use ante=1 / round_score=0 — the reset values).  This verifies the terminal
    state (not the fresh-reset state) is used in shaped_core.
    """
    key = jax.random.PRNGKey(31)
    req_table = _default_required_table()
    state = J.reset_jax(key, req_table)

    # Drive the state to the LAST hand before loss: hands_left=1, not yet done.
    # We step until hands_left == 1 without clearing (force PLAY on non-clearing state).
    cur = state
    max_iters = 200
    for _ in range(max_iters):
        if bool(cur.done) or int(cur.hands_left) == 1:
            break
        mask = np.asarray(legal_mask_core(cur), dtype=bool)
        play_ids = np.nonzero(mask[:PLAY_N])[0]
        if len(play_ids) == 0:
            break
        aid = int(play_ids[0])
        verb, sel = J.decode_action(jnp.int32(aid))
        next_s, sigs = J.step(cur, verb, sel)
        if bool(next_s.done):
            # Already went done before hands_left==1; that's fine — pick this terminal
            cur = next_s
            break
        if not bool(sigs.cleared):
            cur = next_s   # only advance if we didn't clear (want LOST path)
        else:
            # cleared; just advance anyway to keep progressing
            cur = next_s

    # At this point cur has hands_left <= 1 OR done.
    # Take one more PLAY step via step_with_action — the transition will be a loss
    # (or possibly a clear, but either way we get a done transition).
    # Force this using the underlying step + check done flag.
    if not bool(cur.done):
        # Take the terminal step directly to get the done signal
        mask = np.asarray(legal_mask_core(cur), dtype=bool)
        play_ids = np.nonzero(mask[:PLAY_N])[0]
        aid = int(play_ids[0]) if len(play_ids) > 0 else 0
        term_state, term_sigs = J.step(cur, *J.decode_action(jnp.int32(aid)))
        # term_state.done must be True (either LOST or cleared/advanced)
        # Now call step_with_action on cur (the pre-terminal state)
        final_state, reward, done_flag, signals = J.step_with_action(cur, jnp.int32(aid))

        # done_flag must be True (the transition produced a terminal)
        assert bool(done_flag), (
            f"Expected done=True for terminal transition, got done={bool(done_flag)}, "
            f"hands_left={int(cur.hands_left)}, cleared={bool(signals.cleared)}"
        )

        # The reward must be computed on the TERMINAL state (not the fresh reset).
        # Compute the "fresh state reward" as if we had used final_state (post-reset):
        # shaped_core(fresh, fresh, cleared=False, won=False) — a zero-progress reward.
        # The terminal reward uses the terminal state's fields (ante/round_score/required),
        # which differ from a fresh start; thus the rewards must differ.
        fresh_state = J.reset_jax(final_state.rng, final_state.required_table)
        # The terminal state that was actually used for the reward is term_state.
        expected_reward = shaped_core(cur, term_state, term_sigs.cleared, term_sigs.won)
        assert abs(float(reward) - float(expected_reward)) < 1e-5, (
            f"Reward mismatch: step_with_action gave {float(reward):.6f}, "
            f"expected {float(expected_reward):.6f} (computed on terminal state)"
        )

        # Verify fresh-state reward would differ (the whole point of this test).
        # A fresh state has round_score=0, ante=1; a mid-game terminal has higher ante
        # or non-zero round_score — its potential is different.
        fresh_reward = shaped_core(fresh_state, fresh_state, jnp.bool_(False), jnp.bool_(False))
        # This assertion checks the INVARIANT: if the terminal state differs from a
        # fresh state, the rewards must differ. We assert that the terminal reward is
        # NOT the same as a fresh-state trivial reward (unless the states happen to be
        # identical, which is extremely unlikely at a loss terminal with any progress).
        if int(cur.round_score) > 0 or int(cur.ante) > 1:
            assert abs(float(reward) - float(fresh_reward)) > 1e-6, (
                "Reward should differ between terminal and fresh-state potential, "
                f"but both gave {float(reward):.6f}. "
                "This indicates shaped_core used the fresh-reset state, not the terminal."
            )
    else:
        # cur is already done from the driving loop — still test step_with_action
        final_state, reward, done_flag, signals = J.step_with_action(cur, jnp.int32(0))
        assert int(final_state.ante) == 1
        assert not bool(final_state.done)


def test_auto_reset_in_batched_step():
    """batched_step: lanes that hit done auto-reset to fresh episode; live lanes stepped."""
    N_SMALL = 4
    keys = _make_keys(N_SMALL, seed=55)
    req_table = _default_required_table()
    states = J.batched_reset(keys, req_table, _zero_jokers(N_SMALL))

    # Force lane 0 to done by consuming all hands_left via step (underlying, no auto-reset)
    lane0 = jax.tree_util.tree_map(lambda x: x[0], states)
    done_lane0, _ = _force_to_done(lane0, req_table)
    assert bool(done_lane0.done)

    # Rebuild batched state with lane 0 replaced by done_lane0
    def replace_lane0(batch_arr, single_arr):
        return batch_arr.at[0].set(single_arr)

    mod_states = jax.tree_util.tree_map(replace_lane0, states, done_lane0)

    action_ids = jnp.zeros(N_SMALL, dtype=jnp.int32)
    next_states, rewards, dones, signals = J.batched_step(mod_states, action_ids)

    # Lane 0 should have been auto-reset to a fresh episode
    assert int(next_states.ante[0]) == 1
    assert not bool(next_states.done[0])
    assert int(next_states.hands_left[0]) == HANDS_PER_BLIND

    # Other lanes (1..3) should have been stepped once, not reset.
    # They each started with hands_left = HANDS_PER_BLIND and took one PLAY action,
    # so their hands_left must have decreased by 1 (unless they cleared, which is
    # unlikely at full scale for a single-card play).
    for lane_i in range(1, N_SMALL):
        # The lane was alive before the step, so it should now have hands_left < HANDS_PER_BLIND
        # (stepped once = hands_left decreased) OR have cleared (unlikely at scale 1.0).
        lane_hands_left = int(next_states.hands_left[lane_i])
        lane_cleared = bool(signals.cleared[lane_i])
        # After one PLAY step from a fresh state: either cleared a blind (rare) or
        # hands_left dropped by 1.
        assert lane_cleared or lane_hands_left == HANDS_PER_BLIND - 1, (
            f"Lane {lane_i}: expected hands_left={HANDS_PER_BLIND - 1} or cleared, "
            f"got hands_left={lane_hands_left}, cleared={lane_cleared}"
        )


def test_batched_step_large_n_jit():
    """Full N=1024 batched step under jit: smoke test."""
    keys = _make_keys(N)
    req_table = _default_required_table()
    states = jax.jit(J.batched_reset)(keys, req_table, _zero_jokers(N))
    action_ids = jnp.zeros(N, dtype=jnp.int32)
    next_states, rewards, dones, signals = jax.jit(J.batched_step)(states, action_ids)
    assert next_states.ante.shape == (N,)
    assert rewards.shape == (N,)
    assert dones.shape == (N,)
    assert signals.score.shape == (N,)


def test_batched_reset_accepts_jokers_and_empty_is_phase1():
    import jax, jax.numpy as jnp
    from balatro_rl.engine_jax.step import batched_reset, reset_jax
    from balatro_rl.engine_jax.curriculum import build_required_table
    from balatro_rl.envs.actions import MAX_JOKERS
    n = 4
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    rt = build_required_table(1.0)
    jk = jnp.zeros((n, MAX_JOKERS), dtype=jnp.int32)
    st = batched_reset(keys, rt, jk)
    assert st.jokers.shape == (n, MAX_JOKERS)
    # Single-env reset_jax with default jokers matches the all-zero loadout.
    s0 = reset_jax(keys[0], rt)
    assert int(jnp.sum(s0.jokers)) == 0

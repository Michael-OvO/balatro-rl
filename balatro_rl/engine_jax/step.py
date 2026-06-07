"""JAX engine step functions: reset (and future step/play/discard).

Only ``reset`` is implemented here for Task 0.4.  Future tasks will add
``step``, ``play``, and ``discard``.

Note: ``reset`` accepts host-provided deck ordering (ranks + suits as length-52
sequences) and a host-computed ``required`` score so that the JAX engine can be
seeded byte-identically to the Python oracle.  Real JAX-native seeding/shuffling
will be added in Task 1.7; for now the PRNG key is fixed at [0, 0].
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from balatro_rl.engine_jax.config import (
    ANTE_MAX,
    DISCARDS_PER_BLIND,
    HANDS_PER_BLIND,
    MAX_HAND,
    MAX_SELECT,
    N_HAND_TYPES,
    Phase,
    STARTING_MONEY,
    Verb,
)
from balatro_rl.engine_jax.scoring import score_core
from balatro_rl.engine_jax.state import DECK_SIZE, CoreState


def reset(
    deck_rank: Sequence[int],
    deck_suit: Sequence[int],
    required: int,
    required_table=None,
    scale_unused: float = 1.0,
) -> CoreState:
    """Build an initial CoreState from a host-provided 52-card draw order.

    Parameters
    ----------
    deck_rank:
        Length-52 sequence of rank values (int, 2..14) in full draw order:
        positions 0..7 are the cards dealt to the opening hand; 8..51 are the
        remaining draw pile (front = next draw).  Must match the Python oracle's
        ``deck_from_python(gs)`` output to achieve parity.
    deck_suit:
        Length-52 sequence of suit values (int, 0..3), same ordering.
    required:
        Required score to beat the first blind, host-computed via
        ``engine.required_score(1, 0, scale)``.  Stored directly.
    required_table:
        Optional int32[9,3] table of required scores indexed ``[ante, blind_index]``
        (ante 1..8, row 0 unused). Consulted by ``step`` when advancing across a
        blind boundary (Task 1.4) to set the next blind's ``required``. ``None`` ->
        an all-zeros placeholder (fine for tests that never cross a boundary).
    scale_unused:
        Curriculum scale parameter (not used here; kept for API symmetry with
        Python's ``reset(seed, scale, ...)``.  The host computes ``required``
        externally, so this value does not affect state.

    Returns
    -------
    CoreState
        Fully initialised game state for ante=1, small blind, opening hand
        already dealt.
    """
    # Convert host sequences to fixed-dtype JAX arrays.
    dr = jnp.asarray(deck_rank, dtype=jnp.int8)   # shape (52,)
    ds = jnp.asarray(deck_suit, dtype=jnp.int8)   # shape (52,)

    # The first MAX_HAND (8) cards are the opening hand.
    hand_rank = dr[:MAX_HAND]
    hand_suit = ds[:MAX_HAND]
    hand_mask = jnp.ones((MAX_HAND,), dtype=bool)

    # deck_ptr points to the next undrawn card (slot 8 after dealing the hand).
    deck_ptr = jnp.array(MAX_HAND, dtype=jnp.int32)

    # Python oracle initialises levels to all-ones (1-based, level 1 = base).
    # Confirmed from engine.py line: levels=tuple([1] * 12)
    levels = jnp.ones((N_HAND_TYPES,), dtype=jnp.int32)

    # Play-count trackers start at zero.
    hand_plays_run   = jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32)
    hand_plays_round = jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32)

    # PRNG key: fixed placeholder; real seeding added in Task 1.7.
    rng = jnp.array([0, 0], dtype=jnp.uint32)

    # Required-score lookup table for blind advances (Task 1.4). None -> placeholder
    # zeros (acceptable for any rollout that never crosses a blind boundary).
    if required_table is None:
        req_table = jnp.zeros((9, 3), dtype=jnp.int32)
    else:
        req_table = jnp.asarray(required_table, dtype=jnp.int32)

    return CoreState(
        deck_rank=dr,
        deck_suit=ds,
        deck_ptr=deck_ptr,

        hand_rank=hand_rank,
        hand_suit=hand_suit,
        hand_mask=hand_mask,

        ante=jnp.array(1, dtype=jnp.int32),
        blind_index=jnp.array(0, dtype=jnp.int32),
        round_score=jnp.array(0, dtype=jnp.int32),
        required=jnp.array(required, dtype=jnp.int32),
        hands_left=jnp.array(HANDS_PER_BLIND, dtype=jnp.int32),
        discards_left=jnp.array(DISCARDS_PER_BLIND, dtype=jnp.int32),
        hand_size=jnp.array(MAX_HAND, dtype=jnp.int32),
        required_table=req_table,

        money=jnp.array(STARTING_MONEY, dtype=jnp.int32),

        levels=levels,
        hand_plays_run=hand_plays_run,
        hand_plays_round=hand_plays_round,

        phase=jnp.array(Phase.PLAYING, dtype=jnp.int32),
        done=jnp.array(False, dtype=bool),
        won=jnp.array(False, dtype=bool),

        rng=rng,
    )


class StepSignals(NamedTuple):
    """Small per-step result struct for the reward/info layer.

    Fields:
        cleared:   True iff this PLAY pushed ``round_score >= required`` (blind
                   cleared). For DISCARD always False.
        won:       True iff this PLAY cleared the ANTE_MAX boss (blind_index 2) and
                   thereby won the run (phase -> WON). False otherwise.
        hand_type: HandType code (0..11) of the scored hand on a PLAY; 0 on DISCARD.
        score:     Chips scored by this PLAY; 0 on DISCARD.
    """

    cleared: jnp.ndarray   # bool[]
    won: jnp.ndarray       # bool[]
    hand_type: jnp.ndarray  # int32[]
    score: jnp.ndarray     # int32[]


def _compact_and_refill(state: CoreState, sel_mask, target):
    """Remove ``sel_mask`` cards, compact survivors to the front (preserving their
    original relative order), then refill empty front slots from the deck in deck
    order up to ``target`` cards held.

    Mirrors the Python oracle's ``_draw`` slot semantics exactly:
      * ``remaining = [c for i, c in enumerate(hand) if i not in chosen]`` keeps the
        kept (present, non-selected) cards in their original relative order.
      * ``_draw`` front-slices ``deck[:need]`` and appends them after the kept cards.

    Returns ``(hand_rank, hand_suit, hand_mask, deck_ptr)`` for the next state.
    ``target`` is the draw-up-to size (``hand_size``); refill draws
    ``min(need, available)`` cards (Python draws ``deck[:need]``, which silently
    truncates at the deck end).
    """
    kept_mask = state.hand_mask & ~sel_mask                      # bool[8]

    # --- stable compaction: kept cards keep their relative order, packed to front.
    # dest[i] = (#kept slots strictly before i) for kept slots; we scatter each kept
    # slot's card to that destination index. cumsum-1 over kept_mask gives that index.
    kept_i = kept_mask.astype(jnp.int32)
    dest = jnp.cumsum(kept_i) - 1                                # int32[8], valid where kept
    n_kept = jnp.sum(kept_i)                                     # int32[]

    # Scatter kept cards into a fresh, empty hand at their destination slots. Slots
    # that receive nothing stay empty (rank 0 / suit 0 / mask False).
    empty_rank = jnp.zeros((MAX_HAND,), dtype=jnp.int8)
    empty_suit = jnp.zeros((MAX_HAND,), dtype=jnp.int8)
    empty_mask = jnp.zeros((MAX_HAND,), dtype=bool)
    # Route non-kept slots to an out-of-bounds index (dropped by ``mode='drop'``).
    safe_dest = jnp.where(kept_mask, dest, MAX_HAND)
    comp_rank = empty_rank.at[safe_dest].set(state.hand_rank, mode="drop")
    comp_suit = empty_suit.at[safe_dest].set(state.hand_suit, mode="drop")
    comp_mask = empty_mask.at[safe_dest].set(True, mode="drop")

    # --- refill: draw deck[deck_ptr : deck_ptr+need] into slots [n_kept : n_kept+need].
    need = jnp.maximum(0, target - n_kept)                      # int32[]
    available = DECK_SIZE - state.deck_ptr                      # cards left in pile
    n_draw = jnp.minimum(need, available)                       # Python draws min(need, len(deck))

    # For each of the 8 hand slots, decide whether it is filled by a freshly-drawn
    # card. Slot s (0-based) is a draw target iff n_kept <= s < n_kept + n_draw; the
    # card it receives is deck[deck_ptr + (s - n_kept)].
    slot = jnp.arange(MAX_HAND, dtype=jnp.int32)
    draw_here = (slot >= n_kept) & (slot < n_kept + n_draw)     # bool[8]
    draw_src = state.deck_ptr + (slot - n_kept)                 # int32[8], valid where draw_here
    safe_src = jnp.clip(draw_src, 0, DECK_SIZE - 1)
    drawn_rank = state.deck_rank[safe_src]
    drawn_suit = state.deck_suit[safe_src]

    new_rank = jnp.where(draw_here, drawn_rank, comp_rank)
    new_suit = jnp.where(draw_here, drawn_suit, comp_suit)
    new_mask = jnp.where(draw_here, True, comp_mask)
    new_ptr = state.deck_ptr + n_draw

    return new_rank, new_suit, new_mask, new_ptr


def _gather_selected(state: CoreState, sel_mask):
    """Pack the selected (present) cards into 5-slot ranks/suits/mask for ``score_core``.

    The oracle scores ``selected = [hand[i] for i in idx]`` — the chosen cards in
    ASCENDING slot order. We replicate that: a stable left-pack of the slots where
    ``sel_mask & hand_mask`` is True into slots ``[0, n_sel)`` of a 5-wide buffer.
    (Legal actions never select more than MAX_SELECT, so 5 slots always suffice.)
    """
    pick_mask = sel_mask & state.hand_mask                      # bool[8]
    pick_i = pick_mask.astype(jnp.int32)
    dest = jnp.cumsum(pick_i) - 1                               # int32[8], valid where picked
    safe_dest = jnp.where(pick_mask, dest, MAX_SELECT)          # OOB -> dropped

    sel_rank = jnp.zeros((MAX_SELECT,), dtype=jnp.int8).at[safe_dest].set(
        state.hand_rank, mode="drop")
    sel_suit = jnp.zeros((MAX_SELECT,), dtype=jnp.int8).at[safe_dest].set(
        state.hand_suit, mode="drop")
    sel_present = jnp.zeros((MAX_SELECT,), dtype=bool).at[safe_dest].set(
        True, mode="drop")
    return sel_rank, sel_suit, sel_present


def step(state: CoreState, verb, sel_mask):
    """Apply a decoded PLAY or DISCARD to ``state`` (within-blind).

    Args:
        state:    current ``CoreState`` (phase PLAYING; ``done`` False).
        verb:     int (``Verb.PLAY`` / ``Verb.DISCARD``); the flat action id is
                  decoded to ``(verb, sel_mask)`` in the env adapter (Task 1.7).
        sel_mask: bool[8] slot-selection mask over the 8 hand slots.

    Returns:
        ``(next_state, StepSignals)``.

    Branchless (jit/vmap-able): both the PLAY and DISCARD successor states are
    computed unconditionally and selected with ``jnp.where`` per field. No Python
    control flow over traced values.

    SCOPE: DISCARD (+refill), PLAY (score / round_score / hands_left / refill), the
    LOSS terminal (non-clearing PLAY that drops hands_left to 0 -> phase LOST, done
    True, with the refilled hand), and the CLEAR boundary (round_score >= required
    after a PLAY):
      * WIN  — clearing the boss (blind_index 2) of ANTE_MAX: phase WON, done True,
               won True; the hand is left as-is (terminal).
      * ADVANCE — any other clear: roll to the next blind (boss clear -> next ante /
               small blind, else next blind same ante), reset round_score / required
               (from ``required_table``) / hands_left / discards_left / hand_size /
               hand_plays_round, and reshuffle the deck for standalone PPO use. (The
               oracle inserts a SHOP phase here that the JAX engine skips; the parity
               harness walks Python through the shop buying nothing to re-converge.)

    Branchless: the non-clear PLAY, ADVANCE, and WIN successors are all computed
    unconditionally and selected with ``jnp.where``/``tree_map`` on the {cleared, won}
    predicates; the DISCARD successor is selected against the whole PLAY successor.
    """
    verb = jnp.asarray(verb, dtype=jnp.int32)
    sel_mask = jnp.asarray(sel_mask, dtype=bool)
    is_play = verb == Verb.PLAY

    # ----------------------------------------------------------------- scoring
    # Score the selected cards (used only on the PLAY branch; cheap to compute
    # unconditionally for branchlessness).
    sel_rank, sel_suit, sel_present = _gather_selected(state, sel_mask)
    hand_type, _chips, _mult, score = score_core(
        sel_rank, sel_suit, sel_present, state.levels)

    round_score_after = state.round_score + score
    cleared = is_play & (round_score_after >= state.required)
    hands_left_after = state.hands_left - 1

    # ----------------------------------------------------- refill (PLAY / DISCARD)
    # Both verbs refill up to hand_size on the non-clearing path. The CLEAR path
    # must NOT refill (the new hand is dealt at the blind boundary, Task 1.4), so
    # we leave the hand untouched there.
    new_rank, new_suit, new_mask, new_ptr = _compact_and_refill(
        state, sel_mask, state.hand_size)

    # Per-HandType play counters bump for the played hand type (PLAY only). Shared by
    # every PLAY successor (non-clear / advance / win) — the played hand still counts.
    ht_onehot = (jnp.arange(N_HAND_TYPES) == hand_type).astype(jnp.int32)
    play_plays_run = state.hand_plays_run + ht_onehot
    play_plays_round = state.hand_plays_round + ht_onehot

    # Boundary predicates (mirror the oracle's _enter_cashout_or_win / _advance_blind).
    won = cleared & (state.ante >= ANTE_MAX) & (state.blind_index == 2)
    lost = (~cleared) & (hands_left_after <= 0)

    # ====================== non-clearing PLAY successor =======================
    # Refill, round_score += score, hands_left -= 1; LOSS if hands_left hits 0. This
    # successor is used only when ``cleared`` is False (advance/win override it below).
    noclear_phase = jnp.where(lost, jnp.int32(Phase.LOST), jnp.int32(Phase.PLAYING))
    noclear_state = state._replace(
        hand_rank=new_rank, hand_suit=new_suit, hand_mask=new_mask,
        deck_ptr=new_ptr,
        round_score=round_score_after, hands_left=hands_left_after,
        discards_left=state.discards_left,
        hand_plays_run=play_plays_run, hand_plays_round=play_plays_round,
        phase=noclear_phase, done=lost,
    )

    # =========================== ADVANCE successor ============================
    # The oracle clears -> cash-out -> SHOP -> _advance_blind on shop-leave. The JAX
    # engine collapses this to a direct advance (money/shop are out of scope here);
    # the parity harness walks Python through the shop buying nothing to re-converge.
    is_boss_clear = state.blind_index == 2
    new_blind = jnp.where(is_boss_clear, jnp.int32(0), state.blind_index + 1)
    new_ante = jnp.where(is_boss_clear, state.ante + 1, state.ante)
    # Next blind's target from the host-provided table (indexed [ante, blind_index]).
    new_required = state.required_table[new_ante, new_blind]

    # Reshuffle the deck for standalone PPO use (parity overrides this from Python).
    # Treat state.rng (uint32[2]) directly as a PRNG key.
    key = state.rng
    key, sub = jax.random.split(key)
    perm = jax.random.permutation(sub, DECK_SIZE)
    adv_deck_rank = state.deck_rank[perm]
    adv_deck_suit = state.deck_suit[perm]
    adv_hand_rank = adv_deck_rank[:MAX_HAND]
    adv_hand_suit = adv_deck_suit[:MAX_HAND]
    adv_hand_mask = jnp.ones((MAX_HAND,), dtype=bool)

    advance_state = state._replace(
        deck_rank=adv_deck_rank, deck_suit=adv_deck_suit,
        deck_ptr=jnp.array(MAX_HAND, dtype=jnp.int32),
        hand_rank=adv_hand_rank, hand_suit=adv_hand_suit, hand_mask=adv_hand_mask,
        ante=new_ante, blind_index=new_blind,
        round_score=jnp.int32(0), required=new_required,
        hands_left=jnp.int32(HANDS_PER_BLIND),
        discards_left=jnp.int32(DISCARDS_PER_BLIND),
        hand_size=jnp.int32(MAX_HAND),
        hand_plays_run=play_plays_run,                       # persists across blinds
        hand_plays_round=jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32),  # per-round reset
        phase=jnp.int32(Phase.PLAYING), done=jnp.asarray(False, dtype=bool),
        rng=key,
    )

    # ============================= WIN successor ==============================
    # Ante-8 boss cleared: terminal WON. Leave the hand as-is (no refill / advance);
    # only the counters and the bookkeeping flags change.
    win_state = state._replace(
        round_score=round_score_after,
        hands_left=hands_left_after,
        hand_plays_run=play_plays_run, hand_plays_round=play_plays_round,
        phase=jnp.int32(Phase.WON), done=jnp.asarray(True, dtype=bool),
        won=jnp.asarray(True, dtype=bool),
    )

    # Select the PLAY successor among {non-clear, advance, win} per field. ``won`` and
    # ``advance`` are mutually exclusive (won => not advance), so the order is unambiguous.
    play_clear_state = jax.tree_util.tree_map(
        lambda a, b: jnp.where(won, a, b), win_state, advance_state)
    play_state = jax.tree_util.tree_map(
        lambda a, b: jnp.where(cleared, a, b), play_clear_state, noclear_state)

    # ============================= DISCARD successor ==========================
    # Remove selected, refill, discards_left -= 1. No scoring, no clear/loss check.
    disc_state = state._replace(
        hand_rank=new_rank, hand_suit=new_suit, hand_mask=new_mask,
        deck_ptr=new_ptr,
        discards_left=state.discards_left - 1,
    )

    next_state = jax.tree_util.tree_map(
        lambda a, b: jnp.where(is_play, a, b), play_state, disc_state)

    signals = StepSignals(
        cleared=cleared,
        won=won,
        hand_type=jnp.where(is_play, hand_type, jnp.int32(0)),
        score=jnp.where(is_play, score, jnp.int32(0)),
    )
    return next_state, signals

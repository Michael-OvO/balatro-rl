"""Batch 6 STEP 1: GameState tracks how many times each HandType has been played.

`hand_plays_run` never resets; `hand_plays_round` resets at each blind boundary.
Both are length-12 tuples in HandType order (mirroring `levels`). In engine.step's
PLAY branch BOTH are incremented at [int(res.hand_type)] for the hand just played,
AFTER score_play + the joker on_play fold (so a scoring joker reads the PRE-increment
count via ScoreContext). This is engine plumbing (wiki-independent)."""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.state import Phase


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


# --- shape / initial conditions ---

def test_reset_initializes_zero_length_12_counters():
    s = reset(seed=1)
    assert s.hand_plays_run == tuple([0] * 12)
    assert s.hand_plays_round == tuple([0] * 12)
    assert len(s.hand_plays_run) == 12 and len(s.hand_plays_round) == 12


# --- increment in the PLAY branch ---

def _pair_hand():
    # pair of 3s + non-flush kickers
    return (C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0))


def test_play_increments_both_counters_at_hand_type_index():
    s = reset(seed=1)
    s = dataclasses.replace(s, hand=_pair_hand(), required=10_000_000)  # don't clear
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
    assert info["hand_type"] == int(HandType.PAIR)
    assert s2.hand_plays_run[int(HandType.PAIR)] == 1
    assert s2.hand_plays_round[int(HandType.PAIR)] == 1
    # nothing else moved
    assert sum(s2.hand_plays_run) == 1 and sum(s2.hand_plays_round) == 1


def test_play_accumulates_across_hands():
    s = reset(seed=1)
    s = dataclasses.replace(s, hand=_pair_hand(), required=10_000_000)
    for n in (1, 2, 3):
        s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
        s = dataclasses.replace(s, hand=_pair_hand(), hands_left=4)  # refill, keep playing
        assert s.hand_plays_run[int(HandType.PAIR)] == n
        assert s.hand_plays_round[int(HandType.PAIR)] == n


def test_different_hand_types_tracked_separately():
    s = reset(seed=1)
    s = dataclasses.replace(s, hand=_pair_hand(), required=10_000_000)
    s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))  # PAIR
    # now play a high card
    s = dataclasses.replace(s, hand=(C(9, 0), C(2, 1), C(5, 2)), hands_left=4)
    s, info = step(s, (Verb.PLAY, (0, 1, 2)))
    assert info["hand_type"] == int(HandType.HIGH_CARD)
    assert s.hand_plays_run[int(HandType.PAIR)] == 1
    assert s.hand_plays_run[int(HandType.HIGH_CARD)] == 1


def test_discard_does_not_increment():
    s = reset(seed=1)
    s = dataclasses.replace(s, hand=_pair_hand())
    s2, _ = step(s, (Verb.DISCARD, (0, 1)))
    assert s2.hand_plays_run == tuple([0] * 12)
    assert s2.hand_plays_round == tuple([0] * 12)


# --- round reset semantics ---

def test_round_counter_resets_at_blind_boundary_run_counter_persists():
    s = reset(seed=1)
    # clear the blind in one play so we route to the shop, then leave -> next blind.
    big = (C(13, 0), C(13, 1), C(13, 2), C(13, 3), C(2, 0), C(3, 0), C(4, 0), C(5, 0))
    s = dataclasses.replace(s, hand=big, required=10)
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))  # four of a kind
    assert s2.phase == Phase.SHOP
    ht = info["hand_type"]
    assert s2.hand_plays_run[ht] == 1
    assert s2.hand_plays_round[ht] == 1
    # leave the shop -> _advance_blind: round resets, run persists
    s3, _ = step(s2, (Verb.LEAVE_SHOP, 0))
    assert s3.hand_plays_round == tuple([0] * 12)   # round wiped
    assert s3.hand_plays_run[ht] == 1               # run kept


# --- determinism ---

def test_counters_are_deterministic():
    def run():
        s = reset(seed=2024)
        s = dataclasses.replace(s, hand=_pair_hand(), required=10_000_000)
        s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
        return s.hand_plays_run, s.hand_plays_round
    assert run() == run()


def test_step_purity_with_counters():
    s = reset(seed=42)
    s = dataclasses.replace(s, hand=_pair_hand(), required=10_000_000)
    a = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
    b = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
    assert a[0] == b[0] and a[1] == b[1]

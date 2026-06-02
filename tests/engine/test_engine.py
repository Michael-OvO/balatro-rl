from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, legal_actions, step
from balatro_rl.engine.state import Phase


def test_reset_initial_conditions():
    s = reset(seed=1)
    assert len(s.hand) == 8
    assert s.ante == 1 and s.blind_index == 0
    assert s.required == 300
    assert s.hands_left == 4 and s.discards_left == 3
    assert s.phase == Phase.PLAYING and not s.done


def test_reset_is_deterministic():
    assert reset(seed=5).hand == reset(seed=5).hand
    assert reset(seed=5).hand != reset(seed=6).hand


def test_legal_actions_present_and_bounded():
    s = reset(seed=1)
    acts = legal_actions(s)
    assert len(acts) > 0
    for verb, idx in acts:
        assert verb in (Verb.PLAY, Verb.DISCARD)
        assert 1 <= len(idx) <= 5
        assert len(set(idx)) == len(idx)


def test_discard_consumes_a_discard_and_refills_hand():
    s = reset(seed=1)
    s2, info = step(s, (Verb.DISCARD, (0, 1)))
    assert info["verb"] == "discard"
    assert s2.discards_left == s.discards_left - 1
    assert len(s2.hand) == 8


def test_play_consumes_a_hand_and_adds_score():
    s = reset(seed=1)
    s2, info = step(s, (Verb.PLAY, (0,)))
    assert info["verb"] == "play"
    assert s2.hands_left == s.hands_left - 1
    assert s2.round_score == info["score"]
    assert len(s2.hand) == 8


def test_clearing_a_blind_advances_and_resets_counters():
    # Force a clear by handing the engine a state already at the threshold-1
    # via a high-scoring play. Use a constructed hand of four-of-a-kind Kings.
    import dataclasses
    s = reset(seed=1)
    big_hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
                Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=big_hand, required=10)  # trivially clearable
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert info.get("cleared") is True
    assert s2.blind_index == 1            # advanced small -> big
    assert s2.round_score == 0            # reset for the new blind
    assert s2.hands_left == 4 and s2.discards_left == 3
    assert len(s2.hand) == 8


def test_losing_when_hands_exhausted_without_clearing():
    import dataclasses
    s = reset(seed=1)
    s = dataclasses.replace(s, hands_left=1, required=10_000_000, round_score=0)
    s2, info = step(s, (Verb.PLAY, (0,)))   # tiny score, can't clear, no hands left
    assert s2.done and not s2.won
    assert s2.phase == Phase.LOST


def test_winning_after_clearing_ante8_boss():
    import dataclasses
    s = reset(seed=1)
    s = dataclasses.replace(s, ante=8, blind_index=2, required=10,
                            hand=(Card(14, 0),) + reset(seed=1).hand[1:])
    s2, info = step(s, (Verb.PLAY, (0,)))
    assert s2.done and s2.won
    assert s2.phase == Phase.WON


def test_step_is_pure():
    s = reset(seed=42)
    s2a, info_a = step(s, (Verb.PLAY, (0,)))
    s2b, info_b = step(s, (Verb.PLAY, (0,)))
    assert s2a == s2b
    assert info_a == info_b

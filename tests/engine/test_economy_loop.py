import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, step
from balatro_rl.engine.state import Phase
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def _clearable(seed=1, **over):
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    return dataclasses.replace(s, hand=hand, required=10, **over)


def test_play_clear_shop_buy_leave_next_blind():
    s = _clearable(money=100, hands_left=2)
    s, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))     # clear Small -> shop
    assert s.phase == Phase.SHOP
    money_in_shop = s.money
    s, _ = step(s, (Verb.BUY, 0))                    # buy first offer
    assert len(s.jokers) == 1 and s.money < money_in_shop
    s, _ = step(s, (Verb.LEAVE_SHOP, 0))             # leave -> Big blind
    assert s.phase == Phase.PLAYING and s.blind_index == 1
    assert len(s.jokers) == 1                        # joker persists across the shop


def test_golden_joker_pays_out_at_cashout():
    s = _clearable(money=0, hands_left=1,
                   jokers=(JokerState(JokerType.GOLDEN_JOKER),))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))       # clear Small; 1 hand -> 0 left after play
    # money = 0 + reward(3) + interest(0) + hands(0 left) + golden(4) = 7
    assert s2.money == 7


def test_deterministic_full_run_is_reproducible():
    # Same seed + same scripted actions -> identical money/phase trajectory.
    def run():
        s = _clearable(seed=99, money=50, hands_left=1)
        s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
        s, _ = step(s, (Verb.REROLL, 0))
        s, _ = step(s, (Verb.LEAVE_SHOP, 0))
        return s
    a, b = run(), run()
    assert a.money == b.money and a.blind_index == b.blind_index
    assert a.shop_offers == b.shop_offers


def test_clearing_nonfinal_boss_advances_ante_after_shop():
    # Ante 3 Boss (not final) -> cash-out -> shop -> leave -> Ante 4 Small.
    s = _clearable(ante=3, blind_index=2, money=10, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))      # clear boss -> shop (ante<8, not a win)
    assert s2.phase == Phase.SHOP and s2.ante == 3 and s2.blind_index == 2
    s3, _ = step(s2, (Verb.LEAVE_SHOP, 0))          # leave -> next ante
    assert s3.phase == Phase.PLAYING and s3.ante == 4 and s3.blind_index == 0


def test_cavendish_self_destroys_during_cashout():
    # Force a sub-0.001 roll so Cavendish self-destroys at cash-out and is dropped.
    from balatro_rl.engine.rng import RNG
    seed = next(s for s in range(100000) if RNG.from_seed(s).random()[0] < 0.001)
    s = _clearable(money=10, hands_left=1, jokers=(JokerState(JokerType.CAVENDISH),))
    s = dataclasses.replace(s, rng=RNG.from_seed(seed))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))      # clear -> cash-out rolls Cavendish destroy
    assert s2.phase == Phase.SHOP
    assert all(j.type != JokerType.CAVENDISH for j in s2.jokers)

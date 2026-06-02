import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, legal_actions, step
from balatro_rl.engine.state import Phase
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def _clearable(seed=1, **over):
    """A state one play away from clearing, with a known big hand."""
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    return dataclasses.replace(s, hand=hand, required=10, **over)


def test_clearing_a_blind_enters_shop_and_pays_out():
    s = _clearable(money=10, hands_left=3)   # Small blind, $10 held, 3 hands left
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))  # four-of-a-kind kings, clears
    assert s2.phase == Phase.SHOP
    # cash-out = reward(small $3) + interest(10 -> $2) + leftover hands (2 left after play) $2 = $7
    assert s2.money == 10 + 3 + 2 + 2
    assert len(s2.shop_offers) == 2
    assert s2.blind_index == 0   # NOT advanced yet (advance happens on leave)


def test_shop_buy_adds_joker_and_spends():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))  # enter shop
    assert s2.phase == Phase.SHOP
    offer0 = s2.shop_offers[0]
    cost = __import__("balatro_rl.engine.shop", fromlist=["joker_cost"]).joker_cost(offer0.type)
    s3, info = step(s2, (Verb.BUY, 0))
    assert info["verb"] == "buy"
    assert s3.money == s2.money - cost
    assert s3.jokers[-1].type == offer0.type
    assert len(s3.shop_offers) == 1


def test_shop_sell_returns_money_and_frees_slot():
    s = _clearable(money=100, hands_left=1, jokers=(JokerState(JokerType.BARON),))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    s3, info = step(s2, (Verb.SELL, 0))   # sell Baron (cost 8 -> sell 4)
    assert info["verb"] == "sell" and info["value"] == 4
    assert s3.money == s2.money + 4
    assert s3.jokers == ()


def test_shop_reroll_costs_and_replaces_offers():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    before = s2.money
    s3, info = step(s2, (Verb.REROLL, 0))
    assert info["verb"] == "reroll" and info["cost"] == 5
    assert s3.money == before - 5
    assert s3.rerolls_done == 1
    s4, info2 = step(s3, (Verb.REROLL, 0))
    assert info2["cost"] == 6   # +1 each reroll


def test_shop_reorder_jokers():
    s = _clearable(money=10, hands_left=1,
                   jokers=(JokerState(JokerType.JOKER), JokerState(JokerType.BARON)))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    s3, info = step(s2, (Verb.REORDER, (0, 1)))   # move slot0 -> slot1
    assert info["verb"] == "reorder"
    assert [j.type for j in s3.jokers] == [JokerType.BARON, JokerType.JOKER]


def test_leave_shop_advances_to_next_blind():
    s = _clearable(money=10, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))   # cleared Small -> shop
    s3, info = step(s2, (Verb.LEAVE_SHOP, 0))
    assert s3.phase == Phase.PLAYING
    assert s3.blind_index == 1       # advanced to Big
    assert s3.round_score == 0 and s3.hands_left == 4 and len(s3.hand) == 8
    assert s3.shop_offers == () and s3.rerolls_done == 0


def test_clearing_ante8_boss_wins_no_shop():
    # _clearable already sets required=10; passing it again would duplicate the kwarg.
    s = _clearable(ante=8, blind_index=2, hands_left=1)
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert s2.done and s2.won and s2.phase == Phase.WON


def test_legal_actions_in_shop():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    verbs = {a[0] for a in legal_actions(s2)}
    assert Verb.LEAVE_SHOP in verbs
    assert Verb.BUY in verbs     # affordable offers exist
    assert Verb.REROLL in verbs

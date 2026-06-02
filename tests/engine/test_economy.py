from balatro_rl.engine.economy import blind_reward, interest, BLIND_REWARD


def test_blind_rewards():  # wiki: docs/reference/economy-shop.md §2
    assert blind_reward(0) == 3   # Small
    assert blind_reward(1) == 4   # Big
    assert blind_reward(2) == 5   # Boss
    assert BLIND_REWARD == (3, 4, 5)


def test_interest_rate_and_cap():  # +$1 per $5, cap $5 at $25
    assert interest(0) == 0
    assert interest(4) == 0
    assert interest(5) == 1
    assert interest(24) == 4
    assert interest(25) == 5
    assert interest(30) == 5
    assert interest(100) == 5


def test_interest_custom_cap():
    assert interest(100, cap=10) == 10   # Seed Money
    assert interest(40, cap=10) == 8

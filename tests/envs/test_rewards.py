import dataclasses
from balatro_rl.engine.engine import reset
from balatro_rl.engine.state import Phase
from balatro_rl.envs.rewards import make_reward, REWARD_NAMES


def test_reward_names():
    assert set(REWARD_NAMES) == {"win_ante8", "max_depth", "shaped"}


def test_win_ante8_terminal_signals():
    r = make_reward("win_ante8")
    s = reset(seed=1)
    won = dataclasses.replace(s, done=True, won=True, phase=Phase.WON)
    lost = dataclasses.replace(s, done=True, won=False, phase=Phase.LOST)
    assert r(s, 0, won, {"result": "won"}) > 0
    assert r(s, 0, lost, {"result": "lost"}) < 0
    assert r(s, 0, s, {}) == 0           # non-terminal, nothing happened


def test_max_depth_rewards_clearing():
    r = make_reward("max_depth")
    s = reset(seed=1)
    cleared = dataclasses.replace(s, blind_index=1)
    assert r(s, 0, cleared, {"cleared": True}) > 0


def test_shaped_is_potential_based_and_finite():
    r = make_reward("shaped")
    s = reset(seed=1)
    progressed = dataclasses.replace(s, round_score=150)   # required 300 -> halfway
    val = r(s, 0, progressed, {})
    assert val > 0 and abs(val) < 100      # bounded, positive for progress

import dataclasses

import numpy as np

from balatro_rl.engine.engine import reset
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.state import Phase
from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.rewards import (
    HandQuality,
    Shaped,
    ShapedScaled,
    make_reward,
    REWARD_NAMES,
)


def test_reward_names():
    assert set(REWARD_NAMES) == {"win_ante8", "max_depth", "shaped",
                                 "hand_quality", "hand_quality_q05", "shaped_scaled"}
    for name in REWARD_NAMES:
        r = make_reward(name)
        assert callable(r) and hasattr(r, "reset")
        r.reset()


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


def test_handquality_tier_is_strictly_monotone_and_bounded():
    tiers = [HandQuality._tier(h) for h in range(12)]
    assert tiers == sorted(tiers) and len(set(tiers)) == 12   # strictly increasing
    assert 0.0 < tiers[0] <= tiers[-1] <= 1.0
    assert abs(tiers[HandType.HIGH_CARD] - 0.2283) < 1e-3
    assert abs(tiers[HandType.FULL_HOUSE] - 0.6474) < 1e-3
    assert tiers[HandType.FLUSH_FIVE] == 1.0


def test_handquality_play_adds_tier_over_nonplay():
    r = make_reward("hand_quality")
    s = reset(seed=1)                       # prev == nxt -> shaping term identical
    base = r(s, 0, s, {"verb": "discard", "discarded": 2})
    pair = r(s, 0, s, {"verb": "play", "hand_type": int(HandType.PAIR)})
    flush = r(s, 0, s, {"verb": "play", "hand_type": int(HandType.FLUSH)})
    assert pair > base and flush > pair     # higher tier -> strictly larger reward
    assert abs((pair - base) - HandQuality._tier(int(HandType.PAIR))) < 1e-9


def test_handquality_milestones_add_clear_and_win():
    r = make_reward("hand_quality")
    s = reset(seed=1)
    play = {"verb": "play", "hand_type": int(HandType.STRAIGHT)}
    noclear = r(s, 0, s, play)
    clear = r(s, 0, s, {**play, "cleared": True})
    assert abs((clear - noclear) - 1.0) < 1e-9              # +1 for clearing
    won = dataclasses.replace(s, done=True, won=True, phase=Phase.WON)   # phi unchanged
    win_r = r(s, 0, won, {**play, "cleared": True})
    assert abs((win_r - clear) - 10.0) < 1e-9              # +10 for winning


def test_handquality_q05_is_exactly_half_tier():
    r1, r05 = make_reward("hand_quality"), make_reward("hand_quality_q05")
    s = reset(seed=1)
    play = {"verb": "play", "hand_type": int(HandType.THREE_OF_A_KIND)}
    base = {"verb": "discard"}
    gap1 = r1(s, 0, s, play) - r1(s, 0, s, base)
    gap05 = r05(s, 0, s, play) - r05(s, 0, s, base)
    assert abs(gap05 - 0.5 * gap1) < 1e-9


def test_shaped_scaled_scales_only_potential_not_milestones():
    s = reset(seed=1)
    prog = dataclasses.replace(s, round_score=150)         # required 300 -> halfway
    base, s10, s15 = Shaped(), ShapedScaled(scale=1.0), ShapedScaled(scale=1.5)
    assert abs(s10(s, 0, prog, {}) - base(s, 0, prog, {})) < 1e-9        # scale=1 == shaped
    assert abs(s15(s, 0, prog, {}) - 1.5 * base(s, 0, prog, {})) < 1e-9  # potential x1.5
    assert abs((s15(s, 0, prog, {"cleared": True}) - s15(s, 0, prog, {})) - 1.0) < 1e-9


def test_all_rewards_finite_on_random_rollout():
    rng = np.random.default_rng(0)
    for name in REWARD_NAMES:
        env = BalatroEnv(name)
        _obs, mask = env.reset(7)
        for _ in range(300):
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                _obs, mask = env.reset(7)
                continue
            _obs, rew, done, info, mask = env.step(int(rng.choice(legal)))
            assert np.isfinite(rew), (name, info)
            if done:
                _obs, mask = env.reset(7)

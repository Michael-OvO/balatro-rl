import numpy as np
from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.actions import NUM_ACTIONS
from balatro_rl.envs.obs import OBS_SHAPES


def test_reset_returns_obs_and_mask():
    env = BalatroEnv(reward_name="shaped")
    obs, mask = env.reset(seed=1)
    assert set(obs.keys()) == set(OBS_SHAPES.keys())
    assert mask.shape == (NUM_ACTIONS,) and mask.dtype == np.bool_
    assert mask.any()


def test_step_advances_and_returns_contract():
    env = BalatroEnv(reward_name="shaped")
    _, mask = env.reset(seed=1)
    a = int(np.flatnonzero(mask)[0])
    obs, reward, done, info, mask2 = env.step(a)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert mask2.shape == (NUM_ACTIONS,)
    assert "verb" in info


def test_illegal_action_rejected():
    import pytest
    env = BalatroEnv(reward_name="shaped")
    _, mask = env.reset(seed=1)
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(AssertionError):
        env.step(illegal)


def test_episode_runs_to_done_with_masked_random():
    env = BalatroEnv(reward_name="max_depth")
    _, mask = env.reset(seed=3)
    from balatro_rl.engine.rng import RNG
    chooser = RNG.from_seed(123)
    done = False
    steps = 0
    while not done and steps < 5000:
        legal = np.flatnonzero(mask)
        idx, chooser = chooser.randint(0, len(legal) - 1)
        _, _, done, _, mask = env.step(int(legal[idx]))
        steps += 1
    assert done

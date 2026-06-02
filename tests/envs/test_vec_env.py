import numpy as np
from balatro_rl.envs.vec_env import SyncVectorEnv
from balatro_rl.envs.actions import NUM_ACTIONS
from balatro_rl.envs.obs import OBS_SHAPES


def test_reset_batches_obs_and_mask():
    venv = SyncVectorEnv(num_envs=4, reward_name="shaped", base_seed=0)
    obs, mask = venv.reset()
    assert mask.shape == (4, NUM_ACTIONS) and mask.dtype == bool
    for k, shape in OBS_SHAPES.items():
        assert obs[k].shape == (4,) + shape, k
    assert mask.any(axis=1).all()           # every env has at least one legal action


def test_step_shapes_and_autoreset():
    venv = SyncVectorEnv(num_envs=4, reward_name="max_depth", base_seed=1)
    obs, mask = venv.reset()
    for _ in range(50):
        actions = np.array([int(np.flatnonzero(mask[i])[0]) for i in range(4)])
        obs, rewards, dones, infos, mask = venv.step(actions)
        assert rewards.shape == (4,) and dones.shape == (4,)
        assert mask.shape == (4, NUM_ACTIONS)
        # after a done, the env auto-resets and still offers legal actions
        assert mask.any(axis=1).all()

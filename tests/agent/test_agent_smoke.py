import numpy as np
from balatro_rl.agent.spec import dummy_obs, OBS_DTYPES
from balatro_rl.envs.obs import OBS_SHAPES


def test_jax_imports():
    import jax, flax, optax  # noqa: F401
    assert jax.numpy.add(1, 2) == 3


def test_dummy_obs_matches_obs_spec():
    obs = dummy_obs(batch=4)
    assert set(obs.keys()) == set(OBS_SHAPES.keys())
    for k, shape in OBS_SHAPES.items():
        assert obs[k].shape == (4,) + shape, k
    assert obs["joker_types"].dtype == np.int32
    assert obs["global"].dtype == np.float32

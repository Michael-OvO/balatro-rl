def test_numpy_and_envs_import():
    import numpy as np
    import balatro_rl.envs  # noqa: F401
    assert np.__version__

"""JAX-native, GPU-vectorizable reimplementation of the Balatro CORE game loop.

Phase 1 (deal -> play/discard -> score -> blind/ante progression -> win/lose; no
jokers/shop/consumables/vouchers/bosses), proven bit-for-bit equal to the Python
engine (`balatro_rl.engine`) by the parity gate in `tests/engine_jax/`.

Import the submodules directly, e.g.::

    from balatro_rl.engine_jax import step as J          # reset, step, batched_step, ...
    from balatro_rl.engine_jax.state import CoreState
    from balatro_rl.engine_jax.obs import encode_core, legal_mask_core
    from balatro_rl.engine_jax.rewards import shaped_core

`balatro_rl.envs.jax_vec_env.JaxVectorEnv` wraps this engine as a SyncVectorEnv-compatible
vectorized env for the PPO trainer (`TrainConfig.engine="jax"`).
"""

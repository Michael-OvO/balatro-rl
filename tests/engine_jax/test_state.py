import jax, jax.numpy as jnp
from balatro_rl.engine_jax.state import CoreState, zeros_state


def test_is_pytree_and_vmappable():
    s = zeros_state()
    leaves = jax.tree_util.tree_leaves(s)
    assert all(isinstance(x, jnp.ndarray) for x in leaves)
    # batchable: stack 4 states along a new leading axis, still a valid pytree
    b = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 4), s)
    assert b.hand_rank.shape == (4, 8)


def test_field_shapes():
    s = zeros_state()
    assert s.deck_rank.shape == (52,) and s.hand_rank.shape == (8,)
    assert s.levels.shape == (12,) and s.round_score.shape == ()


def test_corestate_has_jokers_field():
    from balatro_rl.engine_jax.state import zeros_state
    import jax.numpy as jnp
    from balatro_rl.envs.actions import MAX_JOKERS
    s = zeros_state()
    assert hasattr(s, "jokers")
    assert s.jokers.shape == (MAX_JOKERS,)
    assert s.jokers.dtype == jnp.int32
    assert int(jnp.sum(s.jokers)) == 0  # empty by default

"""Parity test: JAX reset must produce the same initial state as the Python oracle.

This is the Phase 0 gate: proves the JAX engine can be seeded identically to
the Python engine and produces equal initial state.
"""
from balatro_rl.engine import engine
from balatro_rl.engine_jax import step as J
from tests.engine_jax.parity_util import (
    deck_from_python,
    python_core_fields,
    jax_core_fields,
    assert_states_equal,
)


def test_reset_matches_python():
    gs = engine.reset(0, 0.2, None, False)
    ranks, suits = deck_from_python(gs)
    cs = J.reset(ranks, suits, required=gs.required)
    assert_states_equal(python_core_fields(gs), jax_core_fields(cs))

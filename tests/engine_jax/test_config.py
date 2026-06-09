import jax.numpy as jnp
from balatro_rl.engine_jax import config as C
from balatro_rl.envs.actions import MAX_HAND, MAX_SELECT, NUM_ACTIONS


def test_constants_mirror_python():
    assert C.MAX_HAND == MAX_HAND == 8
    assert C.MAX_SELECT == MAX_SELECT == 5
    assert C.NUM_ACTIONS == NUM_ACTIONS == 708
    assert C.HANDS_PER_BLIND == 4 and C.DISCARDS_PER_BLIND == 3


def test_score_tables_shapes():
    assert C.HAND_BASE_CHIPS.shape == (12,) and C.HAND_BASE_MULT.shape == (12,)
    assert C.HAND_INC_CHIPS.shape == (12,) and C.HAND_INC_MULT.shape == (12,)
    # spot-check the table (PAIR=1): base (10,2), inc (15,1)
    assert int(C.HAND_BASE_CHIPS[1]) == 10 and int(C.HAND_BASE_MULT[1]) == 2
    assert int(C.HAND_INC_CHIPS[1]) == 15 and int(C.HAND_INC_MULT[1]) == 1

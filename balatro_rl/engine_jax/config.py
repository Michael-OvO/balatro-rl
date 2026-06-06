"""Global constants and scoring tables for the JAX engine.

All values are derived from / verified against the Python oracle in
balatro_rl/engine/hands.py and balatro_rl/engine/state.py.

HandType order (matches HandType IntEnum in engine/hands.py):
    0  HIGH_CARD
    1  PAIR
    2  TWO_PAIR
    3  THREE_OF_A_KIND
    4  STRAIGHT
    5  FLUSH
    6  FULL_HOUSE
    7  FOUR_OF_A_KIND
    8  STRAIGHT_FLUSH
    9  FIVE_OF_A_KIND
    10 FLUSH_HOUSE
    11 FLUSH_FIVE
"""
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Action-space / hand-size constants — must match envs/actions.py
# ---------------------------------------------------------------------------
MAX_HAND: int = 8      # maximum cards held at once
MAX_SELECT: int = 5    # maximum cards selectable per play/discard
NUM_ACTIONS: int = 708  # flat action-space size

# ---------------------------------------------------------------------------
# Per-blind limits (base game; vouchers extend discards at the engine level)
# ---------------------------------------------------------------------------
HANDS_PER_BLIND: int = 4
DISCARDS_PER_BLIND: int = 3

# Highest ante; clearing the boss (blind_index 2) of ANTE_MAX wins the run.
ANTE_MAX: int = 8

# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------
STARTING_MONEY: int = 4

# ---------------------------------------------------------------------------
# Hand-type count
# ---------------------------------------------------------------------------
N_HAND_TYPES: int = 12

# ---------------------------------------------------------------------------
# Phase constants (match Phase IntEnum in engine/state.py)
# ---------------------------------------------------------------------------
class Phase:
    PLAYING: int = 0
    WON: int = 1
    LOST: int = 2


# ---------------------------------------------------------------------------
# Verb constants (match Verb IntEnum in engine/engine.py)
# ---------------------------------------------------------------------------
class Verb:
    PLAY: int = 0
    DISCARD: int = 1


# ---------------------------------------------------------------------------
# Scoring tables — indexed by HandType int value (0..11)
#
# A hand scored at level L earns:
#   chips = HAND_BASE_CHIPS[ht] + HAND_INC_CHIPS[ht] * (L - 1)
#   mult  = HAND_BASE_MULT[ht]  + HAND_INC_MULT[ht]  * (L - 1)
#
# Source: balatro_rl/engine/hands.py HAND_BASE / HAND_LEVEL_INC dicts.
# ---------------------------------------------------------------------------
HAND_BASE_CHIPS = jnp.array([5, 10, 20, 30, 30, 35, 40, 60, 100, 120, 140, 160], dtype=jnp.int32)
HAND_BASE_MULT  = jnp.array([1,  2,  2,  3,  4,  4,  4,  7,   8,  12,  14,  16], dtype=jnp.int32)
HAND_INC_CHIPS  = jnp.array([10, 15, 20, 20, 30, 15, 25, 30,  40,  35,  40,  50], dtype=jnp.int32)
HAND_INC_MULT   = jnp.array([1,   1,  1,  2,  3,  2,  2,  3,   4,   3,   4,   3], dtype=jnp.int32)

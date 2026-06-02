"""Play a deterministic random game: `python -m balatro_rl.engine [seed]`.

Uses a SEPARATE RNG (seeded from the game seed) to choose actions, so the run is
fully reproducible without touching the engine's own RNG stream.
"""
from __future__ import annotations

import sys

from .engine import legal_actions, reset, step
from .render import render
from .rng import RNG
from .state import GameState

_MAX_STEPS = 10_000


def play_random(seed: int = 0, verbose: bool = True) -> GameState:
    state = reset(seed)
    chooser = RNG.from_seed(seed ^ 0xABCDEF)
    for _ in range(_MAX_STEPS):
        if state.done:
            break
        acts = legal_actions(state)
        i, chooser = chooser.randint(0, len(acts) - 1)
        state, info = step(state, acts[i])
        if verbose and info.get("verb") == "play":
            print(render(state))
    return state


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    final = play_random(seed)
    print("RESULT:", "WON" if final.won else "LOST", "| ante", final.ante)

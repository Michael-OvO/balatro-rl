"""Minimal ASCII render of a GameState — the seed of the Plan-4 replay viewer."""
from __future__ import annotations

from .cards import card_str
from .state import GameState


def render(state: GameState) -> str:
    hand = " ".join(card_str(c) for c in state.hand)
    head = (f"Ante {state.ante} blind {state.blind_index}  "
            f"score {state.round_score}/{state.required}  "
            f"hands {state.hands_left} discards {state.discards_left}  ${state.money}")
    return f"{head}\nhand: {hand}"

"""Base scoring pipeline: hand base value + scoring-card chips, then chips x mult.

Tier-0 only — no jokers, held cards, enhancements, editions, seals, or hand
levels (all level 1). Later plans extend THIS function's pipeline; the structured
ScoreResult is what the replay viewer renders as a score breakdown.
"""
from __future__ import annotations

import dataclasses

from .cards import Card, rank_chip_value
from .hands import HAND_BASE, HandType, evaluate


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreResult:
    score: int
    hand_type: HandType
    chips: int
    mult: int
    scoring_idx: tuple[int, ...]


def score_play(cards: list[Card]) -> ScoreResult:
    hand_type, scoring_idx = evaluate(cards)
    base_chips, base_mult = HAND_BASE[hand_type]
    chips = base_chips + sum(rank_chip_value(cards[i].rank) for i in scoring_idx)
    mult = base_mult
    return ScoreResult(
        score=chips * mult,
        hand_type=hand_type,
        chips=chips,
        mult=mult,
        scoring_idx=scoring_idx,
    )

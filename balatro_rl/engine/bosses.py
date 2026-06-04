"""Boss blinds (Phase C). This module is the boss BACKBONE: the BossEffect enum, each
boss's metadata (min ante, score-requirement multiplier, finisher flag), and selection.

The boss EFFECTS (card debuffs, legal-action restrictions, draw changes) are layered in
later sub-phases (C1 scoring, C2 legal-mask, C3 draw/state) and compose with the scoring
pipeline's `debuffed_idx` (already wired in Phase B). Selection is gated behind the engine's
`enable_bosses` flag so the default game stays byte-identical and the validated ante-7 agent
keeps running until the final retrain enables and exposes bosses.

Boss table verified against https://balatrowiki.org/w/Blinds (23 regular + 5 finishers).
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

from .hands import evaluate, is_face


class BossEffect(IntEnum):
    NONE = 0
    # --- 23 regular bosses (antes 1-7; "Any" min-ante = 1) ---
    THE_HOOK = 1
    THE_OX = 2
    THE_HOUSE = 3
    THE_WALL = 4
    THE_WHEEL = 5
    THE_ARM = 6
    THE_CLUB = 7
    THE_FISH = 8
    THE_PSYCHIC = 9
    THE_GOAD = 10
    THE_WATER = 11
    THE_WINDOW = 12
    THE_MANACLE = 13
    THE_EYE = 14
    THE_MOUTH = 15
    THE_PLANT = 16
    THE_SERPENT = 17
    THE_PILLAR = 18
    THE_NEEDLE = 19
    THE_HEAD = 20
    THE_TOOTH = 21
    THE_FLINT = 22
    THE_MARK = 23
    # --- 5 finisher (showdown) bosses (ante 8 only) ---
    AMBER_ACORN = 24
    VERDANT_LEAF = 25
    VIOLET_VESSEL = 26
    CRIMSON_HEART = 27
    CERULEAN_BELL = 28


@dataclasses.dataclass(frozen=True, slots=True)
class BossInfo:
    """min_ante: earliest ante this boss can appear. req_mult: score-requirement multiplier
    on the ante base chips (most bosses 2x; Wall 4x, Needle 1x, Violet Vessel 6x).
    is_finisher: a showdown boss that only appears on ante 8."""
    min_ante: int
    req_mult: float
    is_finisher: bool = False


# Default boss multiplier is 2.0 (the standard boss blind). NONE carries 2.0 so a boss
# blind with no boss selected (bosses disabled) keeps the pre-C0 2x requirement -> the
# default game is byte-identical.
_DEFAULT_MULT = 2.0
BOSS_INFO: dict[BossEffect, BossInfo] = {
    BossEffect.NONE: BossInfo(1, _DEFAULT_MULT),
    BossEffect.THE_HOOK: BossInfo(1, 2.0),
    BossEffect.THE_OX: BossInfo(6, 2.0),
    BossEffect.THE_HOUSE: BossInfo(2, 2.0),
    BossEffect.THE_WALL: BossInfo(2, 4.0),
    BossEffect.THE_WHEEL: BossInfo(2, 2.0),
    BossEffect.THE_ARM: BossInfo(2, 2.0),
    BossEffect.THE_CLUB: BossInfo(1, 2.0),
    BossEffect.THE_FISH: BossInfo(2, 2.0),
    BossEffect.THE_PSYCHIC: BossInfo(1, 2.0),
    BossEffect.THE_GOAD: BossInfo(1, 2.0),
    BossEffect.THE_WATER: BossInfo(2, 2.0),
    BossEffect.THE_WINDOW: BossInfo(1, 2.0),
    BossEffect.THE_MANACLE: BossInfo(1, 2.0),
    BossEffect.THE_EYE: BossInfo(3, 2.0),
    BossEffect.THE_MOUTH: BossInfo(2, 2.0),
    BossEffect.THE_PLANT: BossInfo(4, 2.0),
    BossEffect.THE_SERPENT: BossInfo(5, 2.0),
    BossEffect.THE_PILLAR: BossInfo(1, 2.0),
    BossEffect.THE_NEEDLE: BossInfo(2, 1.0),
    BossEffect.THE_HEAD: BossInfo(1, 2.0),
    BossEffect.THE_TOOTH: BossInfo(3, 2.0),
    BossEffect.THE_FLINT: BossInfo(2, 2.0),
    BossEffect.THE_MARK: BossInfo(2, 2.0),
    BossEffect.AMBER_ACORN: BossInfo(8, 2.0, is_finisher=True),
    BossEffect.VERDANT_LEAF: BossInfo(8, 2.0, is_finisher=True),
    BossEffect.VIOLET_VESSEL: BossInfo(8, 6.0, is_finisher=True),
    BossEffect.CRIMSON_HEART: BossInfo(8, 2.0, is_finisher=True),
    BossEffect.CERULEAN_BELL: BossInfo(8, 2.0, is_finisher=True),
}


def boss_req_mult(boss: BossEffect) -> float:
    """Score-requirement multiplier for a boss (2.0 default; Wall 4, Needle 1, Vessel 6)."""
    return BOSS_INFO[boss].req_mult


def is_finisher(boss: BossEffect) -> bool:
    """Whether this is a showdown/finisher boss (ante 8/16/...; pays $8 at cash-out)."""
    return BOSS_INFO[boss].is_finisher


def eligible_bosses(ante: int) -> list[BossEffect]:
    """The selectable boss pool for an ante: finishers on ante 8 (every 8th ante), else the
    regular bosses whose min_ante has been reached. NONE is never selectable."""
    if ante % 8 == 0:
        return [b for b in BossEffect if BOSS_INFO[b].is_finisher]
    return [b for b in BossEffect
            if b != BossEffect.NONE and not BOSS_INFO[b].is_finisher
            and BOSS_INFO[b].min_ante <= ante]


# --- scoring effects (Phase C1): card debuffs + The Flint base halving -----------
# Suit encoding matches the rest of the engine: 0=Spade, 1=Heart, 2=Club, 3=Diamond.
_DEBUFF_SUIT: dict[BossEffect, int] = {
    BossEffect.THE_GOAD: 0,    # Spades
    BossEffect.THE_HEAD: 1,    # Hearts
    BossEffect.THE_CLUB: 2,    # Clubs
    BossEffect.THE_WINDOW: 3,  # Diamonds
}


def boss_debuffed_idx(boss: BossEffect, played, rules) -> tuple:
    """Played-hand indices the boss DEBUFFS (suit bosses by suit; The Plant by face card).
    A debuffed card scores nothing and triggers nothing (the scoring pipeline owns the
    skip via debuffed_idx) but still forms the poker hand. `rules` binds face detection, so
    Pareidolia makes The Plant debuff every card. Empty for bosses with no card debuff."""
    suit = _DEBUFF_SUIT.get(boss)
    if suit is not None:
        return tuple(i for i, c in enumerate(played) if c.suit == suit)
    if boss == BossEffect.THE_PLANT:
        return tuple(i for i, c in enumerate(played) if is_face(c, rules))
    return ()


def boss_halves_base(boss: BossEffect) -> bool:
    """The Flint halves the played hand's base Chips and Mult (rounded up in score_play)."""
    return boss == BossEffect.THE_FLINT


# --- blind-setup + legal-mask effects (Phase C2) ---------------------------------

def boss_hand_size_delta(boss: BossEffect) -> int:
    """Change to the blind's hand size (The Manacle: -1)."""
    return -1 if boss == BossEffect.THE_MANACLE else 0


def boss_hands_left(boss: BossEffect, default: int) -> int:
    """Hands available this blind (The Needle: 1)."""
    return 1 if boss == BossEffect.THE_NEEDLE else default


def boss_discards_left(boss: BossEffect, default: int) -> int:
    """Discards available this blind (The Water: 0)."""
    return 0 if boss == BossEffect.THE_WATER else default


def boss_filters_plays(boss: BossEffect) -> bool:
    """Whether the boss restricts which PLAY actions are legal (Psychic/Eye/Mouth). Lets
    legal_actions skip the per-combo hand evaluation entirely off these blinds."""
    return boss in (BossEffect.THE_PSYCHIC, BossEffect.THE_EYE, BossEffect.THE_MOUTH)


def boss_allows_play(boss: BossEffect, combo_cards, hand_plays_round, rules) -> bool:
    """Whether a PLAY of `combo_cards` is legal under the boss. The Psychic requires
    exactly 5 cards. The Eye forbids a hand type already played this round; The Mouth
    allows only the round's first-played type (any type if none played yet). Both read
    hand_plays_round (per-HandType counts this round). Non-filtering bosses allow all."""
    if boss == BossEffect.THE_PSYCHIC:
        return len(combo_cards) == 5
    if boss in (BossEffect.THE_EYE, BossEffect.THE_MOUTH):
        ht = int(evaluate(list(combo_cards), rules)[0])
        played = hand_plays_round[ht] if ht < len(hand_plays_round) else 0
        if boss == BossEffect.THE_EYE:
            return played == 0                      # no repeats this round
        return sum(hand_plays_round) == 0 or played > 0   # Mouth: locked to the first type
    return True


# --- draw/state effects (Phase C3) -----------------------------------------------

def boss_tooth_cost(boss: BossEffect, n_played: int) -> int:
    """Money lost this PLAY: The Tooth charges $1 per card played (money may go negative)."""
    return n_played if boss == BossEffect.THE_TOOTH else 0


def boss_ox_zeroes_money(boss: BossEffect, played_ht: int, hand_plays_run) -> bool:
    """The Ox sets money to $0 when you play your most-played hand type. Uses the run play
    history (pre-increment of this hand); triggers if the played type is (tied for) the max
    and has actually been played. Approximation of the wiki's ante-entry snapshot (negligible
    while the agent is boss-blind; refine at the retrain if it matters)."""
    if boss != BossEffect.THE_OX or not hand_plays_run:
        return False
    top = max(hand_plays_run)
    played = hand_plays_run[played_ht] if played_ht < len(hand_plays_run) else 0
    return top > 0 and played == top


def boss_draw_target(boss: BossEffect, remaining_len: int, hand_size: int) -> int:
    """Target hand size for the post-action redraw. The Serpent draws exactly 3 (capped at
    hand_size so the agent's fixed 8-slot encoding stays valid); every other boss refills to
    hand_size as usual."""
    if boss == BossEffect.THE_SERPENT:
        return min(hand_size, remaining_len + 3)
    return hand_size


def boss_hook_discard(remaining, rng, k: int = 2):
    """The Hook: remove k random cards from the held `remaining` cards (before the redraw),
    preserving the order of the survivors. Returns (kept_list, rng). Discards all if fewer
    than k are held. Draws one rng value (an index shuffle) so a fixed seed is deterministic."""
    remaining = list(remaining)
    n = len(remaining)
    if n <= k:
        return [], rng
    order, rng = rng.shuffle(list(range(n)))
    removed = set(order[:k])
    return [c for j, c in enumerate(remaining) if j not in removed], rng


def select_boss(rng, ante: int):
    """Pick a boss uniformly from the ante's eligible pool. Returns (BossEffect, rng).
    Draws exactly one rng value (the only perturbation of the deterministic stream), so a
    fixed seed reproduces the same boss. The engine calls this only when bosses are enabled
    AND the new blind is the boss blind, so a disabled game draws zero extra rng."""
    pool = eligible_bosses(ante)
    idx, rng = rng.randint(0, len(pool) - 1)
    return pool[idx], rng

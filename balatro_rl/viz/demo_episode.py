"""Generate a scripted DEMO episode that showcases the enriched viewer: enhanced/edition/
seal cards, a boss blind (with a debuffed card), jokers (with effect descriptions), the
step-by-step score breakdown, and a consumable (USE a Planet to level a hand).

The training env doesn't expose mods yet (acquisition is a stopgap), so this hand-scripts a
rich game purely through the deterministic engine -- no trained agent needed -- and writes a
viewer-loadable episode JSON. Run: `python -m balatro_rl.viz.demo_episode`.
"""
from __future__ import annotations

import dataclasses
import os

from ..engine.bosses import BossEffect
from ..engine.cards import Card, Edition, Enhancement, Seal
from ..engine.consumables import PlanetType, planet
from ..engine.engine import Verb, explain_play, reset, step
from ..engine.jokers.base import JokerState, JokerType
from ..envs.actions import encode_action
from .replay_data import (
    _boss_d, _card_d, _consum_d, _joker_d, _offer_d, action_label, render_board, save_episode,
)
from ..engine.state import Phase

_PHASE = {0: "PLAYING", 1: "WON", 2: "LOST", 3: "SHOP"}


def _mk_step(t: int, state, action_id: int, verb, arg, info, score_trace, reward) -> dict:
    """Assemble one viewer step dict (same schema as replay_data.record_agent_episode)."""
    selected = list(arg) if verb in (Verb.PLAY, Verb.DISCARD) else []
    return {
        "t": t, "ante": int(state.ante), "blind": int(state.blind_index),
        "phase": _PHASE.get(int(state.phase)), "money": int(state.money),
        "board": render_board(state), "action_id": action_id, "action_label": action_label(action_id),
        "reward": float(reward), "value": 0.0,
        "score": info.get("score"), "hand_type": info.get("hand_type"),
        "chips": info.get("chips"), "mult": info.get("mult"), "top_probs": [],
        "schema": 2, "verb": verb.name, "selected": selected,
        "hand": [_card_d(c) for c in state.hand],
        "scoring_idx": list(info.get("scoring_idx", [])),
        "round_score": int(state.round_score), "required": int(state.required),
        "hands_left": int(state.hands_left), "discards_left": int(state.discards_left),
        "jokers": [_joker_d(j) for j in state.jokers],
        "shop_offers": ([_offer_d(o) for o in state.shop_offers]
                        if int(state.phase) == int(Phase.SHOP) else []),
        "hand_reset": False, "earned": info.get("earned"),
        "boss": _boss_d(state), "consumables": [_consum_d(c) for c in state.consumables],
        "score_trace": score_trace,
    }


def build_demo_episode() -> list[dict]:
    """A short scripted episode: PLAY a modded pair under The Club, USE a Planet, PLAY again."""
    # Rich starting state (high `required` so the blind never clears -> stays in play).
    st = dataclasses.replace(
        reset(0),
        hand=(Card(13, 2, enhancement=Enhancement.GLASS),   # King of Clubs (DEBUFFED by The Club)
              Card(13, 1, enhancement=Enhancement.GLASS),   # King of Hearts (scores, Glass X2)
              Card(7, 0, enhancement=Enhancement.BONUS),    # 7 of Spades, Bonus +30
              Card(5, 3, seal=Seal.GOLD),                   # 5 of Diamonds, Gold seal
              Card(9, 0, edition=Edition.FOIL),             # 9 of Spades, Foil
              Card(2, 1), Card(3, 2), Card(4, 3)),
        jokers=(JokerState(type=JokerType.JOKER),           # +4 Mult
                JokerState(type=JokerType.SCARY_FACE),      # +30 Chips per scored face
                JokerState(type=JokerType.GREEDY)),         # +3 Mult per scored Diamond
        boss=int(BossEffect.THE_CLUB),                      # debuffs all Clubs
        consumables=(planet(PlanetType.MERCURY),),          # +1 level to Pair
        required=10_000_000, hands_left=4)

    steps: list[dict] = []
    play_id = encode_action(Verb.PLAY, (0, 1))
    use_id = encode_action(Verb.USE, 0)
    # 1) PLAY the pair of Kings -- one Club (debuffed, inert), one Heart (scores w/ Glass X2).
    trace = explain_play(st, (0, 1))["trace"]
    nxt, info = step(st, (Verb.PLAY, (0, 1)))
    steps.append(_mk_step(0, st, play_id, Verb.PLAY, (0, 1), info, trace, 0.0))

    # 2) USE Mercury -> levels Pair to 2 (consumables panel + level state).
    st2 = nxt
    nxt2, info2 = step(st2, (Verb.USE, 0))
    steps.append(_mk_step(1, st2, use_id, Verb.USE, 0, info2, [], 0.0))

    # 3) PLAY a pair again at level 2 -- the breakdown base now shows "(lvl 2)".
    st3 = dataclasses.replace(
        nxt2, hand=(Card(12, 0), Card(12, 1), Card(7, 0, enhancement=Enhancement.BONUS),
                    Card(5, 3, seal=Seal.GOLD), Card(9, 0, edition=Edition.FOIL),
                    Card(2, 1), Card(3, 2), Card(4, 3)))
    trace3 = explain_play(st3, (0, 1))["trace"]
    _nxt3, info3 = step(st3, (Verb.PLAY, (0, 1)))
    steps.append(_mk_step(2, st3, play_id, Verb.PLAY, (0, 1), info3, trace3, 0.0))
    return steps


def main():
    out_dir = os.environ.get("BALATRO_EPISODE_DIR", "/tmp/sweep_out")
    os.makedirs(out_dir, exist_ok=True)
    # NOTE: the viewer's picker only lists files matching *.episode.json (see
    # viewer.list_episodes), so the demo MUST use that suffix to appear in the dropdown.
    path = os.path.join(out_dir, "demo_rich_content.episode.json")
    save_episode(build_demo_episode(), path)
    print(f"wrote demo episode -> {path}")


if __name__ == "__main__":
    main()

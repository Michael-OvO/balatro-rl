"""Replay data: reconstruct states from (seed, actions), render a text board, and
record a per-step episode (board + action + reward + value + top action-probs) for
the Gradio viewer. Pure/testable; the engine's determinism makes replay exact.
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from ..agent.value_head import value_decode
from ..engine import engine
from ..engine.bosses import BossEffect
from ..engine.cards import card_str
from ..engine.descriptions import (
    boss_desc, boss_name, consumable_desc, consumable_name, joker_desc, joker_name,
    pack_desc, voucher_desc, voucher_name,
)
from ..engine.engine import Verb, explain_play
from ..engine.shop import sell_value
from ..engine.state import GameState, Phase
from ..envs.actions import decode
from ..envs.balatro_env import BalatroEnv

_PHASE = {0: "PLAYING", 1: "WON", 2: "LOST", 3: "SHOP", 4: "OPEN_PACK"}
_MAX_STEPS = 3000


def _card_d(c) -> dict:
    """Serialize a Card for the viewer (enh/ed/seal are 0 today; future-proofs Tier-2)."""
    return {"rank": c.rank, "suit": c.suit,
            "enh": c.enhancement, "ed": c.edition, "seal": c.seal}


def _joker_d(j) -> dict:
    return {"type": int(j.type), "name": joker_name(j.type),
            "desc": joker_desc(j.type),            # human-readable effect for the viewer
            "counter": float(j.counter), "edition": int(j.edition),
            "sell": sell_value(j.type, j.sell_bonus)}


def _consum_d(con) -> dict:
    return {"kind": int(con.kind), "type_id": int(con.type_id),
            "name": _consum_name(con), "desc": consumable_desc(con.kind, con.type_id)}


def _consum_name(con) -> str:
    return consumable_name(con.kind, con.type_id)


def _boss_d(state) -> dict:
    """Active boss: name + effect text (empty when no boss on this blind)."""
    if not state.boss:
        return {}
    return {"id": int(state.boss), "name": boss_name(state.boss), "desc": boss_desc(state.boss)}


def _offer_name(o) -> str:
    """Readable name for a shop offer (joker / planet / tarot), via the canonical
    name helpers in engine.descriptions."""
    from ..engine.shop import SHOP_TO_CONSUMABLE_KIND, ShopKind
    if o.kind == ShopKind.JOKER:
        return joker_name(o.type_id)
    if o.kind in (ShopKind.PLANET, ShopKind.TAROT):
        return consumable_name(SHOP_TO_CONSUMABLE_KIND[o.kind], o.type_id)
    return f"{ShopKind(o.kind).name.title()} {o.type_id}"


def _offer_d(o) -> dict:
    return {"kind": int(o.kind), "type_id": int(o.type_id),
            "name": _offer_name(o), "cost": int(o.cost)}


def _pack_offer_d(p) -> dict:
    """A shop booster-pack offer (kind/size/cost + readable name & effect)."""
    from ..engine.packs import PackKind, PackSize
    name = (PackSize(p.size).name.title() + " " if int(p.size) > 1 else "") + \
        PackKind(p.kind).name.title() + " Pack"
    return {"kind": int(p.kind), "size": int(p.size), "cost": int(p.cost),
            "name": name, "desc": pack_desc(p.kind, p.size)}


def _pack_item_d(it) -> dict:
    """A revealed pack item during OPEN_PACK (a Joker or a consumable)."""
    from ..engine.packs import PackItemKind
    if it.kind == PackItemKind.JOKER:
        return {"kind": int(it.kind), "type_id": int(it.payload.type),
                "name": joker_name(it.payload.type), "desc": joker_desc(it.payload.type)}
    con = it.payload
    return {"kind": int(it.kind), "type_id": int(con.type_id),
            "name": _consum_name(con), "desc": consumable_desc(con.kind, con.type_id)}


def _voucher_name(vid) -> str:
    return voucher_name(int(vid))


def _voucher_d(vid) -> dict:
    return {"type_id": int(vid), "name": _voucher_name(vid), "desc": voucher_desc(vid)}


def _pending_d(state) -> dict:
    """The armed targeting Tarot awaiting target cards (empty when nothing is armed)."""
    if state.pending_consumable < 0:
        return {}
    con = state.consumables[state.pending_consumable]
    return {"slot": int(state.pending_consumable), "name": _consum_name(con),
            "desc": consumable_desc(con.kind, con.type_id)}


def action_label(action_id: int) -> str:
    verb, arg = decode(int(action_id))
    if verb in (Verb.PLAY, Verb.DISCARD):
        return f"{verb.name} cards {tuple(arg)}"
    if verb == Verb.BUY:
        return f"BUY offer {arg}"
    if verb == Verb.SELL:
        return f"SELL joker {arg}"
    if verb == Verb.REROLL:
        return "REROLL"
    if verb == Verb.REORDER:
        return f"REORDER {tuple(arg)}"
    if verb == Verb.LEAVE_SHOP:
        return "LEAVE SHOP"
    return verb.name


def render_board(state: GameState) -> str:
    jokers = " | ".join(joker_name(j.type) for j in state.jokers) or "—"
    hand = " ".join(card_str(c) for c in state.hand) or "—"
    lines = [
        f"Ante {state.ante}  blind {state.blind_index}  [{_PHASE.get(int(state.phase), state.phase)}]",
        f"score {state.round_score}/{state.required}   hands {state.hands_left}  "
        f"discards {state.discards_left}   ${state.money}",
        f"Jokers: {jokers}",
        f"Hand:   {hand}",
    ]
    if int(state.phase) == int(Phase.SHOP) and state.shop_offers:
        offers = "  ".join(f"[{_offer_name(o)} ${o.cost}]" for o in state.shop_offers)
        lines.append(f"Shop:   {offers}")
    return "\n".join(lines)


def replay_states(seed: int, actions: list[int]) -> list[GameState]:
    """States before each action plus the final state (engine is pure-deterministic)."""
    state = engine.reset(int(seed))
    states = [state]
    for a in actions:
        if state.done:
            break
        state, _ = engine.step(state, decode(int(a)))
        states.append(state)
    return states


def _b(obs: dict):
    return {k: jnp.asarray(v)[None] for k, v in obs.items()}


def record_agent_episode(net, params, seed: int, reward_name: str = "shaped",
                         topk: int = 6, greedy: bool = True,
                         enable_bosses: bool = False) -> list[dict]:
    apply = jax.jit(net.apply)
    env = BalatroEnv(reward_name, enable_bosses=enable_bosses)
    obs, mask = env.reset(int(seed))
    key = jax.random.PRNGKey(int(seed))
    steps: list[dict] = []
    done = False
    while not done and len(steps) < _MAX_STEPS:
        state = env.state
        logits, value_logits = apply(params, _b(obs), jnp.asarray(mask)[None])
        probs = np.asarray(jax.nn.softmax(logits[0]))
        value = float(np.asarray(value_decode(value_logits))[0])
        if greedy:
            a = int(np.argmax(np.asarray(logits[0])))
        else:
            key, sub = jax.random.split(key)
            from ..agent.ppo import sample_action
            a = int(np.asarray(sample_action(logits, sub))[0])
        legal = np.flatnonzero(np.asarray(mask))
        order = legal[np.argsort(probs[legal])[::-1][:topk]]
        top = [[action_label(int(i)), float(probs[i])] for i in order]
        verb, arg = decode(a)
        selected = list(arg) if verb in (Verb.PLAY, Verb.DISCARD) else []
        board = render_board(state)            # text blob kept for old-style fallback
        # Score breakdown for a PLAY: re-run scoring deterministically with a trace so the
        # viewer can show the exact base -> cards -> jokers -> mods math for this hand.
        score_trace = explain_play(state, tuple(arg))["trace"] if verb == Verb.PLAY else []
        obs, reward, done, info, mask = env.step(a)
        steps.append({
            "t": len(steps), "ante": int(state.ante), "blind": int(state.blind_index),
            "phase": _PHASE.get(int(state.phase)), "money": int(state.money),
            "board": board, "action_id": a, "action_label": action_label(a),
            "reward": float(reward), "value": value,
            "score": info.get("score"), "hand_type": info.get("hand_type"),
            "chips": info.get("chips"), "mult": info.get("mult"),
            "top_probs": top,
            # --- structured fields (schema v2) for the card-diff viewer ---
            "schema": 2,
            "verb": verb.name,
            "selected": selected,
            "hand": [_card_d(c) for c in state.hand],     # BEFORE-action hand snapshot
            "scoring_idx": list(info.get("scoring_idx", [])),
            "round_score": int(state.round_score),
            "required": int(state.required),
            "hands_left": int(state.hands_left),
            "discards_left": int(state.discards_left),
            "jokers": [_joker_d(j) for j in state.jokers],
            "shop_offers": ([_offer_d(o) for o in state.shop_offers]
                            if int(state.phase) == int(Phase.SHOP) else []),
            "hand_reset": bool(verb == Verb.LEAVE_SHOP),
            "earned": info.get("earned"),
            # --- boss / consumables / score-trace ---
            "boss": _boss_d(state),
            "consumables": [_consum_d(c) for c in state.consumables],
            "score_trace": score_trace,
            # --- E5 acquisition content (so future runs are recorded with ALL the detail) ---
            "pack_offers": ([_pack_offer_d(p) for p in state.pack_offers]
                            if int(state.phase) == int(Phase.SHOP) else []),
            "pack_open": ([_pack_item_d(it) for it in state.pack_open]
                          if int(state.phase) == int(Phase.OPEN_PACK) else []),
            "pack_picks": int(state.pack_picks),
            "voucher_offer": (_voucher_d(state.voucher_offer)
                              if (int(state.phase) == int(Phase.SHOP) and state.voucher_offer)
                              else None),
            "vouchers": [_voucher_d(v) for v in state.vouchers],
            "pending": _pending_d(state),
        })
    if env.state.done:        # explicit terminal frame so the viewer ENDS on the outcome
        st = env.state        # (otherwise the last frame is the pre-action state, e.g. "hands 1")
        steps.append({
            "t": len(steps), "ante": int(st.ante), "blind": int(st.blind_index),
            "phase": _PHASE.get(int(st.phase)), "money": int(st.money),
            "board": render_board(st), "action_id": None,
            "action_label": "WON" if st.won else "LOST", "reward": 0.0, "value": 0.0,
            "score": None, "hand_type": None, "chips": None, "mult": None, "top_probs": [],
            "schema": 2, "verb": "TERMINAL", "selected": [],
            "hand": [_card_d(c) for c in st.hand], "scoring_idx": [],
            "round_score": int(st.round_score), "required": int(st.required),
            "hands_left": int(st.hands_left), "discards_left": int(st.discards_left),
            "jokers": [_joker_d(j) for j in st.jokers], "shop_offers": [],
            "hand_reset": False, "earned": None,
            "boss": _boss_d(st), "consumables": [_consum_d(c) for c in st.consumables],
            "score_trace": [],
            "pack_offers": [], "pack_open": [], "pack_picks": 0,
            "voucher_offer": None, "vouchers": [_voucher_d(v) for v in st.vouchers],
            "pending": _pending_d(st),
        })
    return steps


def save_episode(steps: list[dict], path) -> None:
    with open(path, "w") as f:
        json.dump(steps, f)


def load_episode(path) -> list[dict]:
    with open(path) as f:
        return json.load(f)

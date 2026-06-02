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
from ..engine.cards import card_str
from ..engine.engine import Verb
from ..engine.jokers.base import JokerType
from ..engine.shop import joker_cost
from ..engine.state import GameState, Phase
from ..envs.actions import decode
from ..envs.balatro_env import BalatroEnv

_PHASE = {0: "PLAYING", 1: "WON", 2: "LOST", 3: "SHOP"}
_MAX_STEPS = 3000


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
    jokers = " | ".join(JokerType(j.type).name for j in state.jokers) or "—"
    hand = " ".join(card_str(c) for c in state.hand) or "—"
    lines = [
        f"Ante {state.ante}  blind {state.blind_index}  [{_PHASE.get(int(state.phase), state.phase)}]",
        f"score {state.round_score}/{state.required}   hands {state.hands_left}  "
        f"discards {state.discards_left}   ${state.money}",
        f"Jokers: {jokers}",
        f"Hand:   {hand}",
    ]
    if int(state.phase) == int(Phase.SHOP) and state.shop_offers:
        offers = "  ".join(f"[{JokerType(o.type).name} ${joker_cost(o.type)}]" for o in state.shop_offers)
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
                         topk: int = 6, greedy: bool = True) -> list[dict]:
    apply = jax.jit(net.apply)
    env = BalatroEnv(reward_name)
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
        board = render_board(state)
        obs, reward, done, info, mask = env.step(a)
        steps.append({
            "t": len(steps), "ante": int(state.ante), "blind": int(state.blind_index),
            "phase": _PHASE.get(int(state.phase)), "money": int(state.money),
            "board": board, "action_id": a, "action_label": action_label(a),
            "reward": float(reward), "value": value,
            "score": info.get("score"), "hand_type": info.get("hand_type"),
            "chips": info.get("chips"), "mult": info.get("mult"),
            "top_probs": top,
        })
    return steps


def save_episode(steps: list[dict], path) -> None:
    with open(path, "w") as f:
        json.dump(steps, f)


def load_episode(path) -> list[dict]:
    with open(path) as f:
        return json.load(f)

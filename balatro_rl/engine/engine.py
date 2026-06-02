"""The Tier-0 engine seam: reset / legal_actions / step.

step(state, action) -> (state', info) is a pure function (RNG rides inside the
state). Action = (Verb, tuple-of-hand-indices). The flat-id encoding + legal mask
used by the RL agent live in the env layer (Plan 3); here we use plain tuples.

Tier-0 has no shop: clearing a blind advances directly to the next blind with a
freshly shuffled deck and full hand. Shop/economy arrive in Plan 2.
"""
from __future__ import annotations

import dataclasses
import itertools
from enum import IntEnum

from .blinds import required_score
from .cards import standard_deck
from .economy import blind_reward, interest, MONEY_PER_UNUSED_HAND
from .hands import evaluate
from .jokers.base import REGISTRY, aggregate_rules
from .rng import RNG
from .scoring import score_play
from .shop import generate_offers, joker_cost, reroll_cost, sell_value, CARD_SLOTS
from .state import GameState, Phase

STARTING_MONEY = 4
HANDS_PER_BLIND = 4
DISCARDS_PER_BLIND = 3
HAND_SIZE = 8
MAX_SELECT = 5
JOKER_SLOTS = 5


class Verb(IntEnum):
    PLAY = 0
    DISCARD = 1
    BUY = 2
    SELL = 3
    REROLL = 4
    REORDER = 5
    LEAVE_SHOP = 6


def _draw(hand: list, deck: list, hand_size: int) -> tuple[list, list]:
    """Draw from the front of the (pre-shuffled) deck up to hand_size."""
    need = max(0, hand_size - len(hand))
    drawn = deck[:need]
    return hand + drawn, deck[need:]


def reset(seed: int) -> GameState:
    rng = RNG.from_seed(seed)
    deck, rng = rng.shuffle(standard_deck())
    hand, deck = _draw([], deck, HAND_SIZE)
    return GameState(
        deck=tuple(deck), hand=tuple(hand), ante=1, blind_index=0,
        round_score=0, required=required_score(1, 0),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND,
        hand_size=HAND_SIZE, levels=tuple([1] * 12), money=STARTING_MONEY,
        rng=rng, phase=Phase.PLAYING, done=False, won=False, jokers=(),
        shop_offers=(), rerolls_done=0,
    )


def legal_actions(state: GameState) -> list[tuple[Verb, tuple[int, ...]]]:
    if state.done:
        return []
    if state.phase == Phase.SHOP:
        actions = [(Verb.LEAVE_SHOP, 0)]
        for i, offer in enumerate(state.shop_offers):
            if state.money >= joker_cost(offer.type) and len(state.jokers) < JOKER_SLOTS:
                actions.append((Verb.BUY, i))
        for i in range(len(state.jokers)):
            actions.append((Verb.SELL, i))
        if state.money >= reroll_cost(state.rerolls_done):
            actions.append((Verb.REROLL, 0))
        n = len(state.jokers)
        for i in range(n):
            for j in range(n):
                if i != j:
                    actions.append((Verb.REORDER, (i, j)))
        return actions
    actions: list[tuple[Verb, tuple[int, ...]]] = []
    n = len(state.hand)
    for size in range(1, min(MAX_SELECT, n) + 1):
        for combo in itertools.combinations(range(n), size):
            if state.hands_left > 0:
                actions.append((Verb.PLAY, combo))
            if state.discards_left > 0:
                actions.append((Verb.DISCARD, combo))
    return actions


def _advance_blind(state: GameState):
    if state.blind_index < 2:
        new_ante, new_blind = state.ante, state.blind_index + 1
    else:
        new_ante, new_blind = state.ante + 1, 0
    deck, rng = state.rng.shuffle(standard_deck())
    hand, deck = _draw([], deck, state.hand_size)
    nxt = dataclasses.replace(
        state, ante=new_ante, blind_index=new_blind, deck=tuple(deck), hand=tuple(hand),
        round_score=0, required=required_score(new_ante, new_blind),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND, rng=rng,
        phase=Phase.PLAYING, shop_offers=(), rerolls_done=0)
    return nxt, {"verb": "leave_shop", "result": "next_blind",
                 "ante": new_ante, "blind": new_blind}


def _cash_out(state: GameState):
    """Apply blind reward + interest + leftover-hand money + joker on_round_end."""
    delta = (blind_reward(state.blind_index)
             + interest(state.money)
             + state.hands_left * MONEY_PER_UNUSED_HAND)
    money = state.money + delta
    rng = state.rng
    kept = []
    for js in state.jokers:
        js2, mdelta, destroy, rng = REGISTRY[js.type].on_round_end(state, js, rng)
        money += mdelta
        if not destroy:
            kept.append(js2)
    return money, tuple(kept), rng


def _enter_cashout_or_win(state: GameState, info: dict):
    # Win immediately if the Ante-8 Boss was just cleared (no shop).
    if state.ante >= 8 and state.blind_index == 2:
        won = dataclasses.replace(state, done=True, won=True, phase=Phase.WON)
        return won, {**info, "cleared": True, "result": "won"}
    money, jokers, rng = _cash_out(state)
    offers, rng = generate_offers(rng, CARD_SLOTS)
    shop = dataclasses.replace(state, money=money, jokers=jokers, rng=rng,
                               phase=Phase.SHOP, shop_offers=offers, rerolls_done=0)
    return shop, {**info, "cleared": True, "result": "shop", "earned": money - state.money}


def step(state: GameState, action: tuple[Verb, tuple[int, ...]]) -> tuple[GameState, dict]:
    assert not state.done, "step() called on a terminal state"
    if state.phase == Phase.SHOP:
        return _shop_step(state, action)
    verb, idx = action
    assert 1 <= len(idx) <= MAX_SELECT, "must select 1..5 cards"
    assert len(set(idx)) == len(idx), "duplicate card indices"
    assert all(0 <= i < len(state.hand) for i in idx), "index out of range"

    selected = [state.hand[i] for i in idx]
    chosen = set(idx)
    remaining = [c for i, c in enumerate(state.hand) if i not in chosen]

    if verb == Verb.DISCARD:
        assert state.discards_left > 0, "no discards left"
        hand, deck = _draw(remaining, list(state.deck), state.hand_size)
        nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                  discards_left=state.discards_left - 1)
        return nxt, {"verb": "discard", "discarded": len(idx)}

    # PLAY
    assert state.hands_left > 0, "no hands left"
    held = remaining  # cards still in hand (not played) score in the held phase
    res = score_play(selected, jokers=state.jokers, held=tuple(held))
    # Lifecycle: let scaling jokers (e.g. Ride the Bus) update from this hand.
    rules = aggregate_rules(state.jokers) if state.jokers else None
    if state.jokers:
        _, scoring_idx = evaluate(list(selected), rules)
        new_jokers = tuple(
            REGISTRY[js.type].on_play(state, list(selected), list(scoring_idx), rules, js)
            for js in state.jokers
        )
    else:
        new_jokers = state.jokers
    round_score = state.round_score + res.score
    hands_left = state.hands_left - 1
    info = {"verb": "play", "score": res.score, "hand_type": int(res.hand_type),
            "chips": res.chips, "mult": res.mult}

    if round_score >= state.required:
        # Blind cleared: cash out then enter the shop (or win at the Ante-8 boss);
        # _advance_blind on shop-leave reshuffles a fresh deck and redraws.
        carried = dataclasses.replace(state, jokers=new_jokers, round_score=round_score,
                                      hands_left=hands_left)  # decremented count feeds cash-out
        return _enter_cashout_or_win(carried, info)

    hand, deck = _draw(remaining, list(state.deck), state.hand_size)
    if hands_left <= 0:
        lost = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                   round_score=round_score, hands_left=0,
                                   done=True, won=False, phase=Phase.LOST,
                                   jokers=new_jokers)
        return lost, {**info, "result": "lost"}
    nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                              round_score=round_score, hands_left=hands_left,
                              jokers=new_jokers)
    return nxt, info


def _shop_step(state: GameState, action):
    verb = action[0]
    if verb == Verb.LEAVE_SHOP:
        return _advance_blind(state)
    if verb == Verb.BUY:
        i = action[1]
        offer = state.shop_offers[i]
        cost = joker_cost(offer.type)
        assert state.money >= cost, "cannot afford"
        assert len(state.jokers) < JOKER_SLOTS, "no joker slot"
        offers = tuple(o for k, o in enumerate(state.shop_offers) if k != i)
        nxt = dataclasses.replace(state, money=state.money - cost,
                                  jokers=state.jokers + (offer,), shop_offers=offers)
        return nxt, {"verb": "buy", "joker": int(offer.type), "cost": cost}
    if verb == Verb.SELL:
        i = action[1]
        js = state.jokers[i]
        value = sell_value(js.type, js.sell_bonus)
        jokers = tuple(j for k, j in enumerate(state.jokers) if k != i)
        nxt = dataclasses.replace(state, money=state.money + value, jokers=jokers)
        return nxt, {"verb": "sell", "value": value}
    if verb == Verb.REROLL:
        cost = reroll_cost(state.rerolls_done)
        assert state.money >= cost, "cannot afford reroll"
        offers, rng = generate_offers(state.rng, CARD_SLOTS)
        nxt = dataclasses.replace(state, money=state.money - cost, shop_offers=offers,
                                  rerolls_done=state.rerolls_done + 1, rng=rng)
        return nxt, {"verb": "reroll", "cost": cost}
    if verb == Verb.REORDER:
        i, j = action[1]
        jk = list(state.jokers)
        item = jk.pop(i)
        jk.insert(j, item)
        return dataclasses.replace(state, jokers=tuple(jk)), {"verb": "reorder"}
    raise ValueError(f"illegal shop action: {verb}")

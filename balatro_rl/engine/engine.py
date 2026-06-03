"""The Tier-0 engine seam: reset / legal_actions / step.

step(state, action) -> (state', info) is a pure function (RNG rides inside the
state). Action = (Verb, tuple-of-hand-indices). The flat-id encoding + legal mask
used by the RL agent live in the env layer (Plan 3); here we use plain tuples.

Clearing a blind routes: clear -> win-check (Ante-8 boss -> WON) -> cash-out
(blind reward + interest + leftover hands + joker on_round_end) -> SHOP phase
(buy/sell/reroll/reorder/leave) -> _advance_blind -> next blind.
"""
from __future__ import annotations

import dataclasses
import itertools
from enum import IntEnum

from .blinds import required_score
from .bosses import (
    BossEffect, boss_allows_play, boss_debuffed_idx, boss_discards_left,
    boss_draw_target, boss_filters_plays, boss_halves_base, boss_hand_size_delta,
    boss_hands_left, boss_hook_discard, boss_ox_zeroes_money, boss_tooth_cost,
    select_boss,
)
from .cards import Enhancement, standard_deck
from .consumables import apply_consumable
from .economy import blind_reward, interest, MONEY_PER_UNUSED_HAND
from .hands import evaluate
from .jokers.base import HandEvents, NO_RULES, REGISTRY, aggregate_rules
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
SHOP_ACTION_CAP = 12          # max actions per shop visit; then only LEAVE_SHOP is legal
GOLD_ENH_ROUND_END = 3        # $ per held Gold-enhancement card at round end (wiki: Gold Card)


class Verb(IntEnum):
    PLAY = 0
    DISCARD = 1
    BUY = 2
    SELL = 3
    REROLL = 4
    REORDER = 5
    LEAVE_SHOP = 6
    USE = 7        # use a consumable (payload = consumable index); a free action, any phase


def _draw(hand: list, deck: list, hand_size: int) -> tuple[list, list]:
    """Draw from the front of the (pre-shuffled) deck up to hand_size."""
    need = max(0, hand_size - len(hand))
    drawn = deck[:need]
    return hand + drawn, deck[need:]


def _start_round(state, jokers: tuple, rng):
    """Fold every joker's on_round_start at the start of a blind, threading rng.
    Jokers that pick a random per-round target (Ancient Joker, The Idol, Mail-In
    Rebate) stash it in js.counter here; the advanced rng is returned so the choice
    stays seed-deterministic and persists into the next state."""
    out = []
    for js in jokers:
        js, rng = REGISTRY[js.type].on_round_start(state, js, rng)
        out.append(js)
    return tuple(out), rng


def make_master_deck(card_mods=None) -> tuple:
    """Build a master deck from standard_deck(), optionally applying per-card mods.

    ACQUISITION STOPGAP (Phase B testing + future use): packs/consumables that would
    normally enhance cards are out of scope, so this is the deterministic way to put
    enhanced cards into play. `card_mods` maps a canonical deck index (0..51, the
    standard_deck() order: suit-major, rank-ascending) to a dict of mod fields, e.g.
    `{0: {"enhancement": Enhancement.GLASS, "seal": Seal.GOLD}}`. Default None ->
    the plain 52-card deck (byte-identical to the current game). NOT wired into the
    agent/obs; the agent stays blind to mods until the Phase D retrain.
    """
    deck = standard_deck()
    if card_mods:
        for idx, fields in card_mods.items():
            deck[idx] = dataclasses.replace(deck[idx], **fields)
    return tuple(deck)


def _deck_enh_histogram(master_deck) -> tuple:
    """Count cards by Enhancement across the full owned deck (indexed by Enhancement),
    for deck-reading jokers (Steel/Stone Joker). An all-NONE deck yields a tuple that is
    all-zeros except index 0 -- which those jokers ignore -- so it's a no-op on the
    unmodified game."""
    counts = [0] * len(Enhancement)
    for c in master_deck:
        counts[c.enhancement] += 1
    return tuple(counts)


def reset(seed: int, scale: float = 1.0, card_mods=None,
          enable_bosses: bool = False) -> GameState:
    rng = RNG.from_seed(seed)
    # The persistent master deck (cards + their mod fields) is the canonical
    # owned-card set. We shuffle the WORKING deck FROM a copy of it. With an
    # unmodified standard_deck() this is byte-identical to the old
    # `rng.shuffle(standard_deck())` (same order in, same rng, same result).
    # `card_mods` (default None) is an opt-in acquisition stopgap for enhanced
    # cards; see make_master_deck. The default path is unchanged / byte-identical.
    master_deck = make_master_deck(card_mods)
    deck, rng = rng.shuffle(list(master_deck))
    hand, deck = _draw([], deck, HAND_SIZE)
    # Start-of-blind fold (jokers is empty at reset, so this is a no-op now, but it
    # keeps reset and _advance_blind symmetric for any future starting jokers).
    jokers, rng = _start_round(None, (), rng)
    return GameState(
        deck=tuple(deck), hand=tuple(hand), ante=1, blind_index=0,
        round_score=0, required=required_score(1, 0, scale),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND,
        hand_size=HAND_SIZE, levels=tuple([1] * 12),
        hand_plays_run=tuple([0] * 12), hand_plays_round=tuple([0] * 12),
        money=STARTING_MONEY,
        rng=rng, phase=Phase.PLAYING, done=False, won=False, jokers=jokers,
        shop_offers=(), rerolls_done=0, req_scale=scale,
        master_deck=master_deck, boss=0, bosses_enabled=enable_bosses,
    )


def legal_actions(state: GameState) -> list[tuple[Verb, tuple[int, ...]]]:
    if state.done:
        return []
    if state.phase == Phase.SHOP:
        if state.shop_steps >= SHOP_ACTION_CAP:
            return [(Verb.LEAVE_SHOP, 0)]          # bound shop dithering -> force progress
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
    # Boss PLAY restrictions (Psychic exactly-5 / Eye no-repeat / Mouth single-type). Only
    # engaged on those boss blinds; off them `play_filter` is False and the loop is the
    # original (byte-identical action space for the default game).
    boss = BossEffect(state.boss)
    play_filter = boss_filters_plays(boss)
    rules = aggregate_rules(state.jokers) if boss in (BossEffect.THE_EYE, BossEffect.THE_MOUTH) \
        else NO_RULES
    for size in range(1, min(MAX_SELECT, n) + 1):
        for combo in itertools.combinations(range(n), size):
            if state.hands_left > 0 and (not play_filter or boss_allows_play(
                    boss, [state.hand[i] for i in combo], state.hand_plays_round, rules)):
                actions.append((Verb.PLAY, combo))
            if state.discards_left > 0:
                actions.append((Verb.DISCARD, combo))
    return actions


def _advance_blind(state: GameState):
    if state.blind_index < 2:
        new_ante, new_blind = state.ante, state.blind_index + 1
    else:
        new_ante, new_blind = state.ante + 1, 0
    rng = state.rng
    # Boss selection: only on the boss blind AND only when enabled. Disabled -> NONE and
    # ZERO rng drawn, so the deterministic stream (and every existing replay) is untouched.
    # Selected BEFORE the deal so The Manacle's -1 hand size applies to the draw.
    boss = BossEffect.NONE
    if new_blind == 2 and state.bosses_enabled:
        boss, rng = select_boss(rng, new_ante)
    # Boss blind-setup: hand size (Manacle), hands (Needle), discards (Water). Recomputed
    # fresh from the HAND_SIZE base each blind, so a boss's reduction never compounds and
    # resets on the next blind. All identity for boss == NONE -> byte-identical.
    hand_size = HAND_SIZE + boss_hand_size_delta(boss)
    # Reshuffle the working deck FROM the persistent master deck so any card mods
    # (enhancement/edition/seal) ride forward across the blind boundary. With an
    # unmodified deck this is byte-identical to the old shuffle(standard_deck()).
    deck, rng = rng.shuffle(list(state.master_deck))
    hand, deck = _draw([], deck, hand_size)
    # Start-of-blind fold: re-roll per-round joker targets (Ancient Joker, The Idol,
    # Mail-In Rebate), threading the rng so each round's pick is seed-deterministic.
    jokers, rng = _start_round(state, state.jokers, rng)
    nxt = dataclasses.replace(
        state, ante=new_ante, blind_index=new_blind, deck=tuple(deck), hand=tuple(hand),
        round_score=0, required=required_score(new_ante, new_blind, state.req_scale, boss),
        hand_size=hand_size,
        hands_left=boss_hands_left(boss, HANDS_PER_BLIND),
        discards_left=boss_discards_left(boss, DISCARDS_PER_BLIND), rng=rng,
        hand_plays_round=tuple([0] * 12),  # per-round counter resets each blind
        boss=int(boss),
        jokers=jokers, phase=Phase.PLAYING, shop_offers=(), rerolls_done=0, shop_steps=0)
    return nxt, {"verb": "leave_shop", "result": "next_blind",
                 "ante": new_ante, "blind": new_blind}


def _cash_out(state: GameState):
    """Apply blind reward + interest + leftover-hand money + joker on_round_end."""
    delta = (blind_reward(state.blind_index)
             + interest(state.money)
             + state.hands_left * MONEY_PER_UNUSED_HAND)
    # Gold ENHANCEMENT: +$3 for each Gold card still HELD in hand at round end
    # (wiki: Gold Card). Unmodified hands carry no Gold card -> +$0 (byte-identical).
    delta += GOLD_ENH_ROUND_END * sum(
        1 for c in state.hand if c.enhancement == Enhancement.GOLD)
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
                               phase=Phase.SHOP, shop_offers=offers, rerolls_done=0,
                               shop_steps=0)
    return shop, {**info, "cleared": True, "result": "shop", "earned": money - state.money}


def _use_consumable(state: GameState, ci: int) -> tuple[GameState, dict]:
    """Apply the consumable at index `ci` and remove it. A free action (doesn't end the
    turn or touch hands/discards) usable in any phase. Planets level a hand type; other
    kinds arrive in later sub-phases."""
    assert 0 <= ci < len(state.consumables), "no such consumable"
    con = state.consumables[ci]
    overrides = apply_consumable(state, con)
    remaining = state.consumables[:ci] + state.consumables[ci + 1:]
    nxt = dataclasses.replace(state, consumables=remaining, **overrides)
    return nxt, {"verb": "use", "kind": con.kind, "type_id": con.type_id}


def step(state: GameState, action: tuple[Verb, tuple[int, ...]]) -> tuple[GameState, dict]:
    assert not state.done, "step() called on a terminal state"
    if action[0] == Verb.USE:        # consumables are usable in any phase (free action)
        return _use_consumable(state, action[1])
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
        # The Serpent draws exactly 3 after a discard (else refill to hand size).
        target = boss_draw_target(BossEffect(state.boss), len(remaining), state.hand_size)
        hand, deck = _draw(remaining, list(state.deck), target)
        # Lifecycle: fold every joker's on_discard (mirrors _cash_out / on_round_end):
        # thread rng, accumulate money, persist scaling counters, drop self-consumers.
        money = state.money
        rng = state.rng
        kept = []
        for js in state.jokers:
            js2, mdelta, rng = REGISTRY[js.type].on_discard(state, selected, js, rng)
            money += mdelta
            if not REGISTRY[js.type].destroy_when(js2):
                kept.append(js2)
        nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                  discards_left=state.discards_left - 1,
                                  money=money, jokers=tuple(kept), rng=rng)
        return nxt, {"verb": "discard", "discarded": len(idx)}

    # PLAY
    assert state.hands_left > 0, "no hands left"
    held = remaining  # cards still in hand (not played) score in the held phase
    rules = aggregate_rules(state.jokers)   # empty jokers -> NO_RULES; reused by the on_play fold
    # Boss scoring effects (Phase C1): a suit/face boss debuffs the matching played cards
    # (score_play makes them fully inert), and The Flint halves the hand's base. Both are
    # no-ops off a boss blind (state.boss == 0) -> byte-identical to the pre-boss game.
    boss = BossEffect(state.boss)
    debuffed = boss_debuffed_idx(boss, selected, rules) if state.boss else ()
    res = score_play(selected, jokers=state.jokers, held=tuple(held),
                     joker_slots=JOKER_SLOTS, money=state.money,
                     hands_left=state.hands_left, discards_left=state.discards_left,
                     deck_count=len(state.deck),
                     hand_plays_run=state.hand_plays_run,
                     hand_plays_round=state.hand_plays_round,
                     deck_enh_counts=_deck_enh_histogram(state.master_deck),
                     debuffed_idx=debuffed, levels=state.levels,
                     flint=boss_halves_base(boss), rng=state.rng)
    # Probabilistic scoring jokers (Misprint, Bloodstone) consumed state.rng; the
    # advanced rng rides back on res.rng and MUST be written into every successor
    # state below so a fixed seed reproduces the same rolls deterministically.
    rng = res.rng
    # Apply scoring side effects threaded out on the ScoreResult (mirrors res.rng).
    # money_delta: Lucky/Gold-seal/Gold-enhancement money won during scoring.
    # destroyed_idx: PLAYED-card indices to destroy (Glass) -> drop the matching
    # objects from the persistent master_deck by identity (played cards ARE the
    # same Card objects as their master_deck entries; see reset/_advance_blind).
    # Both are always 0 / empty for the current game (no hook produces them), so
    # this block is a behavioral no-op until Phase B populates them.
    money = state.money + res.money_delta
    # Boss money effects (Phase C3): The Tooth charges $1 per card played (money may go
    # negative); The Ox zeroes money when you play your most-played run hand type. Both
    # no-ops off their blinds. Ox overrides (sets exactly $0) regardless of other deltas.
    money -= boss_tooth_cost(boss, len(selected))
    if boss_ox_zeroes_money(boss, int(res.hand_type), state.hand_plays_run):
        money = 0
    master_deck = state.master_deck
    if res.destroyed_idx:
        destroyed = {id(selected[i]) for i in res.destroyed_idx}
        master_deck = tuple(c for c in master_deck if id(c) not in destroyed)
    # Persistent enhancement overrides (Vampire strip -> NONE, Midas face -> GOLD), applied
    # to the surviving master_deck by identity (played cards ARE master_deck objects). Last
    # write wins if a card is recorded twice. Empty unless Vampire/Midas is owned -> no-op.
    if res.mutations:
        mut = {id(selected[i]): enh for i, enh in res.mutations}
        master_deck = tuple(dataclasses.replace(c, enhancement=int(mut[id(c)]))
                            if id(c) in mut else c for c in master_deck)
    # Lifecycle: let scaling jokers (e.g. Ride the Bus) update from this hand.
    if state.jokers:
        _, scoring_idx = evaluate(list(selected), rules)
        new_jokers = tuple(
            REGISTRY[js.type].on_play(state, list(selected), list(scoring_idx), rules, js)
            for js in state.jokers
        )
    else:
        new_jokers = state.jokers
    # Scaling lifecycle from this hand's enhancement EVENTS (Glass shattered / Lucky
    # triggered). Folded AFTER on_play, and only when an event actually fired, so an
    # unmodified hand never touches a joker counter (byte-identical to pre-Batch-8).
    if new_jokers and (res.destroyed_idx or res.lucky_triggered or res.vampire_consumed):
        events = HandEvents(glass_destroyed=len(res.destroyed_idx),
                            lucky_triggered=res.lucky_triggered,
                            vampire_consumed=res.vampire_consumed)
        new_jokers = tuple(REGISTRY[js.type].on_hand_events(js, events) for js in new_jokers)
    # Now (AFTER the joker on_play fold, which reads PRE-increment counts off `state`
    # for Obelisk) bump the per-HandType play counters for the hand just played.
    ht = int(res.hand_type)
    plays_run = tuple(c + (1 if i == ht else 0) for i, c in enumerate(state.hand_plays_run))
    plays_round = tuple(c + (1 if i == ht else 0) for i, c in enumerate(state.hand_plays_round))
    round_score = state.round_score + res.score
    hands_left = state.hands_left - 1
    info = {"verb": "play", "score": res.score, "hand_type": int(res.hand_type),
            "chips": res.chips, "mult": res.mult}

    if round_score >= state.required:
        # Blind cleared: cash out then enter the shop (or win at the Ante-8 boss);
        # _advance_blind on shop-leave reshuffles a fresh deck and redraws.
        carried = dataclasses.replace(state, jokers=new_jokers, round_score=round_score,
                                      hands_left=hands_left, rng=rng, money=money,
                                      master_deck=master_deck,
                                      hand_plays_run=plays_run, hand_plays_round=plays_round)
        return _enter_cashout_or_win(carried, info)

    # Boss draw effects (Phase C3): The Hook discards 2 random held cards before the redraw
    # (advancing rng -> threaded into the successor); The Serpent draws only 3. No-ops off
    # their blinds -> the default refill to hand_size is byte-identical.
    if boss == BossEffect.THE_HOOK:
        remaining, rng = boss_hook_discard(remaining, rng, 2)
    target = boss_draw_target(boss, len(remaining), state.hand_size)
    hand, deck = _draw(remaining, list(state.deck), target)
    if hands_left <= 0:
        lost = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                   round_score=round_score, hands_left=0,
                                   done=True, won=False, phase=Phase.LOST,
                                   jokers=new_jokers, rng=rng, money=money,
                                   master_deck=master_deck,
                                   hand_plays_run=plays_run, hand_plays_round=plays_round)
        return lost, {**info, "result": "lost"}
    nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                              round_score=round_score, hands_left=hands_left,
                              jokers=new_jokers, rng=rng, money=money,
                              master_deck=master_deck,
                              hand_plays_run=plays_run, hand_plays_round=plays_round)
    return nxt, info


def _shop_step(state: GameState, action):
    verb = action[0]
    if verb == Verb.LEAVE_SHOP:
        return _advance_blind(state)
    if verb == Verb.BUY:
        i = action[1]
        assert 0 <= i < len(state.shop_offers), "no such shop offer"
        offer = state.shop_offers[i]
        cost = joker_cost(offer.type)
        assert state.money >= cost, "cannot afford"
        assert len(state.jokers) < JOKER_SLOTS, "no joker slot"
        offers = tuple(o for k, o in enumerate(state.shop_offers) if k != i)
        nxt = dataclasses.replace(state, money=state.money - cost,
                                  jokers=state.jokers + (offer,), shop_offers=offers,
                                  shop_steps=state.shop_steps + 1)
        return nxt, {"verb": "buy", "joker": int(offer.type), "cost": cost}
    if verb == Verb.SELL:
        i = action[1]
        assert 0 <= i < len(state.jokers), "no such joker"
        js = state.jokers[i]
        value = sell_value(js.type, js.sell_bonus)
        jokers = tuple(j for k, j in enumerate(state.jokers) if k != i)
        nxt = dataclasses.replace(state, money=state.money + value, jokers=jokers,
                                  shop_steps=state.shop_steps + 1)
        return nxt, {"verb": "sell", "value": value}
    if verb == Verb.REROLL:
        cost = reroll_cost(state.rerolls_done)
        assert state.money >= cost, "cannot afford reroll"
        offers, rng = generate_offers(state.rng, CARD_SLOTS)
        nxt = dataclasses.replace(state, money=state.money - cost, shop_offers=offers,
                                  rerolls_done=state.rerolls_done + 1, rng=rng,
                                  shop_steps=state.shop_steps + 1)
        return nxt, {"verb": "reroll", "cost": cost}
    if verb == Verb.REORDER:
        i, j = action[1]
        assert 0 <= i < len(state.jokers) and 0 <= j < len(state.jokers), "reorder index out of range"
        jk = list(state.jokers)
        item = jk.pop(i)
        jk.insert(j, item)
        return dataclasses.replace(state, jokers=tuple(jk),
                                   shop_steps=state.shop_steps + 1), {"verb": "reorder"}
    raise ValueError(f"illegal shop action: {verb}")

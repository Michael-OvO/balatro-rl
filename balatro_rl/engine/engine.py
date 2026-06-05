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
    is_finisher, select_boss,
)
from .cards import Enhancement, standard_deck
from .consumables import (
    Consumable, apply_consumable, consumable_needs_target, max_targets, min_targets,
)
from .economy import blind_reward, interest, MONEY_PER_UNUSED_HAND
from .hands import evaluate
from .jokers.base import HandEvents, JokerState, NO_RULES, REGISTRY, aggregate_rules
from .rng import RNG
from .scoring import score_play
from .packs import PackItemKind, open_pack, roll_pack
from .shop import (
    CARD_SLOTS, SHOP_TO_CONSUMABLE_KIND, ShopKind, generate_offers,
    reroll_cost, sell_value,
)
from .state import GameState, Phase
from .vouchers import (
    VOUCHER_COST, VoucherType, extra_card_slots,
    extra_discards, extra_hand_size, extra_hands, extra_joker_slots, interest_cap,
    planet_weight_mult, prereq_met, reroll_discount, roll_voucher, tarot_weight_mult,
)

PACK_SLOTS = 2                # number of booster-pack offers generated per shop visit

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
    # E3 booster packs:
    OPEN = 8       # in SHOP: buy + open pack_offers[i] -> enter OPEN_PACK
    PICK = 9       # in OPEN_PACK: take pack_open[i] (consumable/joker slot) -> decrement picks
    SKIP_PACK = 10  # in OPEN_PACK: end picking -> back to SHOP
    # E4 vouchers:
    BUY_VOUCHER = 11  # in SHOP: buy the offered voucher -> apply a persistent run modifier
    # E5 targeting two-step: apply the armed (pending) targeting Tarot to the selected hand cards.
    USE_TARGET = 12   # payload = target hand indices (a subset); resolves state.pending_consumable


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
    the plain 52-card deck (byte-identical to the current game).
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
    # E5 targeting two-step: while a card-targeting Tarot is ARMED, the ONLY legal moves are the
    # target-subset selections (USE_TARGET). It is only ever armed in PLAYING with a non-empty
    # hand, so size-1 always exists -> the agent is never stuck. Bounded to the Tarot's reach.
    if state.pending_consumable >= 0:
        con = state.consumables[state.pending_consumable]
        n = len(state.hand)
        # Subsets sized [min_targets, max_targets]: Death needs EXACTLY 2 (a size-1 pick would
        # waste it), so it offers only size-2; every other targeting Tarot offers 1..max. Arming
        # only happens with n >= min_targets, so this range is always non-empty (never stuck).
        lo, hi = min_targets(con), min(max_targets(con), n)
        return [(Verb.USE_TARGET, combo)
                for size in range(lo, hi + 1)
                for combo in itertools.combinations(range(n), size)]
    # No-target consumables (Planets, no-target Tarots) USE in any phase — a free action.
    # Empty by default -> no USE actions (byte-identical until consumables are acquired).
    use = [(Verb.USE, i) for i, c in enumerate(state.consumables)
           if not consumable_needs_target(c)]
    if state.phase == Phase.OPEN_PACK:
        # E3/E5: pick K-of-M revealed items. Only items that CAN be added are pickable (a free
        # joker/consumable slot); SKIP_PACK always ends picking.
        actions = list(use)
        for i, item in enumerate(state.pack_open):
            if _can_take_pack_item(state, item):
                actions.append((Verb.PICK, i))
        actions.append((Verb.SKIP_PACK, 0))
        return actions
    if state.phase == Phase.SHOP:
        if state.shop_steps >= SHOP_ACTION_CAP:
            return [(Verb.LEAVE_SHOP, 0)]          # bound shop dithering -> force progress
        actions = use + [(Verb.LEAVE_SHOP, 0)]
        # E5: the agent now BUYS everything the shop offers. A JOKER offer needs a free joker
        # slot (voucher-raised by Antimatter); a consumable offer needs a free consumable slot.
        joker_cap = JOKER_SLOTS + extra_joker_slots(state.vouchers)
        for i, offer in enumerate(state.shop_offers):
            if state.money < offer.cost:
                continue
            if offer.kind == ShopKind.JOKER:
                if len(state.jokers) < joker_cap:
                    actions.append((Verb.BUY, i))
            elif len(state.consumables) < state.consumable_slots:
                actions.append((Verb.BUY, i))
        # E5: booster packs (buy + open into the OPEN_PACK sub-phase) and the single voucher slot.
        for i, pack in enumerate(state.pack_offers):
            if state.money >= pack.cost:
                actions.append((Verb.OPEN, i))
        if (state.voucher_offer and state.money >= VOUCHER_COST
                and prereq_met(state.vouchers, VoucherType(state.voucher_offer))):
            actions.append((Verb.BUY_VOUCHER, 0))
        for i in range(len(state.jokers)):
            actions.append((Verb.SELL, i))
        if state.money >= reroll_cost(state.rerolls_done,
                                      discount=reroll_discount(state.vouchers)):
            actions.append((Verb.REROLL, 0))
        n = len(state.jokers)
        for i in range(n):
            for j in range(n):
                if i != j:
                    actions.append((Verb.REORDER, (i, j)))
        return actions
    # PLAYING. Card-targeting Tarots are now ARM-able here via a bare (USE, ci); stepping it enters
    # the pending two-step above. Only offered when the hand holds enough cards to satisfy the
    # Tarot's min_targets (so Death — exactly 2 — never arms into a stuck pending state). No-target
    # consumables stay in `use`.
    actions: list[tuple[Verb, tuple[int, ...]]] = list(use)
    actions += [(Verb.USE, i) for i, c in enumerate(state.consumables)
                if consumable_needs_target(c) and len(state.hand) >= min_targets(c)]
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
    # resets on the next blind. All identity for boss == NONE -> byte-identical. Voucher
    # bonuses (Paint Brush/Palette hand size; Grabber/Nacho Tong hands; Wasteful/Recyclomancy
    # discards) are DERIVED from the owned vouchers and added on the base BEFORE the boss
    # override, so a boss clamp (Water's 0 discards) still wins. Empty vouchers -> +0.
    vs = state.vouchers
    hand_size = HAND_SIZE + extra_hand_size(vs) + boss_hand_size_delta(boss)
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
        hands_left=boss_hands_left(boss, HANDS_PER_BLIND + extra_hands(vs)),
        discards_left=boss_discards_left(boss, DISCARDS_PER_BLIND + extra_discards(vs)),
        rng=rng,
        hand_plays_round=tuple([0] * 12),  # per-round counter resets each blind
        boss=int(boss),
        jokers=jokers, phase=Phase.PLAYING, shop_offers=(), rerolls_done=0, shop_steps=0,
        pack_offers=(), pack_open=(), pack_picks=0)
    return nxt, {"verb": "leave_shop", "result": "next_blind",
                 "ante": new_ante, "blind": new_blind}


def _cash_out(state: GameState):
    """Apply blind reward + interest + leftover-hand money + joker on_round_end."""
    # Interest uses the voucher-derived cap (Seed Money $10 / Money Tree $20 / else $5).
    delta = (blind_reward(state.blind_index, is_finisher(BossEffect(state.boss)))
             + interest(state.money, cap=interest_cap(state.vouchers))
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


def _generate_pack_offers(rng, n: int = PACK_SLOTS):
    """Roll n booster-pack offers (E3), mirroring generate_offers for the card slots.
    Returns (tuple[Pack, ...], rng)."""
    packs = []
    for _ in range(n):
        pack, rng = roll_pack(rng)
        packs.append(pack)
    return tuple(packs), rng


def _enter_cashout_or_win(state: GameState, info: dict):
    # Win immediately if the Ante-8 Boss was just cleared (no shop).
    if state.ante >= 8 and state.blind_index == 2:
        won = dataclasses.replace(state, done=True, won=True, phase=Phase.WON)
        return won, {**info, "cleared": True, "result": "won"}
    money, jokers, rng = _cash_out(state)
    vs = state.vouchers
    # Card-slot count + Tarot/Planet shop weights are voucher-derived (Overstock(+Plus);
    # Tarot/Planet Merchant/Tycoon). Empty vouchers -> base counts/weights (byte-identical).
    offers, rng = generate_offers(rng, CARD_SLOTS + extra_card_slots(vs),
                                  tarot_mult=tarot_weight_mult(vs),
                                  planet_mult=planet_weight_mult(vs))
    pack_offers, rng = _generate_pack_offers(rng, PACK_SLOTS)
    # Roll the shop's single voucher slot: a uniform eligible voucher (None if none eligible).
    voucher, rng = roll_voucher(rng, vs)
    shop = dataclasses.replace(state, money=money, jokers=jokers, rng=rng,
                               phase=Phase.SHOP, shop_offers=offers, rerolls_done=0,
                               shop_steps=0, pack_offers=pack_offers,
                               pack_open=(), pack_picks=0,
                               voucher_offer=0 if voucher is None else int(voucher))
    return shop, {**info, "cleared": True, "result": "shop", "earned": money - state.money}


def _use_consumable(state: GameState, ci: int, targets=()) -> tuple[GameState, dict]:
    """Apply the consumable at index `ci` and remove it. A free action (doesn't end the
    turn or touch hands/discards) usable in any phase. Planets level a hand type; Tarots
    enhance/transform/destroy SELECTED hand cards (`targets` = hand indices), give money,
    or create more consumables/jokers.

    apply_consumable returns `(overrides, rng)`; we apply the overrides via
    dataclasses.replace, thread the (possibly advanced) rng back into the successor, and
    drop the used consumable. The overrides dict may itself set `consumables` (the create-*
    Tarots), so we splice the used card out FIRST and let the override win if present."""
    assert 0 <= ci < len(state.consumables), "no such consumable"
    con = state.consumables[ci]
    remaining = state.consumables[:ci] + state.consumables[ci + 1:]
    # Resolve against a view with the used card already removed, so create-* Tarots that
    # return a new `consumables` tuple build on the post-removal slots (cap respected).
    base = dataclasses.replace(state, consumables=remaining)
    overrides, rng = apply_consumable(base, con, targets=targets, rng=state.rng)
    nxt = dataclasses.replace(base, rng=rng, **overrides)
    return nxt, {"verb": "use", "kind": con.kind, "type_id": con.type_id}


def step(state: GameState, action: tuple[Verb, tuple[int, ...]]) -> tuple[GameState, dict]:
    assert not state.done, "step() called on a terminal state"
    # While a targeting Tarot is armed, legal_actions returns ONLY USE_TARGET, so any other verb
    # here means the caller ignored the mask. Reject it rather than silently leaking pending into
    # the successor (which would desync the stored consumable index).
    assert state.pending_consumable < 0 or action[0] == Verb.USE_TARGET, \
        "a consumable is armed; only USE_TARGET is legal until it resolves"
    if action[0] == Verb.USE_TARGET:   # E5: apply the ARMED targeting Tarot to the selected cards
        assert state.pending_consumable >= 0, "no consumable armed for targeting"
        ci, targets = state.pending_consumable, tuple(action[1])
        # Clear pending FIRST so the successor carries no stale arm, then apply.
        return _use_consumable(dataclasses.replace(state, pending_consumable=-1), ci, targets)
    if action[0] == Verb.USE:        # consumables are usable in any phase (free action)
        # USE encoding has three forms:
        #   (Verb.USE, (ci, *target_idx))  -> direct apply-with-targets (E2 scripted path / tests)
        #   (Verb.USE, ci)  on a no-target consumable  -> apply now (Planets, no-target Tarots)
        #   (Verb.USE, ci)  on a card-targeting Tarot   -> ARM the two-step (E5): set pending, then
        #                                                  the agent picks (USE_TARGET, subset)
        payload = action[1]
        if isinstance(payload, tuple):
            return _use_consumable(state, payload[0], payload[1:])
        ci = payload
        assert 0 <= ci < len(state.consumables), "no such consumable"
        con = state.consumables[ci]
        if consumable_needs_target(con):
            # Arm only when the hand can satisfy the Tarot's min_targets (so the pending state
            # below always has a legal USE_TARGET subset — never stuck; matters for Death's 2).
            assert len(state.hand) >= min_targets(con), "not enough hand cards to target"
            return dataclasses.replace(state, pending_consumable=ci), {"verb": "use_arm", "ci": ci}
        return _use_consumable(state, ci, ())
    if state.phase == Phase.OPEN_PACK:        # E3: pick K-of-M revealed pack items
        return _open_pack_step(state, action)
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
                     joker_slots=JOKER_SLOTS + extra_joker_slots(state.vouchers),
                     money=state.money,
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


def explain_play(state: GameState, idx) -> dict:
    """Re-run a PLAY's scoring WITH a breakdown trace, for the replay viewer. Uses the same
    inputs (and state.rng) as engine.step would, so the score is byte-identical to the real
    play and the trace faithfully explains it. Returns {trace, score, chips, mult, hand_type}.
    The trace is an ordered list of running-total events (base -> cards -> jokers -> mods)."""
    selected = [state.hand[i] for i in idx]
    held = tuple(c for i, c in enumerate(state.hand) if i not in set(idx))
    rules = aggregate_rules(state.jokers)
    boss = BossEffect(state.boss)
    debuffed = boss_debuffed_idx(boss, selected, rules) if state.boss else ()
    trace: list = []
    res = score_play(selected, jokers=state.jokers, held=held,
                     joker_slots=JOKER_SLOTS + extra_joker_slots(state.vouchers),
                     money=state.money,
                     hands_left=state.hands_left, discards_left=state.discards_left,
                     deck_count=len(state.deck),
                     hand_plays_run=state.hand_plays_run, hand_plays_round=state.hand_plays_round,
                     deck_enh_counts=_deck_enh_histogram(state.master_deck),
                     debuffed_idx=debuffed, levels=state.levels,
                     flint=boss_halves_base(boss), trace=trace, rng=state.rng)
    return {"trace": trace, "score": res.score, "chips": res.chips,
            "mult": float(res.mult), "hand_type": int(res.hand_type)}


def _can_take_pack_item(state: GameState, item) -> bool:
    """Whether a revealed pack item can be added (a free slot of the right kind). A JOKER
    needs a free joker slot (voucher-raised by Antimatter, matching the shop BUY check); a
    CONSUMABLE needs a free consumable slot."""
    if item.kind == PackItemKind.JOKER:
        return len(state.jokers) < JOKER_SLOTS + extra_joker_slots(state.vouchers)
    return len(state.consumables) < state.consumable_slots


def _open_pack_step(state: GameState, action):
    """E3 OPEN_PACK sub-phase: PICK item i (add to the run, decrement picks) or SKIP_PACK
    (end picking). Returns to SHOP once picks are exhausted or on SKIP."""
    verb = action[0]
    if verb == Verb.SKIP_PACK:
        nxt = dataclasses.replace(state, phase=Phase.SHOP, pack_open=(), pack_picks=0)
        return nxt, {"verb": "skip_pack"}
    if verb == Verb.PICK:
        i = action[1]
        assert 0 <= i < len(state.pack_open), "no such pack item"
        item = state.pack_open[i]
        assert _can_take_pack_item(state, item), "no slot for this pack item"
        common = dict(pack_open=tuple(o for k, o in enumerate(state.pack_open) if k != i))
        if item.kind == PackItemKind.JOKER:
            taken = dict(jokers=state.jokers + (item.payload,))
            info = {"verb": "pick", "kind": int(PackItemKind.JOKER),
                    "type_id": int(item.payload.type)}
        else:
            taken = dict(consumables=state.consumables + (item.payload,))
            info = {"verb": "pick", "kind": int(PackItemKind.CONSUMABLE),
                    "type_id": int(item.payload.type_id)}
        picks = state.pack_picks - 1
        if picks <= 0:        # picks exhausted -> resume the shop
            nxt = dataclasses.replace(state, phase=Phase.SHOP, pack_open=(),
                                      pack_picks=0, **taken)
        else:
            nxt = dataclasses.replace(state, pack_picks=picks, **common, **taken)
        return nxt, info
    raise ValueError(f"illegal open-pack action: {verb}")


def _shop_step(state: GameState, action):
    verb = action[0]
    if verb == Verb.LEAVE_SHOP:
        return _advance_blind(state)
    if verb == Verb.OPEN:        # E3: buy + open a booster pack -> enter OPEN_PACK
        i = action[1]
        assert 0 <= i < len(state.pack_offers), "no such pack offer"
        pack = state.pack_offers[i]
        assert state.money >= pack.cost, "cannot afford pack"
        items, picks, rng = open_pack(pack, state.rng)
        offers = tuple(p for k, p in enumerate(state.pack_offers) if k != i)
        nxt = dataclasses.replace(state, money=state.money - pack.cost,
                                  pack_offers=offers, phase=Phase.OPEN_PACK,
                                  pack_open=items, pack_picks=picks, rng=rng,
                                  shop_steps=state.shop_steps + 1)
        return nxt, {"verb": "open", "kind": int(pack.kind), "size": int(pack.size),
                     "cost": pack.cost, "shown": len(items), "picks": picks}
    if verb == Verb.BUY_VOUCHER:        # E4: buy the offered voucher -> persistent run modifier
        assert state.voucher_offer != 0, "no voucher offered"
        voucher = VoucherType(state.voucher_offer)
        assert state.money >= VOUCHER_COST, "cannot afford voucher"
        assert prereq_met(state.vouchers, voucher), "voucher prerequisite not met"
        vouchers = state.vouchers + (int(voucher),)
        over = dict(money=state.money - VOUCHER_COST, vouchers=vouchers,
                    voucher_offer=0, shop_steps=state.shop_steps + 1)
        # Most voucher effects are DERIVED at use-site from `vouchers` (per-blind/shop/cap),
        # so nothing else changes now. Crystal Ball's +1 consumable slot is a stored FIELD,
        # so apply it immediately (other vouchers move no immediate field).
        if voucher == VoucherType.CRYSTAL_BALL:
            over["consumable_slots"] = state.consumable_slots + 1
        nxt = dataclasses.replace(state, **over)
        return nxt, {"verb": "buy_voucher", "type_id": int(voucher), "cost": VOUCHER_COST}
    if verb == Verb.BUY:
        i = action[1]
        assert 0 <= i < len(state.shop_offers), "no such shop offer"
        offer = state.shop_offers[i]
        cost = offer.cost
        assert state.money >= cost, "cannot afford"
        offers = tuple(o for k, o in enumerate(state.shop_offers) if k != i)
        common = dict(money=state.money - cost, shop_offers=offers,
                      shop_steps=state.shop_steps + 1)
        # Kind-aware acquisition: a JOKER fills a joker slot; any consumable kind (PLANET
        # here, more in later phases) goes to a consumable slot (respecting the cap).
        if offer.kind == ShopKind.JOKER:
            assert len(state.jokers) < JOKER_SLOTS + extra_joker_slots(state.vouchers), \
                "no joker slot"
            nxt = dataclasses.replace(state, jokers=state.jokers + (
                JokerState(type=offer.type_id),), **common)
        else:
            assert len(state.consumables) < state.consumable_slots, "no consumable slot"
            # Store the consumable under its ConsumableKind (so USE/obs/replay read it
            # correctly), not the raw ShopKind — the two enums number members differently.
            con_kind = SHOP_TO_CONSUMABLE_KIND[offer.kind]
            nxt = dataclasses.replace(state, consumables=state.consumables + (
                Consumable(kind=con_kind, type_id=offer.type_id),), **common)
        return nxt, {"verb": "buy", "kind": int(offer.kind),
                     "type_id": int(offer.type_id), "cost": cost}
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
        vs = state.vouchers
        cost = reroll_cost(state.rerolls_done, discount=reroll_discount(vs))
        assert state.money >= cost, "cannot afford reroll"
        # Re-rolled offers honor the same voucher-derived card-slot count + shop weights.
        offers, rng = generate_offers(state.rng, CARD_SLOTS + extra_card_slots(vs),
                                      tarot_mult=tarot_weight_mult(vs),
                                      planet_mult=planet_weight_mult(vs))
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

"""The proof-set joker implementations. Each value is wiki-cited; verify against
balatrowiki.org before changing. This module registers jokers on import.
"""
from __future__ import annotations

import dataclasses

from ..hands import HandType, contains, is_face
from .base import (
    Effect, JokerEffect, JokerState, JokerType, Rarity, RuleFlags,
    NO_RULES, aggregate_rules, register,
)


@register(JokerType.JOKER)
class _Joker(JokerEffect):  # wiki: /w/Joker  — +4 Mult
    rarity = Rarity.COMMON
    cost = 2
    def independent(self, ctx, js):
        return Effect(mult=4)


@register(JokerType.CAVENDISH)
class _Cavendish(JokerEffect):  # wiki: /w/Cavendish  — X3 Mult; 1 in 1000 self-destroy at end of round
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(xmult=3.0)
    def on_round_end(self, state, js, rng):
        roll, rng = rng.random()
        return js, 0, roll < 0.001, rng


@register(JokerType.GREEDY)
class _Greedy(JokerEffect):  # wiki: /w/Greedy_Joker  — +3 Mult per scored Diamond
    rarity = Rarity.COMMON
    cost = 5
    def on_score(self, ctx, card, index, js):
        return Effect(mult=3) if card.suit == 3 else Effect()


@register(JokerType.SCARY_FACE)
class _ScaryFace(JokerEffect):  # wiki: /w/Scary_Face  — +30 Chips per scored face card
    rarity = Rarity.COMMON
    cost = 4
    def on_score(self, ctx, card, index, js):
        return Effect(chips=30) if is_face(card, ctx.rules) else Effect()


@register(JokerType.PHOTOGRAPH)
class _Photograph(JokerEffect):  # wiki: /w/Photograph  — X2 on first scoring face card (re-applies on retrigger)
    rarity = Rarity.COMMON
    cost = 5
    def on_score(self, ctx, card, index, js):
        return Effect(xmult=2.0) if index == ctx.first_face_idx else Effect()


@register(JokerType.BARON)
class _Baron(JokerEffect):  # wiki: /w/Baron  — each King held gives X1.5 Mult
    rarity = Rarity.RARE
    cost = 8
    def on_held(self, ctx, card, js):
        return Effect(xmult=1.5) if card.rank == 13 else Effect()


@register(JokerType.HACK)
class _Hack(JokerEffect):  # wiki: /w/Hack  — retrigger each played 2,3,4,5
    rarity = Rarity.UNCOMMON
    cost = 6
    def retrigger(self, ctx, card, js):
        return 1 if card.rank in (2, 3, 4, 5) else 0


@register(JokerType.SPLASH)
class _Splash(JokerEffect):  # wiki: /w/Splash  — every played card scores
    copyable = False
    rarity = Rarity.COMMON
    cost = 3
    def rules(self):
        return RuleFlags(splash=True)


@register(JokerType.PAREIDOLIA)
class _Pareidolia(JokerEffect):  # wiki: /w/Pareidolia  — all cards are face cards
    copyable = False
    rarity = Rarity.UNCOMMON
    cost = 5
    def rules(self):
        return RuleFlags(all_face=True)


@register(JokerType.RIDE_THE_BUS)
class _RideTheBus(JokerEffect):  # wiki: /w/Ride_the_Bus
    rarity = Rarity.COMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(mult=js.counter)

    def on_play(self, state, played, scoring_idx, rules, js):
        scored_face = any(is_face(played[i], rules) for i in scoring_idx)
        new_counter = 0.0 if scored_face else js.counter + 1.0
        return dataclasses.replace(js, counter=new_counter)


@register(JokerType.BLUEPRINT)
class _Blueprint(JokerEffect):  # wiki: /w/Blueprint  — copy resolution handled in base.resolve_providers
    rarity = Rarity.RARE
    cost = 10


@register(JokerType.GOLDEN_JOKER)
class _GoldenJoker(JokerEffect):  # wiki: /w/Golden_Joker  — +$4 at end of round
    rarity = Rarity.COMMON
    cost = 6
    def on_round_end(self, state, js, rng):
        return js, 4, False, rng


@register(JokerType.EGG)
class _Egg(JokerEffect):  # wiki: /w/Egg  — gains +$3 sell value at end of round
    rarity = Rarity.COMMON
    cost = 4
    def on_round_end(self, state, js, rng):
        return dataclasses.replace(js, sell_bonus=js.sell_bonus + 3), 0, False, rng


# --- Batch 1: suit on-scored (+3 Mult per scored card of a suit) ---

@register(JokerType.LUSTY)
class _Lusty(JokerEffect):  # wiki: /w/Lusty_Joker  — +3 Mult per scored Heart
    rarity = Rarity.COMMON
    cost = 5
    def on_score(self, ctx, card, index, js):
        return Effect(mult=3) if card.suit == 1 else Effect()


@register(JokerType.WRATHFUL)
class _Wrathful(JokerEffect):  # wiki: /w/Wrathful_Joker  — +3 Mult per scored Spade
    rarity = Rarity.COMMON
    cost = 5
    def on_score(self, ctx, card, index, js):
        return Effect(mult=3) if card.suit == 0 else Effect()


@register(JokerType.GLUTTONOUS)
class _Gluttonous(JokerEffect):  # wiki: /w/Gluttonous_Joker  — +3 Mult per scored Club
    rarity = Rarity.COMMON
    cost = 5
    def on_score(self, ctx, card, index, js):
        return Effect(mult=3) if card.suit == 2 else Effect()


# --- Batch 1: hand-type +Mult (independent) ---

@register(JokerType.JOLLY)
class _Jolly(JokerEffect):  # wiki: /w/Jolly_Joker  — +8 Mult if hand contains a Pair
    rarity = Rarity.COMMON
    cost = 3
    def independent(self, ctx, js):
        return Effect(mult=8) if HandType.PAIR in ctx.contains else Effect()


@register(JokerType.ZANY)
class _Zany(JokerEffect):  # wiki: /w/Zany_Joker  — +12 Mult if contains Three of a Kind
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(mult=12) if HandType.THREE_OF_A_KIND in ctx.contains else Effect()


@register(JokerType.MAD)
class _Mad(JokerEffect):  # wiki: /w/Mad_Joker  — +10 Mult if contains Two Pair
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(mult=10) if HandType.TWO_PAIR in ctx.contains else Effect()


@register(JokerType.CRAZY)
class _Crazy(JokerEffect):  # wiki: /w/Crazy_Joker  — +12 Mult if contains a Straight
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(mult=12) if HandType.STRAIGHT in ctx.contains else Effect()


@register(JokerType.DROLL)
class _Droll(JokerEffect):  # wiki: /w/Droll_Joker  — +10 Mult if contains a Flush
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(mult=10) if HandType.FLUSH in ctx.contains else Effect()


# --- Batch 1: hand-type +Chips (independent) ---

@register(JokerType.SLY)
class _Sly(JokerEffect):  # wiki: /w/Sly_Joker  — +50 Chips if contains a Pair
    rarity = Rarity.COMMON
    cost = 3
    def independent(self, ctx, js):
        return Effect(chips=50) if HandType.PAIR in ctx.contains else Effect()


@register(JokerType.WILY)
class _Wily(JokerEffect):  # wiki: /w/Wily_Joker  — +100 Chips if contains Three of a Kind
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(chips=100) if HandType.THREE_OF_A_KIND in ctx.contains else Effect()


@register(JokerType.CLEVER)
class _Clever(JokerEffect):  # wiki: /w/Clever_Joker  — +80 Chips if contains Two Pair
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(chips=80) if HandType.TWO_PAIR in ctx.contains else Effect()


@register(JokerType.DEVIOUS)
class _Devious(JokerEffect):  # wiki: /w/Devious_Joker  — +100 Chips if contains a Straight
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(chips=100) if HandType.STRAIGHT in ctx.contains else Effect()


@register(JokerType.CRAFTY)
class _Crafty(JokerEffect):  # wiki: /w/Crafty_Joker  — +80 Chips if contains a Flush
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(chips=80) if HandType.FLUSH in ctx.contains else Effect()


@register(JokerType.HALF)
class _Half(JokerEffect):  # wiki: /w/Half_Joker  — +20 Mult if played hand has <=3 cards
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(mult=20) if len(ctx.played) <= 3 else Effect()


# --- Batch 1: on-scored per-card ---

@register(JokerType.FIBONACCI)
class _Fibonacci(JokerEffect):  # wiki: /w/Fibonacci  — each played A,2,3,5,8 gives +8 Mult
    rarity = Rarity.UNCOMMON
    cost = 8
    def on_score(self, ctx, card, index, js):
        return Effect(mult=8) if card.rank in (14, 2, 3, 5, 8) else Effect()


@register(JokerType.EVEN_STEVEN)
class _EvenSteven(JokerEffect):  # wiki: /w/Even_Steven  — even-rank scored cards give +4 Mult
    rarity = Rarity.COMMON
    cost = 4
    def on_score(self, ctx, card, index, js):
        return Effect(mult=4) if card.rank in (2, 4, 6, 8, 10) else Effect()


@register(JokerType.ODD_TODD)
class _OddTodd(JokerEffect):  # wiki: /w/Odd_Todd  — odd-rank scored cards give +31 Chips (Ace counts odd)
    rarity = Rarity.COMMON
    cost = 4
    def on_score(self, ctx, card, index, js):
        return Effect(chips=31) if card.rank in (3, 5, 7, 9, 14) else Effect()


@register(JokerType.SCHOLAR)
class _Scholar(JokerEffect):  # wiki: /w/Scholar  — scored Aces give +20 Chips and +4 Mult
    rarity = Rarity.COMMON
    cost = 4
    def on_score(self, ctx, card, index, js):
        return Effect(chips=20, mult=4) if card.rank == 14 else Effect()


@register(JokerType.WALKIE_TALKIE)
class _WalkieTalkie(JokerEffect):  # wiki: /w/Walkie_Talkie  — each scored 10 or 4 gives +10 Chips +4 Mult
    rarity = Rarity.COMMON
    cost = 4
    def on_score(self, ctx, card, index, js):
        return Effect(chips=10, mult=4) if card.rank in (10, 4) else Effect()


@register(JokerType.SMILEY_FACE)
class _SmileyFace(JokerEffect):  # wiki: /w/Smiley_Face  — scored face cards give +5 Mult
    rarity = Rarity.COMMON
    cost = 4
    def on_score(self, ctx, card, index, js):
        return Effect(mult=5) if is_face(card, ctx.rules) else Effect()


# --- Batch 1: retrigger ---

@register(JokerType.SOCK_AND_BUSKIN)
class _SockAndBuskin(JokerEffect):  # wiki: /w/Sock_and_Buskin  — retrigger all played face cards
    rarity = Rarity.UNCOMMON
    cost = 6
    def retrigger(self, ctx, card, js):
        return 1 if is_face(card, ctx.rules) else 0


# --- Batch 1: independent + economy ---

@register(JokerType.GROS_MICHEL)
class _GrosMichel(JokerEffect):  # wiki: /w/Gros_Michel  — +15 Mult; 1 in 6 self-destroy at end of round
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(mult=15)
    def on_round_end(self, state, js, rng):
        roll, rng = rng.random()
        return js, 0, roll < 1 / 6, rng


# --- Batch 1: scaling ---

@register(JokerType.RUNNER)
class _Runner(JokerEffect):  # wiki: /w/Runner  — +15 Chips per played hand that contains a Straight (start +0)
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(chips=int(js.counter))
    def on_play(self, state, played, scoring_idx, rules, js):
        bump = 15.0 if HandType.STRAIGHT in contains(list(played)) else 0.0
        return dataclasses.replace(js, counter=js.counter + bump)


@register(JokerType.ICE_CREAM)
class _IceCream(JokerEffect):  # wiki: /w/Ice_Cream  — +100 Chips, -5 per hand played (counter starts 0)
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(chips=max(0, 100 - 5 * int(js.counter)))
    def on_play(self, state, played, scoring_idx, rules, js):
        return dataclasses.replace(js, counter=js.counter + 1.0)


# --- Batch 2: state-reading jokers (read ScoreContext game-state fields) ---

@register(JokerType.ABSTRACT_JOKER)
class _AbstractJoker(JokerEffect):  # wiki: /w/Abstract_Joker  — +3 Mult per owned joker (incl. itself)
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(mult=3 * ctx.n_jokers)


@register(JokerType.JOKER_STENCIL)
class _JokerStencil(JokerEffect):  # wiki: /w/Joker_Stencil  — X1 Mult per empty slot, own slot counts as empty
    rarity = Rarity.UNCOMMON
    cost = 8
    def independent(self, ctx, js):
        return Effect(xmult=float(ctx.empty_joker_slots + 1))


@register(JokerType.BULL)
class _Bull(JokerEffect):  # wiki: /w/Bull  — +2 Chips per $1 held (no bonus at $0 or less)
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(chips=2 * max(0, ctx.money))


@register(JokerType.BANNER)
class _Banner(JokerEffect):  # wiki: /w/Banner  — +30 Chips per remaining discard
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(chips=30 * ctx.discards_left)


@register(JokerType.MYSTIC_SUMMIT)
class _MysticSummit(JokerEffect):  # wiki: /w/Mystic_Summit  — +15 Mult when 0 discards remaining
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(mult=15) if ctx.discards_left == 0 else Effect()


@register(JokerType.BLUE_JOKER)
class _BlueJoker(JokerEffect):  # wiki: /w/Blue_Joker  — +2 Chips per remaining card in deck
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(chips=2 * ctx.deck_count)


# --- Batch 2: scaling state-reading jokers ---

@register(JokerType.SQUARE_JOKER)
class _SquareJoker(JokerEffect):  # wiki: /w/Square_Joker  — +4 Chips if played hand has exactly 4 cards (start 0)
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(chips=int(js.counter))
    def on_play(self, state, played, scoring_idx, rules, js):
        bump = 4.0 if len(played) == 4 else 0.0
        return dataclasses.replace(js, counter=js.counter + bump)


@register(JokerType.SPARE_TROUSERS)
class _SpareTrousers(JokerEffect):  # wiki: /w/Spare_Trousers  — +2 Mult if played hand contains Two Pair (start 0)
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(mult=js.counter)
    def on_play(self, state, played, scoring_idx, rules, js):
        bump = 2.0 if HandType.TWO_PAIR in contains(list(played)) else 0.0
        return dataclasses.replace(js, counter=js.counter + bump)


@register(JokerType.WEE_JOKER)
class _WeeJoker(JokerEffect):  # wiki: /w/Wee_Joker  — +8 Chips per scored 2 (start 0)
    rarity = Rarity.RARE
    cost = 8
    def independent(self, ctx, js):
        return Effect(chips=int(js.counter))
    def on_play(self, state, played, scoring_idx, rules, js):
        twos = sum(1 for i in scoring_idx if played[i].rank == 2)
        return dataclasses.replace(js, counter=js.counter + 8.0 * twos)


@register(JokerType.POPCORN)
class _Popcorn(JokerEffect):  # wiki: /w/Popcorn  — +20 Mult, -4 Mult per round played (counter = rounds passed)
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(mult=max(0.0, 20.0 - 4.0 * js.counter))
    def on_round_end(self, state, js, rng):
        return dataclasses.replace(js, counter=js.counter + 1.0), 0, False, rng


# --- Batch 3: hand-contains xMult (independent) ---

@register(JokerType.THE_DUO)
class _TheDuo(JokerEffect):  # wiki: /w/The_Duo  — X2 Mult if hand contains a Pair
    rarity = Rarity.RARE
    cost = 8
    def independent(self, ctx, js):
        return Effect(xmult=2.0) if HandType.PAIR in ctx.contains else Effect()


@register(JokerType.THE_TRIO)
class _TheTrio(JokerEffect):  # wiki: /w/The_Trio  — X3 Mult if contains Three of a Kind
    rarity = Rarity.RARE
    cost = 8
    def independent(self, ctx, js):
        return Effect(xmult=3.0) if HandType.THREE_OF_A_KIND in ctx.contains else Effect()


@register(JokerType.THE_FAMILY)
class _TheFamily(JokerEffect):  # wiki: /w/The_Family  — X4 Mult if contains Four of a Kind
    rarity = Rarity.RARE
    cost = 8
    def independent(self, ctx, js):
        return Effect(xmult=4.0) if HandType.FOUR_OF_A_KIND in ctx.contains else Effect()


@register(JokerType.THE_ORDER)
class _TheOrder(JokerEffect):  # wiki: /w/The_Order  — X3 Mult if contains a Straight
    rarity = Rarity.RARE
    cost = 8
    def independent(self, ctx, js):
        return Effect(xmult=3.0) if HandType.STRAIGHT in ctx.contains else Effect()


@register(JokerType.THE_TRIBE)
class _TheTribe(JokerEffect):  # wiki: /w/The_Tribe  — X2 Mult if contains a Flush
    rarity = Rarity.RARE
    cost = 8
    def independent(self, ctx, js):
        return Effect(xmult=2.0) if HandType.FLUSH in ctx.contains else Effect()


# --- Batch 3: suit on-scored (per-card on_score) ---

@register(JokerType.ONYX_AGATE)
class _OnyxAgate(JokerEffect):  # wiki: /w/Onyx_Agate  — +7 Mult per scored Club
    rarity = Rarity.UNCOMMON
    cost = 7
    def on_score(self, ctx, card, index, js):
        return Effect(mult=7) if card.suit == 2 else Effect()


@register(JokerType.ARROWHEAD)
class _Arrowhead(JokerEffect):  # wiki: /w/Arrowhead  — +50 Chips per scored Spade
    rarity = Rarity.UNCOMMON
    cost = 7
    def on_score(self, ctx, card, index, js):
        return Effect(chips=50) if card.suit == 0 else Effect()


# --- Batch 3: independent jokers reading the scoring / held cards ---

@register(JokerType.SEEING_DOUBLE)
class _SeeingDouble(JokerEffect):  # wiki: /w/Seeing_Double  — X2 Mult if scoring cards include a Club and a card of any other suit
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        suits = {ctx.played[i].suit for i in ctx.scoring_idx}
        return Effect(xmult=2.0) if (2 in suits and suits - {2}) else Effect()


@register(JokerType.FLOWER_POT)
class _FlowerPot(JokerEffect):  # wiki: /w/Flower_Pot  — X3 Mult if scoring cards include all four suits
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        suits = {ctx.played[i].suit for i in ctx.scoring_idx}
        return Effect(xmult=3.0) if {0, 1, 2, 3} <= suits else Effect()


@register(JokerType.BLACKBOARD)
class _Blackboard(JokerEffect):  # wiki: /w/Blackboard  — X3 Mult if every held card is a Spade or Club (vacuously true if none held)
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        all_dark = all(card.suit in (0, 2) for card in ctx.held)
        return Effect(xmult=3.0) if all_dark else Effect()


# --- Batch 3: economy (on_round_end) ---

@register(JokerType.TO_THE_MOON)
class _ToTheMoon(JokerEffect):  # wiki: /w/To_the_Moon  — extra $1 interest per $5 held at end of round (capped like normal interest)
    rarity = Rarity.UNCOMMON
    cost = 5
    def on_round_end(self, state, js, rng):
        from ..economy import interest
        return js, interest(state.money), False, rng


@register(JokerType.DELAYED_GRATIFICATION)
class _DelayedGratification(JokerEffect):  # wiki: /w/Delayed_Gratification  — $2 per remaining discard if no discards used this round
    rarity = Rarity.COMMON
    cost = 4
    def on_round_end(self, state, js, rng):
        from ..engine import DISCARDS_PER_BLIND
        if state.discards_left == DISCARDS_PER_BLIND:
            return js, 2 * state.discards_left, False, rng
        return js, 0, False, rng


# --- Batch 4: on_discard lifecycle jokers ---

@register(JokerType.FACELESS_JOKER)
class _FacelessJoker(JokerEffect):  # wiki: /w/Faceless_Joker  — earn $5 if 3+ face cards discarded at once
    rarity = Rarity.COMMON
    cost = 4
    def on_discard(self, state, discarded, js, rng):
        rules = aggregate_rules(state.jokers) if state and state.jokers else NO_RULES
        faces = sum(1 for c in discarded if is_face(c, rules))
        return js, (5 if faces >= 3 else 0), rng


@register(JokerType.GREEN_JOKER)
class _GreenJoker(JokerEffect):  # wiki: /w/Green_Joker  — +1 Mult per hand played, -1 per discard (start 0, floor 0)
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        return Effect(mult=js.counter)
    def on_play(self, state, played, scoring_idx, rules, js):
        return dataclasses.replace(js, counter=js.counter + 1.0)
    def on_discard(self, state, discarded, js, rng):
        return dataclasses.replace(js, counter=max(0.0, js.counter - 1.0)), 0, rng


@register(JokerType.RAMEN)
class _Ramen(JokerEffect):  # wiki: /w/Ramen  — X2 Mult, -X0.01 per card discarded; eaten at 100 cards (would hit X1)
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(xmult=2.0 - 0.01 * js.counter)
    def on_discard(self, state, discarded, js, rng):
        return dataclasses.replace(js, counter=js.counter + len(discarded)), 0, rng
    def destroy_when(self, js):
        return js.counter >= 100

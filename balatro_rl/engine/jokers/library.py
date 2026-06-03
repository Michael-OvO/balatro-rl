"""The proof-set joker implementations. Each value is wiki-cited; verify against
balatrowiki.org before changing. This module registers jokers on import.
"""
from __future__ import annotations

import dataclasses

from ..cards import Enhancement
from ..hands import HandType, contains, evaluate, is_face
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


# --- Batch 5: probabilistic scoring (consume ctx.rng during a PLAY) ---

@register(JokerType.MISPRINT)
class _Misprint(JokerEffect):  # wiki: /w/Misprint  — +0..+23 Mult (uniform int), changes every hand
    rarity = Rarity.COMMON
    cost = 4
    def independent(self, ctx, js):
        roll, ctx.rng = ctx.rng.random()
        return Effect(mult=int(roll * 24))


@register(JokerType.BLOODSTONE)
class _Bloodstone(JokerEffect):  # wiki: /w/Bloodstone  — 1 in 2 chance per scored Heart -> X1.5 Mult
    rarity = Rarity.UNCOMMON
    cost = 7
    def on_score(self, ctx, card, index, js):
        if card.suit != 1:  # Hearts only; consume the rng for Hearts alone
            return Effect()
        roll, ctx.rng = ctx.rng.random()
        return Effect(xmult=1.5) if roll < 0.5 else Effect()


# --- Batch 5: per-round randomized state (on_round_start picks the target) ---

@register(JokerType.ANCIENT_JOKER)
class _AncientJoker(JokerEffect):  # wiki: /w/Ancient_Joker  — X1.5 Mult per scored card of [suit]; suit re-rolled each round
    rarity = Rarity.RARE
    cost = 8
    def on_round_start(self, state, js, rng):
        suit, rng = rng.randint(0, 3)
        return dataclasses.replace(js, counter=float(suit)), rng
    def on_score(self, ctx, card, index, js):
        return Effect(xmult=1.5) if card.suit == int(js.counter) else Effect()


@register(JokerType.THE_IDOL)
class _TheIdol(JokerEffect):  # wiki: /w/The_Idol  — X2 Mult per scored [rank] of [suit]; re-rolled each round
    rarity = Rarity.UNCOMMON
    cost = 6
    def on_round_start(self, state, js, rng):
        rank, rng = rng.randint(2, 14)
        suit, rng = rng.randint(0, 3)
        return dataclasses.replace(js, counter=float(rank * 4 + suit)), rng  # encode both
    def on_score(self, ctx, card, index, js):
        code = int(js.counter)
        rank, suit = code // 4, code % 4
        return Effect(xmult=2.0) if (card.rank == rank and card.suit == suit) else Effect()


@register(JokerType.MAIL_IN_REBATE)
class _MailInRebate(JokerEffect):  # wiki: /w/Mail-In_Rebate  — $5 per discarded [rank]; rank re-rolled each round
    rarity = Rarity.COMMON
    cost = 4
    def on_round_start(self, state, js, rng):
        rank, rng = rng.randint(2, 14)
        return dataclasses.replace(js, counter=float(rank)), rng
    def on_discard(self, state, discarded, js, rng):
        target = int(js.counter)
        money = 5 * sum(1 for c in discarded if c.rank == target)
        return js, money, rng


# --- Batch 6: hand-play-count jokers (read GameState.hand_plays_run / _round) ---

@register(JokerType.SUPERNOVA)
class _Supernova(JokerEffect):  # wiki: /w/Supernova
    # "Adds the number of times poker hand has been played this run to Mult."
    # Wiki/game: the current play IS counted (retroactive + this hand). ScoreContext
    # exposes the PRE-increment run count, so add +1 to include the hand being scored
    # (first-ever play of a hand type -> +1 Mult).
    rarity = Rarity.COMMON
    cost = 5
    def independent(self, ctx, js):
        return Effect(mult=ctx.hand_plays_run + 1)


@register(JokerType.CARD_SHARP)
class _CardSharp(JokerEffect):  # wiki: /w/Card_Sharp
    # "X3 Mult if played poker hand has already been played this round."
    # "Already this round" excludes the current play -> fire when the PRE-increment
    # round count for this hand_type is >= 1 (a prior play this round).
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(xmult=3.0) if ctx.hand_plays_round >= 1 else Effect()


@register(JokerType.OBELISK)
class _Obelisk(JokerEffect):  # wiki: /w/Obelisk
    # "This Joker gains X0.2 Mult per consecutive hand played without playing your
    # most played poker hand"; resets to X1 when you play your most-played hand.
    # Wiki: "The Obelisk resets before the hand is scored" and "Obelisk will only
    # reset once you break the tie ... (you play one of those hands one more time)".
    #
    # Key timing: the counter is UPDATED BEFORE this hand scores, so the current
    # hand reflects the post-update value (reset -> scores at X1; gain -> scores
    # with the new +X0.2 step). js.counter = number of accumulated +X0.2 steps;
    # xMult = 1 + 0.2 * counter. `independent` computes the updated counter from the
    # PRE-increment counts on the context and applies it to THIS hand; `on_play`
    # persists the same update (engine increments the run counter afterwards).
    rarity = Rarity.RARE
    cost = 8

    @staticmethod
    def _next_counter(counter, plays_run_self_pre, plays_run_max_other):
        # `plays_run_self_pre` is the run count for THIS hand_type BEFORE this play.
        cur = plays_run_self_pre + 1            # count incl. the hand being played
        # Reset only when THIS hand becomes the STRICT unique most-played hand;
        # merely tying the leader is "safe" and keeps scaling (wiki tie note).
        if cur > plays_run_max_other:
            return 0.0
        return counter + 1.0

    def independent(self, ctx, js):
        counter = self._next_counter(js.counter, ctx.hand_plays_run,
                                     ctx.hand_plays_run_max_other)
        return Effect(xmult=1.0 + 0.2 * counter)

    def on_play(self, state, played, scoring_idx, rules, js):
        # state.hand_plays_run is PRE-increment; include the hand just played.
        ht = int(evaluate(list(played), rules)[0])
        runs = list(state.hand_plays_run) if state is not None else [0] * 12
        self_pre = runs[ht] if ht < len(runs) else 0
        max_other = max((c for i, c in enumerate(runs) if i != ht), default=0)
        counter = self._next_counter(js.counter, self_pre, max_other)
        return dataclasses.replace(js, counter=counter)


# --- Batch 7 (B2a): full-deck-enhancement readers + economy-on-score ------------
# Money jokers add to ctx.money_delta directly (Effect has no money channel; this
# mirrors the Lucky enhancement and Bloodstone's in-place ctx mutation). Business
# Card / Reserved Parking consume ctx.rng ONLY for face cards, so a joker-absent or
# face-less hand draws zero extra rng -> byte-identical to the pre-Batch-7 game.

@register(JokerType.STEEL_JOKER)
class _SteelJoker(JokerEffect):  # wiki: /w/Steel_Joker  — X Mult: +0.2 per Steel card in full deck
    rarity = Rarity.UNCOMMON
    cost = 7
    def independent(self, ctx, js):
        return Effect(xmult=1.0 + 0.2 * ctx.deck_enh_counts[Enhancement.STEEL])


@register(JokerType.STONE_JOKER)
class _StoneJoker(JokerEffect):  # wiki: /w/Stone_Joker  — +25 Chips per Stone card in full deck
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(chips=25 * ctx.deck_enh_counts[Enhancement.STONE])


@register(JokerType.GOLDEN_TICKET)
class _GoldenTicket(JokerEffect):  # wiki: /w/Golden_Ticket  — played Gold-enhancement card earns $4
    rarity = Rarity.COMMON
    cost = 5
    def on_score(self, ctx, card, index, js):
        if card.enhancement == Enhancement.GOLD:
            ctx.money_delta += 4
        return Effect()


@register(JokerType.ROUGH_GEM)
class _RoughGem(JokerEffect):  # wiki: /w/Rough_Gem  — played Diamond earns $1
    rarity = Rarity.UNCOMMON
    cost = 7
    def on_score(self, ctx, card, index, js):
        if card.suit == 3:  # Diamonds
            ctx.money_delta += 1
        return Effect()


@register(JokerType.BUSINESS_CARD)
class _BusinessCard(JokerEffect):  # wiki: /w/Business_Card  — played face card has 1 in 2 to earn $2
    rarity = Rarity.COMMON
    cost = 4
    def on_score(self, ctx, card, index, js):
        if is_face(card, ctx.rules):  # respects Pareidolia via ctx.rules
            roll, ctx.rng = ctx.rng.random()
            if roll < 0.5:
                ctx.money_delta += 2
        return Effect()


@register(JokerType.RESERVED_PARKING)
class _ReservedParking(JokerEffect):  # wiki: /w/Reserved_Parking  — each held face card has 1 in 2 to earn $1
    rarity = Rarity.COMMON
    cost = 6
    def on_held(self, ctx, card, js):
        if is_face(card, ctx.rules):  # respects Pareidolia via ctx.rules
            roll, ctx.rng = ctx.rng.random()
            if roll < 0.5:
                ctx.money_delta += 1
        return Effect()


# --- Batch 8 (B2b-i): event-scaling enhancement jokers --------------------------
# Both scale a per-instance X Mult counter from this hand's enhancement events
# (HandEvents) via on_hand_events. Glass Joker scales on glass SHATTERS, which happen
# after scoring -> independent reads the persistent counter only (next-hand). Lucky Cat
# scales on lucky TRIGGERS, which fire in the scored-card phase (before the joker phase)
# -> independent adds this hand's ctx.lucky_triggers so the same hand benefits.

@register(JokerType.GLASS_JOKER)
class _GlassJoker(JokerEffect):  # wiki: /w/Glass_Joker  — gains X0.75 Mult per Glass card destroyed
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(xmult=1.0 + 0.75 * js.counter)
    def on_hand_events(self, js, events):
        return dataclasses.replace(js, counter=js.counter + events.glass_destroyed)


@register(JokerType.LUCKY_CAT)
class _LuckyCat(JokerEffect):  # wiki: /w/Lucky_Cat  — gains X0.25 Mult per Lucky card trigger
    rarity = Rarity.UNCOMMON
    cost = 6
    def independent(self, ctx, js):
        return Effect(xmult=1.0 + 0.25 * (js.counter + ctx.lucky_triggers))
    def on_hand_events(self, js, events):
        return dataclasses.replace(js, counter=js.counter + events.lucky_triggered)


# --- Batch 9 (B2b-ii): card-mutation enhancement jokers -------------------------
# Both publish a RuleFlag that score_play reads to record persistent master_deck
# enhancement overrides (applied by engine.step). Vampire also scales like Lucky Cat:
# X0.1 per enhanced card stripped this hand (same-hand via ctx.vampire_consumed, persisted
# via on_hand_events). Midas Mask is a pure mutation with no scoring/scaling effect.

@register(JokerType.VAMPIRE)
class _Vampire(JokerEffect):  # wiki: /w/Vampire  — X0.1 Mult per scored Enhanced card, removes Enhancement
    rarity = Rarity.UNCOMMON
    cost = 7
    def rules(self):
        return RuleFlags(vampire=True)
    def independent(self, ctx, js):
        return Effect(xmult=1.0 + 0.1 * (js.counter + ctx.vampire_consumed))
    def on_hand_events(self, js, events):
        return dataclasses.replace(js, counter=js.counter + events.vampire_consumed)


@register(JokerType.MIDAS_MASK)
class _MidasMask(JokerEffect):  # wiki: /w/Midas_Mask  — all scored face cards become Gold
    copyable = False
    rarity = Rarity.UNCOMMON
    cost = 7
    def rules(self):
        return RuleFlags(midas=True)

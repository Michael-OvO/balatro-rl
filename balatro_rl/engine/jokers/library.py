"""The proof-set joker implementations. Each value is wiki-cited; verify against
balatrowiki.org before changing. This module registers jokers on import.
"""
from __future__ import annotations

import dataclasses

from ..hands import is_face
from .base import Effect, JokerEffect, JokerState, JokerType, RuleFlags, register


@register(JokerType.JOKER)
class _Joker(JokerEffect):  # wiki: /w/Joker  — +4 Mult
    def independent(self, ctx, js):
        return Effect(mult=4)


@register(JokerType.CAVENDISH)
class _Cavendish(JokerEffect):  # wiki: /w/Cavendish  — X3 Mult
    def independent(self, ctx, js):
        return Effect(xmult=3.0)


@register(JokerType.GREEDY)
class _Greedy(JokerEffect):  # wiki: /w/Greedy_Joker  — +3 Mult per scored Diamond
    def on_score(self, ctx, card, index, js):
        return Effect(mult=3) if card.suit == 3 else Effect()


@register(JokerType.SCARY_FACE)
class _ScaryFace(JokerEffect):  # wiki: /w/Scary_Face  — +30 Chips per scored face card
    def on_score(self, ctx, card, index, js):
        return Effect(chips=30) if is_face(card, ctx.rules) else Effect()


@register(JokerType.PHOTOGRAPH)
class _Photograph(JokerEffect):  # wiki: /w/Photograph  — X2 on first scoring face card (re-applies on retrigger)
    def on_score(self, ctx, card, index, js):
        return Effect(xmult=2.0) if index == ctx.first_face_idx else Effect()


@register(JokerType.BARON)
class _Baron(JokerEffect):  # wiki: /w/Baron  — each King held gives X1.5 Mult
    def on_held(self, ctx, card, js):
        return Effect(xmult=1.5) if card.rank == 13 else Effect()


@register(JokerType.HACK)
class _Hack(JokerEffect):  # wiki: /w/Hack  — retrigger each played 2,3,4,5
    def retrigger(self, ctx, card, js):
        return 1 if card.rank in (2, 3, 4, 5) else 0


@register(JokerType.SPLASH)
class _Splash(JokerEffect):  # wiki: /w/Splash  — every played card scores
    copyable = False
    def rules(self):
        return RuleFlags(splash=True)


@register(JokerType.PAREIDOLIA)
class _Pareidolia(JokerEffect):  # wiki: /w/Pareidolia  — all cards are face cards
    copyable = False
    def rules(self):
        return RuleFlags(all_face=True)


@register(JokerType.RIDE_THE_BUS)
class _RideTheBus(JokerEffect):  # wiki: /w/Ride_the_Bus
    def independent(self, ctx, js):
        return Effect(mult=js.counter)

    def on_play(self, state, played, scoring_idx, rules, js):
        scored_face = any(is_face(played[i], rules) for i in scoring_idx)
        new_counter = 0.0 if scored_face else js.counter + 1.0
        return dataclasses.replace(js, counter=new_counter)


@register(JokerType.BLUEPRINT)
class _Blueprint(JokerEffect):  # wiki: /w/Blueprint  — copy resolution handled in base.resolve_providers
    pass

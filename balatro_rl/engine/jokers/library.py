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

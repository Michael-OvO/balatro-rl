"""Single-game Balatro environment. Thin pure adapter over the engine:
reset() -> (obs, mask); step(action_id) -> (obs, reward, done, info, mask).
The action is validated against the legal mask (the policy must mask).
"""
from __future__ import annotations

import numpy as np

import dataclasses

from ..engine import engine
from ..engine.state import Phase
from .actions import decode, legal_mask
from .exposure import make_exposure
from .obs import encode
from .rewards import make_reward


class BalatroEnv:
    def __init__(self, reward_name: str = "shaped", req_scale: float = 1.0,
                 enable_bosses: bool = False, enhance_rate: float = 0.0,
                 grant_planets: int = 0):
        self._reward = make_reward(reward_name)
        self._req_scale = req_scale
        self._enable_bosses = enable_bosses   # boss blinds on the boss slot (Phase D retrain)
        # Acquisition exposure for the retrain (default off -> byte-identical plain game).
        self._enhance_rate = enhance_rate     # prob each deck card starts enhanced
        self._grant_planets = grant_planets   # # of Planet consumables to start with
        self.state = None

    def set_req_scale(self, scale: float):
        """Curriculum target scale; applied at the NEXT reset (in-progress episode keeps its)."""
        self._req_scale = scale

    def reset(self, seed: int = 0):
        card_mods, consumables = make_exposure(seed, self._enhance_rate, self._grant_planets)
        self.state = engine.reset(seed, self._req_scale, card_mods=card_mods,
                                  enable_bosses=self._enable_bosses)
        if consumables:
            self.state = dataclasses.replace(self.state, consumables=consumables)
        self._reward.reset()
        return encode(self.state), legal_mask(self.state)

    def step(self, action_id: int):
        mask = legal_mask(self.state)
        assert mask[action_id], f"illegal action {action_id} (decoded {decode(action_id)})"
        prev = self.state
        verb, arg = decode(action_id)
        nxt, info = engine.step(prev, (verb, arg))
        # A boss blind can leave NO legal move (e.g. Mouth/Eye when you can't form the
        # required hand type and discards are spent). That's a stuck position = a lost
        # blind; mark it terminal so the policy never faces an all-masked state (which the
        # categorical sampler would resolve to a random ILLEGAL action). Off a boss blind
        # the hand always refills, so this never triggers in the plain game.
        if not nxt.done and not legal_mask(nxt).any():
            nxt = dataclasses.replace(nxt, done=True, won=False, phase=Phase.LOST)
            info = {**info, "result": "lost", "stuck": True}
        self.state = nxt
        reward = float(self._reward(prev, action_id, nxt, info))
        done = bool(nxt.done)
        new_mask = legal_mask(nxt) if not done else np.zeros_like(mask)
        # surface depth/score so the training loop can log antes & max scores reached
        info = {**info, "ante": int(nxt.ante), "round_score": int(nxt.round_score)}
        return encode(nxt), reward, done, info, new_mask

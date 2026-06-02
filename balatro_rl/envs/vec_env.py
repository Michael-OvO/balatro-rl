"""In-process synchronous vectorized env: N BalatroEnv stepped in a Python loop,
with auto-reset on done (the standard PPO vec-env contract). Obs are stacked into
batched numpy arrays for the JAX policy.
"""
from __future__ import annotations

import numpy as np

from .actions import NUM_ACTIONS
from .balatro_env import BalatroEnv


def _stack(obs_list: list[dict]) -> dict:
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}


class SyncVectorEnv:
    def __init__(self, num_envs: int, reward_name: str = "shaped", base_seed: int = 0,
                 req_scale: float = 1.0):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self._envs = [BalatroEnv(reward_name, req_scale) for _ in range(num_envs)]
        self._next_seed = base_seed
        self._obs = None
        self._mask = None

    def set_req_scale(self, scale: float):
        """Update the curriculum scale on EVERY sub-env; new (incl. auto-reset) episodes use it."""
        for env in self._envs:
            env.set_req_scale(scale)

    def _fresh_seed(self) -> int:
        s = self._next_seed
        self._next_seed += 1
        return s

    def reset(self):
        obs_list, masks = [], []
        for env in self._envs:
            o, m = env.reset(self._fresh_seed())
            obs_list.append(o); masks.append(m)
        self._obs = _stack(obs_list)
        self._mask = np.stack(masks, axis=0)
        return self._obs, self._mask

    def step(self, actions):
        obs_list, masks = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []
        for i, env in enumerate(self._envs):
            o, r, d, info, m = env.step(int(actions[i]))
            rewards[i] = r
            dones[i] = d
            if d:
                o, m = env.reset(self._fresh_seed())   # auto-reset; o/m are the new episode's
            obs_list.append(o); masks.append(m)
            infos.append(info)
        self._obs = _stack(obs_list)
        self._mask = np.stack(masks, axis=0)
        return self._obs, rewards, dones, infos, self._mask

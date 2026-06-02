"""Run an agent for one episode and record a replayable Trajectory.

A Trajectory is (seed, [action_ids]) plus a light summary. Because the engine is
a pure function of (seed, actions), the whole game reconstructs from those — the
basis for the Plan-5 replay viewer and parity tests.
"""
from __future__ import annotations

import dataclasses
import json

from ..engine import engine
from .actions import decode
from .balatro_env import BalatroEnv

_MAX_STEPS = 10_000


@dataclasses.dataclass
class Trajectory:
    seed: int
    actions: list[int]
    total_reward: float
    final_ante: int
    won: bool

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @staticmethod
    def load(path) -> "Trajectory":
        with open(path) as f:
            return Trajectory(**json.load(f))


def run_episode(env: BalatroEnv, agent, seed: int) -> Trajectory:
    obs, mask = env.reset(seed)
    actions: list[int] = []
    total = 0.0
    done = False
    for _ in range(_MAX_STEPS):
        if done:
            break
        a = int(agent.act(env.state, mask))
        actions.append(a)
        obs, reward, done, info, mask = env.step(a)
        total += reward
    return Trajectory(seed=seed, actions=actions, total_reward=total,
                      final_ante=env.state.ante, won=bool(env.state.won))


def replay(traj: Trajectory):
    """Reconstruct the final GameState from (seed, actions) — engine determinism."""
    state = engine.reset(traj.seed)
    for a in traj.actions:
        state, _ = engine.step(state, decode(a))
    return state

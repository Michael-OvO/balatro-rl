"""BalatroTextEnv: a single-game TEXT environment over the engine, for LLM agents.

    reset(seed, req_scale) -> (observation_text, info)
    step(action_text)      -> (observation_text, reward, done, info)

Wraps the existing BalatroEnv (reward, masking, terminal handling) and the M1 text
boundary (serialize_state + render_menu for the observation; parse_action for the action).
verl-AGNOSTIC and fully unit-testable; verl-agent's BalatroEnvManager (verl_env.py)
vectorizes this and uses parse_action as its projection_f.

On an unparseable / illegal action the engine is NOT stepped: reward 0.0, the same
observation is re-presented, and info["is_action_valid"] = False so the trainer can apply
its invalid-action penalty (the model learns to emit valid actions). The rollout's
max_steps bounds any garbage loop. Curriculum lives in req_scale on reset (start low so
early GRPO rollouts clear some blinds -> reward variance -> a learning signal).
"""
from __future__ import annotations

from ..envs.balatro_env import BalatroEnv
from .actions_text import build_menu, parse_action, render_menu
from .serialize import serialize_state


class BalatroTextEnv:
    def __init__(self, reward_name: str = "shaped", enable_bosses: bool = False):
        self._reward_name = reward_name
        self._enable_bosses = enable_bosses
        self._env: BalatroEnv | None = None

    @property
    def state(self):
        return self._env.state

    def _obs_text(self) -> str:
        state = self._env.state
        return serialize_state(state) + "\n\n" + render_menu(build_menu(state))

    def reset(self, seed: int = 0, req_scale: float = 1.0) -> tuple[str, dict]:
        self._env = BalatroEnv(reward_name=self._reward_name, req_scale=req_scale,
                               enable_bosses=self._enable_bosses)
        self._env.reset(seed)
        s = self._env.state
        info = {"is_action_valid": True, "won": bool(s.won), "ante": int(s.ante),
                "cleared": False}
        return self._obs_text(), info

    def step(self, action_text: str) -> tuple[str, float, bool, dict]:
        state = self._env.state
        res = parse_action(action_text, state)
        if res.error is not None:
            # Invalid action: do NOT step the engine; re-present the same observation and
            # flag it so the trainer's invalid-action penalty fires. The model learns to
            # produce valid JSON; max_steps bounds repeated failures.
            info = {"is_action_valid": False, "won": bool(state.won), "ante": int(state.ante),
                    "cleared": False, "parse_error": res.error}
            return self._obs_text(), 0.0, bool(state.done), info
        _, reward, done, info, _ = self._env.step(res.action_id)
        nxt = self._env.state
        info = {**info, "is_action_valid": True, "won": bool(nxt.won), "ante": int(nxt.ante)}
        return self._obs_text(), float(reward), bool(done), info

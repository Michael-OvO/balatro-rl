"""BalatroTextEnv: a single-game TEXT environment over the engine, for LLM agents.

    reset(seed, req_scale) -> (observation_text, info)
    step(action_text)      -> (observation_text, reward, done, info)

Wraps the existing BalatroEnv (reward, masking, terminal handling) and the M1 text boundary
(`observation_text` = serialize_state + render_menu for the observation; `parse_action` for the
action). verl-AGNOSTIC and fully unit-testable; verl-agent's BalatroEnvManager (verl_env.py)
vectorizes this and uses parse_action as its projection_f.

On an unparseable / illegal action the engine is NOT stepped: reward 0.0, the SAME (cached)
observation is re-presented, and info["is_action_valid"] = False so the trainer can apply its
invalid-action penalty (the model learns to emit valid actions). The rollout's max_steps bounds
any garbage loop.

info["cleared"] is a cumulative "cleared >= 1 blind this episode" flag (the curriculum's signal):
it latches True the first time a step clears a blind, so success_evaluator can read it from any
later step. (`ante` only advances after a full 3-blind ante, so it is the wrong proxy for the
"did this rollout clear a blind" signal the req_scale curriculum needs.)

Curriculum lives in req_scale on reset (start low so early GRPO rollouts clear some blinds ->
reward variance -> a learning signal).
"""
from __future__ import annotations

from ..envs.actions import legal_mask
from ..envs.balatro_env import BalatroEnv
from .actions_text import build_menu, observation_text, parse_action


class BalatroTextEnv:
    def __init__(self, reward_name: str = "shaped", enable_bosses: bool = False):
        self._reward_name = reward_name
        self._enable_bosses = enable_bosses
        self._env: BalatroEnv | None = None
        self._last_obs = ""
        self._last_menu = None          # menu the cached obs was built from (current state)
        self._cleared_any = False

    @property
    def state(self):
        return self._env.state

    def _refresh_obs(self) -> None:
        """Recompute the cached observation (+ its menu) for the current engine state."""
        self._last_menu = build_menu(self._env.state)
        self._last_obs = observation_text(self._env.state, self._last_menu)

    def _info(self, valid: bool, **extra) -> dict:
        s = self._env.state
        return {"is_action_valid": valid, "won": bool(s.won), "ante": int(s.ante),
                "cleared": self._cleared_any, **extra}

    def reset(self, seed: int = 0, req_scale: float = 1.0) -> tuple[str, dict]:
        self._env = BalatroEnv(reward_name=self._reward_name, req_scale=req_scale,
                               enable_bosses=self._enable_bosses)
        self._env.reset(seed)
        self._cleared_any = False
        self._refresh_obs()
        return self._last_obs, self._info(valid=True)

    def step(self, action_text: str) -> tuple[str, float, bool, dict]:
        state = self._env.state
        # Reuse the menu the cached obs (the prompt the agent saw) was built from, so the menu
        # SHOWN == the menu VALIDATED; parse_action then does no menu rebuild.
        res = parse_action(action_text, state, menu=self._last_menu, mask=legal_mask(state))
        if res.error is not None:
            # Invalid: no engine step; re-present the cached obs (state unchanged) so the
            # trainer's invalid-action penalty fires. max_steps bounds repeated failures.
            return self._last_obs, 0.0, bool(state.done), self._info(valid=False, parse_error=res.error)
        return self.apply_action(res.action_id)

    def apply_action(self, action_id) -> tuple[str, float, bool, dict]:
        """Step the engine with a PRE-VALIDATED action_id (or None = invalid no-op). Used by the
        vectorized verl path, where parsing happens once in the projection function and the env
        just applies the result. None -> the cached obs is re-presented, reward 0.0, and
        is_action_valid=False so the trainer's invalid-action penalty fires."""
        if action_id is None:
            return self._last_obs, 0.0, bool(self._env.state.done), self._info(valid=False)
        _, reward, done, info, _ = self._env.step(action_id)
        if info.get("cleared"):
            self._cleared_any = True    # latch the cumulative "cleared >= 1 blind" curriculum signal
        self._refresh_obs()
        return self._last_obs, float(reward), bool(done), self._info(valid=True)

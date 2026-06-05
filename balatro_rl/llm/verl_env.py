"""BalatroEnvManager — the verl-agent EnvironmentManager that vectorizes BalatroTextEnv for
multi-turn GRPO/GiGPO.

POD-ONLY. It subclasses verl-agent's `agent_system.environments.base.EnvironmentManagerBase`,
which is NOT installed on a CPU dev box, so the import is GUARDED: importing this module
without verl-agent yields a base of `object` and `HAVE_VERL_AGENT == False` (so the rest of
balatro_rl, and CI, are unaffected). This module is intentionally NOT re-exported from
balatro_rl.llm.__init__.

The mapping is the part that is genuinely ours and grounded in M1:
  - observation  : serialize_state + render_menu               (via BalatroTextEnv._obs_text)
  - projection_f : parse_action  (text reply -> validated action_id, or invalid -> penalty)
  - per-step reward : envs.rewards `shaped`                     (via BalatroTextEnv.step)
  - success      : info["won"]                                  (env already surfaces it)
  - anchor       : the observation text (GiGPO groups identical states across rollouts)
  - curriculum   : ReqScaleCurriculum.current applied at reset; ramped from success_rate

VERIFY ON THE POD: verl-agent's base-class constructor signature and the make_envs/registry
wiring drift across versions. Treat the first launch as a smoke test (see docs/RUNPOD_M2.md),
not a trusted run. Adapt the reset/step return shapes to the installed version if needed.
"""
from __future__ import annotations

import numpy as np

from .curriculum import ReqScaleCurriculum
from .text_env import BalatroTextEnv

try:
    from agent_system.environments.base import EnvironmentManagerBase
    HAVE_VERL_AGENT = True
except Exception:                       # verl-agent not installed (CPU dev box / CI)
    EnvironmentManagerBase = object
    HAVE_VERL_AGENT = False


class BalatroEnvManager(EnvironmentManagerBase):
    """Manages `n` parallel single-game BalatroTextEnv for one GRPO group/batch."""

    def __init__(self, n: int, reward_name: str = "shaped", enable_bosses: bool = False,
                 req_scale_start: float = 0.1, req_scale_end: float = 1.0, seed: int = 0,
                 **base_kwargs):
        if HAVE_VERL_AGENT:
            super().__init__(**base_kwargs)   # init the real base whenever verl-agent is present
        self._envs = [BalatroTextEnv(reward_name, enable_bosses) for _ in range(n)]
        self._n = n
        self._seed = seed
        self._curriculum = ReqScaleCurriculum(start=req_scale_start, end=req_scale_end)
        self._obs_cache: list[str] = [""] * n

    # --- verl-agent EnvironmentManager surface ---

    def reset(self, kwargs=None):
        scale = self._curriculum.current
        seed = self._seed                 # SAME seed for every env in the group (see below)
        infos = []
        for i, env in enumerate(self._envs):
            # All n rollouts in a GRPO/GiGPO group MUST face the same game: the advantage is
            # group-relative (subtracts the group mean as a baseline), valid only when the task
            # is shared — the variance comes from temperature sampling, not from different decks.
            # GiGPO's anchor-based step grouping likewise needs identical states across rollouts.
            obs, info = env.reset(seed=seed, req_scale=scale)
            self._obs_cache[i] = obs
            infos.append(info)
        self._seed += 1                   # next group -> next task/seed
        return {"text": list(self._obs_cache), "image": None, "anchor": list(self._obs_cache)}, infos

    def step(self, text_actions):
        rewards, dones, infos = [], [], []
        for i, (env, action) in enumerate(zip(self._envs, text_actions)):
            obs, reward, done, info = env.step(action)
            self._obs_cache[i] = obs
            rewards.append(reward)
            dones.append(done)
            infos.append(info)            # info["is_action_valid"] drives use_invalid_action_penalty
        return (
            {"text": list(self._obs_cache), "image": None, "anchor": list(self._obs_cache)},
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )

    def build_text_obs(self):
        return list(self._obs_cache)

    def success_evaluator(self, **kwargs):
        """Win-rate over the group's finished episodes (the deploy metric + curriculum signal).
        Reads info["won"] from the last active step of each rollout, matching the verl-agent
        EnvironmentManager contract (total_batch_list / total_infos)."""
        total_batch_list = kwargs["total_batch_list"]
        total_infos = kwargs["total_infos"]
        results = []
        for bs in range(len(total_batch_list)):
            for i in reversed(range(len(total_batch_list[bs]))):
                if total_batch_list[bs][i].get("active_masks"):
                    info = total_infos[bs][i]
                    results.append(float(info.get("won", 0.0)))
                    # Curriculum signal = "cleared >= 1 blind this episode" (the cumulative flag
                    # BalatroTextEnv latches into info["cleared"]), NOT ante>1 which needs a full
                    # 3-blind ante and would stall the ramp at its floor.
                    self._curriculum.record(cleared=bool(info.get("cleared", False)))
                    break
        self._curriculum.maybe_ramp()      # ramp req_scale for the NEXT reset on rolling clear-rate
        return {"success_rate": np.asarray(results, dtype=np.float32)}

    def close(self):
        self._envs = []

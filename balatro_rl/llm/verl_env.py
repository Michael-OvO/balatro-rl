"""verl-agent integration for multi-turn GRPO/GiGPO on Balatro.

Mirrors verl-agent's real env interface (verified against agent_system/environments/base.py +
env_manager.py on the pod), so the pod-side hook is just 3 lines in make_envs (see
docs/RUNPOD_M2.md) and ALL the substance lives here, in the committed repo:

    EnvironmentManagerBase(envs, projection_f, config)        # base contract
      self.envs : vectorized env  .reset()->(obs,infos)  .step(actions)->(obs,rews,dones,infos)
      projection_f(text_actions, states) -> (action_ids, valids)

Mapping (all from M1):
  - vectorized env   = BalatroVecEnv (N in-process BalatroTextEnv; Balatro is CPU-cheap, no Ray)
  - projection_f     = balatro_projection = parse_action per env (text -> validated action_id)
  - observation      = observation_text (serialize_state + render_menu), as obs["text"]/["anchor"]
  - per-step reward  = the `shaped` reward (dense)
  - success/win-rate = info["won"] (base success_evaluator default)
  - curriculum       = ReqScaleCurriculum ramps req_scale from the rolling clear-rate

The base class is import-guarded: without verl-agent installed it falls back to `object`, and the
manager fully overrides reset/step (which never call super), so the whole stack — vec env,
projection, manager — is exercised by local unit tests with NO verl-agent dependency. verl-agent
is needed only to run the trainer on the pod.
"""
from __future__ import annotations

from .actions_text import observation_text, parse_action
from .curriculum import ReqScaleCurriculum
from .text_env import BalatroTextEnv

try:
    from agent_system.environments.base import EnvironmentManagerBase, to_numpy
    HAVE_VERL_AGENT = True
except Exception:                       # verl-agent absent (CPU dev box / CI)
    HAVE_VERL_AGENT = False
    EnvironmentManagerBase = object

    def to_numpy(x):
        import numpy as np
        return np.asarray(x)


class BalatroVecEnv:
    """In-process vectorized Balatro env: `env_num` task-groups x `group_n` rollouts. Each group
    shares ONE seed so a GRPO/GiGPO group faces the same game (variance comes from the policy's
    temperature sampling, not from different decks) — the group-relative advantage baseline and
    GiGPO anchor grouping both require this. Balatro steps are microseconds, so no Ray/workers."""

    def __init__(self, seed: int, env_num: int, group_n: int, reward_name: str = "shaped",
                 enable_bosses: bool = False, req_scale: float = 1.0):
        self._base_seed = int(seed)
        self._group_n = max(1, int(group_n))
        self._req_scale = float(req_scale)
        self._envs = [BalatroTextEnv(reward_name, enable_bosses)
                      for _ in range(int(env_num) * self._group_n)]

    def __len__(self):
        return len(self._envs)

    def set_req_scale(self, scale: float) -> None:
        self._req_scale = float(scale)

    def reset(self, kwargs=None):
        obs, infos = [], []
        for i, env in enumerate(self._envs):
            seed = self._base_seed + (i // self._group_n)   # same seed across a group
            o, info = env.reset(seed=seed, req_scale=self._req_scale)
            obs.append(o)
            infos.append(info)
        return obs, infos

    def step(self, action_ids):
        obs, rewards, dones, infos = [], [], [], []
        for env, aid in zip(self._envs, action_ids):
            o, r, d, info = env.apply_action(aid)   # aid None (invalid) -> no-op, reward 0
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return obs, rewards, dones, infos

    def get_states(self):
        return [env.state for env in self._envs]

    def text_obs(self):
        return [env._last_obs for env in self._envs]

    def close(self):
        self._envs = []


def balatro_projection(text_actions, states):
    """Map text replies -> (action_ids, valids) by parsing each against its env's current state.
    action_id is None when the reply is unparseable or illegal (the env no-ops and the trainer's
    invalid-action penalty fires); valids[i] tells the manager what to put in info."""
    action_ids, valids = [], []
    for text, state in zip(text_actions, states):
        res = parse_action(text, state)
        action_ids.append(res.action_id)            # None on error
        valids.append(res.error is None)
    return action_ids, valids


def _obs_dict(text_list):
    return {"text": list(text_list), "image": None, "anchor": list(text_list)}


class BalatroEnvManager(EnvironmentManagerBase):
    """verl-agent EnvironmentManager for Balatro. Fully overrides reset/step (so it works with the
    object-base fallback for local tests), formats the text observation, threads game state into
    the projection, and ramps the curriculum from finished-episode clear-rate."""

    def __init__(self, envs, projection_f, config):
        if HAVE_VERL_AGENT:
            super().__init__(envs, projection_f, config)
        else:                                        # object base: set the attrs the base would
            self.envs, self.projection_f, self.config = envs, projection_f, config
        b = config.env.balatro
        self._curriculum = ReqScaleCurriculum(start=b.req_scale_start, end=b.req_scale_end)
        self.envs.set_req_scale(self._curriculum.current)

    def reset(self, kwargs=None):
        self.envs.set_req_scale(self._curriculum.current)
        obs, infos = self.envs.reset(kwargs)
        return _obs_dict(obs), infos

    def step(self, text_actions):
        action_ids, valids = self.projection_f(text_actions, self.envs.get_states())
        obs, rewards, dones, infos = self.envs.step(action_ids)
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        return _obs_dict(obs), to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self):
        return self.envs.text_obs()

    def success_evaluator(self, *args, **kwargs):
        """Record the curriculum signal (cleared >= 1 blind) from each finished episode and ramp
        req_scale, then return the win-rate. Delegates the win-rate to the verl-agent base when
        present; computes it directly in the standalone (test) path."""
        total_batch_list = kwargs["total_batch_list"]
        total_infos = kwargs["total_infos"]
        for bs in range(len(total_batch_list)):
            for i in reversed(range(len(total_batch_list[bs]))):
                if total_batch_list[bs][i].get("active_masks"):
                    self._curriculum.record(cleared=bool(total_infos[bs][i].get("cleared", False)))
                    break
        self._curriculum.maybe_ramp()
        if HAVE_VERL_AGENT:
            return super().success_evaluator(*args, **kwargs)
        import numpy as np
        rates = []
        for bs in range(len(total_batch_list)):
            for i in reversed(range(len(total_batch_list[bs]))):
                if total_batch_list[bs][i].get("active_masks"):
                    rates.append(float(total_infos[bs][i].get("won", 0.0)))
                    break
        return {"success_rate": np.array(rates)}

    def close(self):
        self.envs.close()


def _vec_from_config(config, env_num, group_n):
    b = config.env.balatro
    return BalatroVecEnv(seed=int(config.env.seed), env_num=int(env_num), group_n=int(group_n),
                         reward_name=b.reward_name, enable_bosses=b.enable_bosses,
                         req_scale=b.req_scale_start)


def make_balatro_envs(config):
    """Factory matching verl-agent's make_envs contract -> (train_manager, val_manager).
    The pod-side hook in agent_system/environments/env_manager.py is just:

        if "balatro" in config.env.env_name.lower():
            from balatro_rl.llm.verl_env import make_balatro_envs
            return make_balatro_envs(config)
    """
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    train = _vec_from_config(config, config.data.train_batch_size, group_n)
    val = _vec_from_config(config, config.data.val_batch_size, 1)
    return (BalatroEnvManager(train, balatro_projection, config),
            BalatroEnvManager(val, balatro_projection, config))

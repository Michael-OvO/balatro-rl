"""Curriculum scale plumbing through BalatroEnv + SyncVectorEnv."""
from balatro_rl.engine.blinds import required_score
from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.vec_env import SyncVectorEnv


def test_balatro_env_applies_req_scale_at_reset():
    env = BalatroEnv(req_scale=0.2)
    env.reset(0)
    assert env.state.required == required_score(1, 0, 0.2)   # 60
    env.set_req_scale(1.0)
    env.reset(0)
    assert env.state.required == 300                          # next reset picks up the new scale


def test_step_info_surfaces_ante_and_round_score():
    import numpy as np
    env = BalatroEnv()
    _o, mask = env.reset(1)
    a = int(np.flatnonzero(mask)[0])
    _o, _r, _d, info, _m = env.step(a)
    assert info["ante"] == env.state.ante           # depth reached (for train-time logging)
    assert info["round_score"] == env.state.round_score


def test_vec_env_set_req_scale_updates_every_subenv():
    ve = SyncVectorEnv(num_envs=4, req_scale=0.2)
    ve.reset()
    assert all(e.state.required == 60 for e in ve._envs)
    ve.set_req_scale(0.5)
    assert all(e._req_scale == 0.5 for e in ve._envs)         # every sub-env updated
    ve.reset()
    assert all(e.state.required == required_score(1, 0, 0.5) for e in ve._envs)   # 150

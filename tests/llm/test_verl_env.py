"""Local tests for the verl-agent integration. The whole Balatro env stack is pure Python and
the manager fully overrides reset/step, so it runs WITHOUT verl-agent installed (base = object)."""
import types

from balatro_rl.engine.cards import card_str
from balatro_rl.llm.verl_env import (BalatroEnvManager, BalatroVecEnv, balatro_projection,
                                      make_balatro_envs)


def _cfg(train_bs=2, val_bs=1, n=3, seed=0, start=0.1, end=1.0):
    return types.SimpleNamespace(
        data=types.SimpleNamespace(train_batch_size=train_bs, val_batch_size=val_bs),
        env=types.SimpleNamespace(
            seed=seed, rollout=types.SimpleNamespace(n=n),
            balatro=types.SimpleNamespace(reward_name="shaped", enable_bosses=False,
                                          req_scale_start=start, req_scale_end=end)))


def _hand(env):
    return tuple(card_str(c) for c in env.state.hand)


def test_vec_env_groups_share_a_seed():
    # env_num=2 groups x group_n=3 -> 6 envs; envs in a group share a seed (same deck/hand),
    # different groups differ. This is the GRPO grouping invariant.
    vec = BalatroVecEnv(seed=0, env_num=2, group_n=3)
    vec.reset()
    g0 = [_hand(vec._envs[i]) for i in range(3)]
    g1 = [_hand(vec._envs[i]) for i in range(3, 6)]
    assert g0[0] == g0[1] == g0[2]          # same seed within group 0 -> identical game
    assert g1[0] == g1[1] == g1[2]          # same within group 1
    assert g0[0] != g1[0]                   # different groups -> different game


def test_projection_parses_valid_and_flags_invalid():
    vec = BalatroVecEnv(seed=0, env_num=1, group_n=2)
    vec.reset()
    states = vec.get_states()
    action_ids, valids = balatro_projection(['{"action": "play", "cards": [0]}', "no json"], states)
    assert valids == [True, False]
    assert action_ids[0] is not None and action_ids[1] is None


def test_manager_reset_and_step_standalone():
    cfg = _cfg(train_bs=2, val_bs=1, n=2)
    vec = BalatroVecEnv(seed=0, env_num=2, group_n=2)
    mgr = BalatroEnvManager(vec, balatro_projection, cfg)
    obs, infos = mgr.reset()
    assert set(obs) == {"text", "image", "anchor"}
    assert len(obs["text"]) == 4 and isinstance(obs["text"][0], str)
    # one valid play + the rest garbage -> step returns aligned arrays + is_action_valid flags
    actions = ['{"action": "play", "cards": [0]}'] + ["garbage"] * 3
    nobs, rewards, dones, infos = mgr.step(actions)
    assert len(rewards) == 4 and len(dones) == 4
    assert bool(infos[0]["is_action_valid"]) is True
    assert bool(infos[1]["is_action_valid"]) is False


def test_make_balatro_envs_builds_train_and_val_managers():
    cfg = _cfg(train_bs=2, val_bs=1, n=3)
    train, val = make_balatro_envs(cfg)
    assert isinstance(train, BalatroEnvManager) and isinstance(val, BalatroEnvManager)
    assert len(train.envs) == 2 * 3            # train_batch_size x group_n
    assert len(val.envs) == 1 * 1              # val: group_n collapses to 1


def test_success_evaluator_returns_win_rate_and_records_curriculum():
    cfg = _cfg()
    mgr = BalatroEnvManager(BalatroVecEnv(seed=0, env_num=1, group_n=2), balatro_projection, cfg)
    out = mgr.success_evaluator(
        total_batch_list=[[{"active_masks": True}], [{"active_masks": True}]],
        total_infos=[[{"won": 1.0, "cleared": True}], [{"won": 0.0, "cleared": False}]])
    assert list(out["success_rate"]) == [1.0, 0.0]

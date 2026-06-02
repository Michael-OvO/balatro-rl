import jax, jax.numpy as jnp
import numpy as np
from balatro_rl.engine.engine import reset, Verb
from balatro_rl.envs.actions import encode_action
from balatro_rl.agent.networks import ActorCritic
from balatro_rl.agent.spec import dummy_obs
from balatro_rl.envs.actions import NUM_ACTIONS
from balatro_rl.viz.replay_data import (
    action_label, render_board, replay_states, record_agent_episode, save_episode, load_episode,
)


def test_action_label_covers_verbs():
    assert "PLAY" in action_label(encode_action(Verb.PLAY, (0, 1)))
    assert action_label(encode_action(Verb.REROLL, 0)) == "REROLL"
    assert action_label(encode_action(Verb.LEAVE_SHOP, 0)) == "LEAVE SHOP"
    assert "BUY" in action_label(encode_action(Verb.BUY, 1))


def test_render_board_has_key_fields():
    txt = render_board(reset(seed=1))
    assert "Ante 1" in txt and "Hand:" in txt and "Jokers:" in txt and "/300" in txt


def test_replay_states_reconstructs_deterministically():
    seed = 7
    # a short scripted action sequence (discard then play)
    s = reset(seed)
    a0 = encode_action(Verb.DISCARD, (0,))
    a1 = encode_action(Verb.PLAY, (0, 1))
    states = replay_states(seed, [a0, a1])
    assert len(states) == 3                      # before a0, before a1, terminal-ish
    # re-running yields identical states (engine determinism)
    states2 = replay_states(seed, [a0, a1])
    assert all(x.hand == y.hand and x.round_score == y.round_score
               for x, y in zip(states, states2))


def _net_params(d=32):
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d)
    p = net.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in dummy_obs(1).items()},
                 jnp.ones((1, NUM_ACTIONS), bool))
    return net, p


def test_record_agent_episode_step_dicts():
    net, p = _net_params()
    steps = record_agent_episode(net, p, seed=3, reward_name="max_depth")
    assert len(steps) > 0
    s0 = steps[0]
    assert set(["t", "ante", "blind", "phase", "money", "board", "action_id",
                "action_label", "reward", "value", "top_probs"]).issubset(s0.keys())
    # top_probs is a list of [label, prob]; probs are valid
    assert all(0.0 <= p_ <= 1.0 for _lbl, p_ in s0["top_probs"])
    # recorded actions replay back to the same terminal state
    actions = [st["action_id"] for st in steps]
    assert replay_states(3, actions)[-1].done


def test_record_is_deterministic_greedy(tmp_path):
    net, p = _net_params()
    a = record_agent_episode(net, p, seed=5, reward_name="max_depth")
    b = record_agent_episode(net, p, seed=5, reward_name="max_depth")
    assert [s["action_id"] for s in a] == [s["action_id"] for s in b]
    path = tmp_path / "ep.json"
    save_episode(a, path)
    assert [s["action_id"] for s in load_episode(path)] == [s["action_id"] for s in a]

import jax, jax.numpy as jnp
import numpy as np
from balatro_rl.agent.networks import ActorCritic
from balatro_rl.agent.spec import dummy_obs
from balatro_rl.agent.eval import evaluate
from balatro_rl.envs.actions import NUM_ACTIONS


def _params(d=32):
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d)
    p = net.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in dummy_obs(1).items()},
                 jnp.ones((1, NUM_ACTIONS), bool))
    return net, p


def test_evaluate_returns_metric_keys():
    net, p = _params()
    m = evaluate(net, p, seeds=[0, 1, 2], reward_name="max_depth")
    assert set(m.keys()) == {"eval/mean_ante", "eval/max_ante", "eval/win_rate",
                             "eval/mean_run_chips", "eval/mean_ep_len",
                             "eval/mean_blinds_cleared", "eval/max_blinds_cleared",
                             "eval/blind1_clear_rate"}
    assert all(np.isfinite(v) for v in m.values())
    assert m["eval/mean_ante"] >= 1.0          # every run reaches at least ante 1
    assert 0.0 <= m["eval/win_rate"] <= 1.0
    assert 0.0 <= m["eval/blind1_clear_rate"] <= 1.0


def test_evaluate_is_deterministic():
    net, p = _params()
    a = evaluate(net, p, seeds=[5, 6], reward_name="max_depth")
    b = evaluate(net, p, seeds=[5, 6], reward_name="max_depth")
    assert a == b                              # greedy policy + fixed seeds -> identical

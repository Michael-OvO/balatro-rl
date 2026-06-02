import jax, jax.numpy as jnp
import numpy as np
from balatro_rl.agent.networks import ActorCritic
from balatro_rl.agent.spec import dummy_obs
from balatro_rl.agent.value_head import NBINS
from balatro_rl.envs.actions import NUM_ACTIONS


def _init(B=4, d=32):
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d)
    obs = {k: jnp.asarray(v) for k, v in dummy_obs(B).items()}
    mask = jnp.ones((B, NUM_ACTIONS), dtype=bool)
    params = net.init(jax.random.PRNGKey(0), obs, mask)
    return net, params, obs, mask


def test_forward_shapes():
    net, params, obs, mask = _init()
    logits, value_logits = net.apply(params, obs, mask)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value_logits.shape == (4, NBINS)
    assert np.all(np.isfinite(np.asarray(logits)))
    assert np.all(np.isfinite(np.asarray(value_logits)))


def test_illegal_actions_get_min_logit():
    net, params, obs, _ = _init()
    mask = jnp.zeros((4, NUM_ACTIONS), dtype=bool).at[:, 0].set(True)  # only action 0 legal
    logits, _ = net.apply(params, obs, mask)
    probs = jax.nn.softmax(logits, -1)
    assert np.allclose(np.asarray(probs[:, 0]), 1.0, atol=1e-4)        # all mass on the legal action
    assert np.asarray(probs[:, 1:]).max() < 1e-4


def test_batch_one_works():
    net, params, _, _ = _init(B=1)
    obs = {k: jnp.asarray(v) for k, v in dummy_obs(1).items()}
    logits, value_logits = net.apply(params, obs, jnp.ones((1, NUM_ACTIONS), bool))
    assert logits.shape == (1, NUM_ACTIONS)

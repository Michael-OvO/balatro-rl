import jax, jax.numpy as jnp
import numpy as np
from balatro_rl.agent.ppo import sample_action, log_prob, masked_entropy, gae


def test_sample_respects_mask():
    logits = jnp.zeros((6, 10)).at[:, 3].set(0.0)
    mask = jnp.zeros((6, 10), bool).at[:, 3].set(True)
    masked = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    a = sample_action(masked, jax.random.PRNGKey(1))
    assert np.all(np.asarray(a) == 3)          # only legal action sampled


def test_log_prob_matches_softmax():
    logits = jnp.array(np.random.randn(4, 7), dtype=jnp.float32)
    actions = jnp.array([0, 1, 2, 6])
    lp = log_prob(logits, actions)
    ref = jax.nn.log_softmax(logits, -1)[jnp.arange(4), actions]
    assert np.allclose(np.asarray(lp), np.asarray(ref), atol=1e-5)


def test_masked_entropy_ignores_illegal():
    # Two legal actions with equal logits -> entropy = ln(2); illegal logits don't contribute.
    logits = jnp.array([[0.0, 0.0, 5.0]])
    mask = jnp.array([[True, True, False]])
    masked = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    ent = masked_entropy(masked, mask)
    assert np.allclose(np.asarray(ent), np.log(2.0), atol=1e-4)


def test_gae_shapes_and_finite():
    T, N = 5, 3
    rew = jnp.ones((T, N)); val = jnp.ones((T, N)); done = jnp.zeros((T, N))
    adv, tgt = gae(rew, val, done, last_value=jnp.zeros((N,)), gamma=0.99, lam=0.95)
    assert adv.shape == (T, N) and tgt.shape == (T, N)
    assert np.all(np.isfinite(np.asarray(adv)))


def test_ppo_loss_finite():
    import jax
    from balatro_rl.agent.networks import ActorCritic
    from balatro_rl.agent.spec import dummy_obs
    from balatro_rl.agent.ppo import ppo_loss
    from balatro_rl.envs.actions import NUM_ACTIONS
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=32)
    B = 8
    obs = {k: jnp.asarray(v) for k, v in dummy_obs(B).items()}
    mask = jnp.ones((B, NUM_ACTIONS), bool)
    params = net.init(jax.random.PRNGKey(0), obs, mask)
    mb = dict(obs=obs, masks=mask, actions=jnp.zeros((B,), jnp.int32),
              old_logp=jnp.zeros((B,)), adv=jnp.ones((B,)), targets=jnp.full((B,), 100.0))
    loss, aux = ppo_loss(params, net.apply, mb)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(float(x)) for x in aux)

import jax.numpy as jnp
import numpy as np
from balatro_rl.agent.value_head import symlog, symexp, two_hot, value_decode, value_loss, NBINS, BINS


def test_symexp_inverts_symlog():
    xs = jnp.array([0.0, 5.0, -5.0, 100.0, 1e6, -1e6])
    assert np.allclose(np.asarray(symexp(symlog(xs))), np.asarray(xs), rtol=1e-4, atol=1e-3)


def test_two_hot_rows_sum_to_one():
    th = two_hot(jnp.array([0.0, 12.3, -7.0, 1e5]))
    assert th.shape[-1] == NBINS
    assert np.allclose(np.asarray(th.sum(-1)), 1.0, atol=1e-5)


def test_two_hot_mean_reconstructs_target():
    s = jnp.array([0.0, 50.0, 5000.0, -300.0, 1e6])
    recon = symexp((two_hot(s) * BINS).sum(-1))   # encode then decode-from-probs
    assert np.allclose(np.asarray(recon), np.asarray(s), rtol=1e-3, atol=1e-2)


def test_value_loss_nonnegative_finite():
    logits = jnp.zeros((4, NBINS))
    loss = value_loss(logits, jnp.array([10.0, -10.0, 1000.0, 0.0]))
    assert loss.shape == (4,)
    assert np.all(np.asarray(loss) >= 0) and np.all(np.isfinite(np.asarray(loss)))


def test_value_decode_finite():
    logits = jnp.array(np.random.randn(8, NBINS), dtype=jnp.float32)
    v = value_decode(logits)
    assert v.shape == (8,) and np.all(np.isfinite(np.asarray(v)))

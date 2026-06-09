"""Gate A: component parity of score_with_jokers vs engine.scoring.score_play."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from balatro_rl.engine_jax.jokers import score_with_jokers
from balatro_rl.engine_jax.scoring import score_core


def _empty_loadout():
    from balatro_rl.envs.actions import MAX_JOKERS
    return jnp.zeros((MAX_JOKERS,), dtype=jnp.int32)


def _pad5(ranks, suits):
    r = np.zeros(5, np.int8); s = np.zeros(5, np.int8); m = np.zeros(5, bool)
    for i, (rk, su) in enumerate(zip(ranks, suits)):
        r[i] = rk; s[i] = su; m[i] = True
    return jnp.asarray(r), jnp.asarray(s), jnp.asarray(m)


def test_empty_loadout_reduces_to_score_core():
    """With no jokers, score_with_jokers == score_core for random plain hands."""
    rng = np.random.default_rng(0)
    h_r = jnp.zeros(8, jnp.int8); h_s = jnp.zeros(8, jnp.int8); h_m = jnp.zeros(8, bool)
    levels = jnp.ones(12, jnp.int32)
    jk = _empty_loadout()
    for _ in range(300):
        n = rng.integers(1, 6)
        ranks = rng.integers(2, 15, size=n); suits = rng.integers(0, 4, size=n)
        pr, ps, pm = _pad5(ranks, suits)
        ht0, c0, m0, sc0 = score_core(pr, ps, pm, levels)
        ht1, c1, m1, sc1 = score_with_jokers(
            pr, ps, pm, h_r, h_s, h_m, levels, jk,
            money=jnp.int32(0), discards_left=jnp.int32(0), deck_count=jnp.int32(0),
            hand_plays_run=jnp.zeros(12, jnp.int32), hand_plays_round=jnp.zeros(12, jnp.int32))
        assert (int(ht0), int(c0), int(sc0)) == (int(ht1), int(c1), int(sc1))
        assert int(sc1) == int(jnp.floor(c1.astype(jnp.float32) * m1))


def test_score_with_jokers_jit_vmap_batches():
    import jax
    jf = jax.jit(lambda *a, **k: score_with_jokers(*a, **k))
    # build a tiny batch of 3 plain hands, empty loadout
    pr = jnp.array([[14,14,5,0,0],[2,3,4,5,6],[10,10,10,2,2]], jnp.int32)
    ps = jnp.array([[0,1,2,0,0],[0,0,0,0,0],[0,1,2,3,0]], jnp.int32)
    pm = jnp.array([[1,1,1,0,0],[1,1,1,1,1],[1,1,1,1,1]], bool)
    hr = jnp.zeros((3,8), jnp.int32); hs = jnp.zeros((3,8), jnp.int32); hm = jnp.zeros((3,8), bool)
    lv = jnp.ones((3,12), jnp.int32)
    from balatro_rl.envs.actions import MAX_JOKERS
    jk = jnp.zeros((3, MAX_JOKERS), jnp.int32)
    z = jnp.zeros(3, jnp.int32); z12 = jnp.zeros((3,12), jnp.int32)
    out = jax.vmap(lambda *a: score_with_jokers(
        a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
        money=a[8],discards_left=a[9],deck_count=a[10],
        hand_plays_run=a[11],hand_plays_round=a[12]))(
        pr,ps,pm,hr,hs,hm,lv,jk,z,z,z,z12,z12)
    assert out[3].shape == (3,)  # score per env

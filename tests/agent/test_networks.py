"""Tests for the card-aware ActorCritic (per-card SAB encoder + candidate-scoring
play/discard head + shop head). Covers the spec test plan #1-6: static tensors,
contract/shape, logit-order round-trip, card-awareness, masked-pool safety, and
jit-stability. The load-bearing invariant is the logit-assembly ORDER (silent if
transposed) — pinned by the round-trip test.
"""
import jax, jax.numpy as jnp
import numpy as np
import pytest

from balatro_rl.agent import networks
from balatro_rl.agent.networks import ActorCritic, SUBSET_IDX, SUBSET_CNT, PAIR_I, PAIR_J
from balatro_rl.agent.spec import dummy_obs
from balatro_rl.agent.value_head import NBINS
from balatro_rl.envs import actions
from balatro_rl.envs.actions import NUM_ACTIONS, _SUBSETS, _PAIRS
from balatro_rl.envs.obs import encode
from balatro_rl.engine.engine import reset


def _init(B=4, d=32, **kw):
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d, **kw)
    obs = {k: jnp.asarray(v) for k, v in dummy_obs(B).items()}
    mask = jnp.ones((B, NUM_ACTIONS), dtype=bool)
    params = net.init(jax.random.PRNGKey(0), obs, mask)
    return net, params, obs, mask


# ---------------------------------------------------------------- 1. static tensors
def test_static_subset_idx():
    assert SUBSET_IDX.shape == (218, 5)
    assert SUBSET_IDX.dtype == jnp.int32
    assert int(SUBSET_IDX.max()) == 7          # MAX_HAND-1; in-range for an [8,d] gather


def test_static_subset_cnt_matches_row_lengths():
    assert SUBSET_CNT.shape == (218, 5)
    counts = np.asarray(SUBSET_CNT.sum(1)).astype(int)
    lengths = np.asarray([len(c) for c in _SUBSETS])
    assert np.array_equal(counts, lengths)     # count of real members == subset cardinality
    # and the real members sit in the leading slots, pad slots index 0 / count 0
    idx = np.asarray(SUBSET_IDX)
    for i, c in enumerate(_SUBSETS):
        assert tuple(idx[i, : len(c)].tolist()) == tuple(c)


def test_static_pair_columns():
    assert PAIR_I.shape == (30,) and PAIR_J.shape == (30,)   # MAX_JOKERS=6 -> 6*5 ordered pairs
    assert PAIR_I.dtype == jnp.int32 and PAIR_J.dtype == jnp.int32
    assert np.array_equal(np.asarray(PAIR_I), [p[0] for p in _PAIRS])
    assert np.array_equal(np.asarray(PAIR_J), [p[1] for p in _PAIRS])


# ---------------------------------------------------------------- 2. contract / shape
def test_init_contract():
    net, params, obs, mask = _init(B=1)
    logits, value_logits = net.apply(params, obs, mask)
    assert logits.shape == (1, NUM_ACTIONS)
    assert value_logits.shape == (1, NBINS)


@pytest.mark.parametrize("B", [1, 64])
def test_apply_shapes_and_dtypes(B):
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=32)
    obs = {k: jnp.asarray(v) for k, v in dummy_obs(B).items()}
    mask = jnp.ones((B, NUM_ACTIONS), dtype=bool)
    params = net.init(jax.random.PRNGKey(0), obs, mask)
    logits, value_logits = net.apply(params, obs, mask)
    assert logits.shape == (B, NUM_ACTIONS)
    assert value_logits.shape == (B, NBINS)
    assert logits.dtype == jnp.float32 and value_logits.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(value_logits)))


# ---------------------------------------------------------------- 3. logit-order round-trip
def test_logit_order_roundtrip_real_state():
    """For a real seeded engine state + its legal mask: every illegal index is
    finfo.min and the set of non-min indices is exactly the legal True set."""
    state = reset(seed=7)
    obs = {k: jnp.asarray(v)[None] for k, v in encode(state).items()}
    legal = actions.legal_mask(state)
    assert legal.any() and not legal.all()              # a real mid-game mask, partly legal
    mask = jnp.asarray(legal)[None]

    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=32)
    params = net.init(jax.random.PRNGKey(0), obs, mask)
    logits, _ = net.apply(params, obs, mask)
    logits = np.asarray(logits[0])

    floor = np.finfo(np.float32).min
    is_min = logits == floor
    assert np.array_equal(~is_min, legal)               # non-min set == legal True set, index-for-index


def test_shop_offsets_map_correctly():
    """A flipped-on mask at one shop offset survives where the env would place that
    verb — pins the play/disc/shop assembly order against actions.py offsets."""
    net, params, obs, _ = _init(B=1)
    floor = np.finfo(np.float32).min
    # explicit offsets from actions.py (MAX_JOKERS=6: buy2 sell6 reroll1 reorder30 leave1)
    assert (actions._BUY0, actions._SELL0, actions._REROLL, actions._REORDER0, actions._LEAVE) \
        == (436, 438, 444, 445, 475)
    for off in [actions._BUY0, actions._BUY0 + 1, actions._SELL0, actions._SELL0 + 5,
                actions._REROLL, actions._REORDER0, actions._REORDER0 + 29, actions._LEAVE,
                actions._USE_TARGET0, actions._OPEN0, actions._PICK0, actions._SKIP_PACK,
                actions._BUY_VOUCHER]:
        mask = np.zeros((1, NUM_ACTIONS), dtype=bool)
        mask[0, off] = True
        logits, _ = net.apply(params, obs, jnp.asarray(mask))
        logits = np.asarray(logits[0])
        assert logits[off] != floor                     # the one legal index survives
        assert (np.delete(logits, off) == floor).all()  # everything else floored


# ---------------------------------------------------------------- 4. card-awareness
def _perturbed_obs(B=1, slot=0, key=0):
    """A real-card obs with one card in `slot` perturbed to a different card."""
    obs = {k: np.asarray(v) for k, v in dummy_obs(B).items()}
    obs["hand_mask"][:] = 1.0                            # all 8 slots present
    rng = np.random.default_rng(key)
    for b in range(B):
        for s in range(8):
            obs["hand"][b, s] = 0.0
            obs["hand"][b, s, rng.integers(0, 13)] = 1.0  # a rank
            obs["hand"][b, s, 13 + rng.integers(0, 4)] = 1.0  # a suit
    return obs


def test_card_awareness_candidate_head_isolated():
    """Candidate-head invariant (n_layers=0 so the per-card tokens DON'T cross-mix):
    perturb ONE card slot -> ONLY play/discard logits whose subset CONTAINS that slot
    change; subsets that omit it are byte-identical. This is the load-bearing proof
    that a play-logit is a function of *its selected cards* (vs. one pooled state).
    (With n_layers>=1 the SAB encoder cross-mixes by design — covered separately below.)"""
    base = _perturbed_obs(B=1, key=1)
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=32, n_layers=0)
    mask = jnp.ones((1, NUM_ACTIONS), bool)
    params = net.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in base.items()}, mask)

    target = 3                                           # perturb slot 3
    pert = {k: v.copy() for k, v in base.items()}
    pert["hand"][0, target] = 0.0
    pert["hand"][0, target, 11] = 1.0                    # a different rank
    pert["hand"][0, target, 13 + 2] = 1.0               # a different suit

    l0 = np.asarray(net.apply(params, {k: jnp.asarray(v) for k, v in base.items()}, mask)[0][0])
    l1 = np.asarray(net.apply(params, {k: jnp.asarray(v) for k, v in pert.items()}, mask)[0][0])

    changed = ~np.isclose(l0, l1, atol=1e-6)
    PLAY_N = len(_SUBSETS)
    for i, sub in enumerate(_SUBSETS):
        if target not in sub:                            # subset omits the perturbed slot
            assert not changed[i], f"play subset {i} changed but omits slot {target}"
            assert not changed[PLAY_N + i], f"disc subset {i} changed but omits slot {target}"
    # AND some containing subset actually moved (the perturbation mattered, and it is
    # card-aware rather than ignoring the hand entirely).
    moved = sum(int(changed[i]) for i, sub in enumerate(_SUBSETS) if target in sub)
    assert moved > 0, "perturbing a card moved no containing-subset logit"


def test_card_awareness_full_net_is_card_dependent():
    """Full net (n_layers=2): a play-logit is a genuine function of the cards — perturbing
    a card moves play logits, and DIFFERENT subsets get DIFFERENT deltas (a flat softmax
    over one pooled vector could not produce subset-specific responses)."""
    base = _perturbed_obs(B=1, key=5)
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=32)   # default n_layers=2
    mask = jnp.ones((1, NUM_ACTIONS), bool)
    params = net.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in base.items()}, mask)

    pert = {k: v.copy() for k, v in base.items()}
    pert["hand"][0, 2] = 0.0
    pert["hand"][0, 2, 8] = 1.0
    pert["hand"][0, 2, 13 + 1] = 1.0

    l0 = np.asarray(net.apply(params, {k: jnp.asarray(v) for k, v in base.items()}, mask)[0][0])
    l1 = np.asarray(net.apply(params, {k: jnp.asarray(v) for k, v in pert.items()}, mask)[0][0])

    play_delta = (l1 - l0)[: len(_SUBSETS)]
    assert np.abs(play_delta).max() > 1e-4               # the hand genuinely drives play logits
    # subset-specific (not a single shared shift): a flat-pool head would move all 218 by the
    # same amount up to fixed head weights -> deltas span a real range, not one constant.
    assert play_delta.std() > 1e-5


# ---------------------------------------------------------------- 5. masked-pool safety
def test_masked_pool_ignores_padded_slots():
    """With some hand_mask==0 slots, flipping a padded slot's raw obs changes NO
    surviving legal logit (the SUBSET_CNT*hand_mask gate zeroes absent-slot junk)."""
    base = _perturbed_obs(B=1, key=2)
    base["hand_mask"][0, 5:] = 0.0                       # slots 5,6,7 absent
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=32)
    # legal mask: only subsets entirely within present slots 0..4 (mirror the engine)
    present = set(range(5))
    legal = np.zeros((1, NUM_ACTIONS), bool)
    PLAY_N = len(_SUBSETS)
    for i, sub in enumerate(_SUBSETS):
        if set(sub) <= present:
            legal[0, i] = True
            legal[0, PLAY_N + i] = True
    mask = jnp.asarray(legal)
    params = net.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in base.items()}, mask)

    pert = {k: v.copy() for k, v in base.items()}
    pert["hand"][0, 6] = 0.0                             # scramble a PADDED slot's raw features
    pert["hand"][0, 6, 4] = 1.0
    pert["hand"][0, 6, 15] = 1.0

    l0 = np.asarray(net.apply(params, {k: jnp.asarray(v) for k, v in base.items()}, mask)[0][0])
    l1 = np.asarray(net.apply(params, {k: jnp.asarray(v) for k, v in pert.items()}, mask)[0][0])

    legal_idx = np.where(legal[0])[0]
    assert np.allclose(l0[legal_idx], l1[legal_idx], atol=1e-6), \
        "flipping a padded slot moved a surviving legal logit"


# ---------------------------------------------------------------- 6. jit-stability
def test_jit_stability_two_batch_sizes_no_retrace():
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=32)
    obs1 = {k: jnp.asarray(v) for k, v in dummy_obs(1).items()}
    params = net.init(jax.random.PRNGKey(0), obs1, jnp.ones((1, NUM_ACTIONS), bool))

    n_trace = [0]

    @jax.jit
    def fwd(p, o, m):
        n_trace[0] += 1                                  # increments once per TRACE, not per call
        return net.apply(p, o, m)

    for B in (1, 64):
        o = {k: jnp.asarray(v) for k, v in dummy_obs(B).items()}
        m = jnp.ones((B, NUM_ACTIONS), bool)
        fwd(p=params, o=o, m=m)
        fwd(p=params, o=o, m=m)                          # repeat at same B -> no new trace
    assert n_trace[0] == 2, f"expected exactly 2 traces (B=1, B=64), got {n_trace[0]}"

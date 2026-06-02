import numpy as np
from balatro_rl.engine.engine import reset
from balatro_rl.envs.obs import encode, OBS_SHAPES, symlog


def test_symlog_monotone_and_signed():
    assert symlog(0.0) == 0.0
    assert symlog(99999) > symlog(10)
    assert symlog(-5) == -symlog(5)


def test_encode_returns_declared_shapes_and_dtypes():
    obs = encode(reset(seed=1))
    assert set(obs.keys()) == set(OBS_SHAPES.keys())
    for k, shape in OBS_SHAPES.items():
        assert obs[k].shape == shape, k
        assert obs[k].dtype in (np.float32, np.int32), k
    assert np.isfinite(obs["global"]).all()


def test_hand_mask_counts_real_cards():
    obs = encode(reset(seed=1))
    assert obs["hand_mask"].sum() == 8        # fresh hand is 8 cards
    # rank one-hot present on the first card (ranks 2..14 -> index rank-2)
    assert obs["hand"][0, :13].sum() == 1
    assert obs["hand"][0, 13:17].sum() == 1   # one suit set


def test_empty_jokers_and_shop_masked():
    obs = encode(reset(seed=1))
    assert obs["joker_mask"].sum() == 0
    assert obs["shop_mask"].sum() == 0
    assert obs["levels"].shape == (12,)


def test_deck_histogram_sums_to_remaining_deck():
    s = reset(seed=1)
    obs = encode(s)
    assert int(obs["deck_rank_hist"].sum()) == len(s.deck)
    assert int(obs["deck_suit_hist"].sum()) == len(s.deck)

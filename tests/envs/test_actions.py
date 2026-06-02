import numpy as np
from balatro_rl.engine.engine import Verb, reset, legal_actions, step
from balatro_rl.engine.state import Phase
from balatro_rl.envs.actions import NUM_ACTIONS, decode, legal_mask, PLAY_N, SHOP_BASE


def test_action_space_size():
    assert PLAY_N == 218            # C(8,1..5)
    assert NUM_ACTIONS == 465       # 218 play + 218 discard + 29 shop


def test_decode_play_and_discard():
    assert decode(0) == (Verb.PLAY, (0,))
    assert decode(PLAY_N) == (Verb.DISCARD, (0,))         # first discard id
    v, arg = decode(PLAY_N - 1)                            # last play subset
    assert v == Verb.PLAY and len(arg) == 5


def test_decode_shop_actions():
    assert decode(SHOP_BASE) == (Verb.BUY, 0)
    assert decode(SHOP_BASE + 1) == (Verb.BUY, 1)
    assert decode(SHOP_BASE + 2) == (Verb.SELL, 0)
    assert decode(SHOP_BASE + 7) == (Verb.REROLL, 0)
    assert decode(NUM_ACTIONS - 1) == (Verb.LEAVE_SHOP, 0)
    v, arg = decode(SHOP_BASE + 8)                          # first reorder
    assert v == Verb.REORDER and isinstance(arg, tuple) and arg[0] != arg[1]


def test_legal_mask_matches_engine_in_playing():
    s = reset(seed=1)
    mask = legal_mask(s)
    assert mask.dtype == np.bool_ and mask.shape == (NUM_ACTIONS,)
    # Every engine-legal action maps to a True mask bit, and counts match.
    eng = legal_actions(s)
    assert mask.sum() == len(eng)
    # Each legal action decodes back to an engine action in the legal set.
    legal_ids = np.flatnonzero(mask)
    decoded = {decode(int(i)) for i in legal_ids}
    assert decoded == set(eng)


def test_legal_mask_in_shop_only_shop_actions():
    import dataclasses
    from balatro_rl.engine.cards import Card
    s = reset(seed=1)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3),
            Card(2, 0), Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, required=10, money=100)
    s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))   # -> SHOP
    assert s.phase == Phase.SHOP
    mask = legal_mask(s)
    # No play/discard bits set in the shop.
    assert mask[:2 * PLAY_N].sum() == 0
    assert mask[NUM_ACTIONS - 1]                 # LEAVE_SHOP always legal

"""Phase D2: obs/action widening. The agent now SEES enhancements/editions/seals,
debuffs, the active boss, level state, and owned consumables, and can USE them.

Parity check: on a plain game (no mods, no boss, no consumables) every NEW feature is 0
and every EXISTING feature is unchanged -> the widened obs reproduces the pre-D2 game
(the engine is untouched, so game dynamics are byte-identical regardless).
"""
import dataclasses

import numpy as np

from balatro_rl.engine.bosses import BossEffect
from balatro_rl.engine.cards import Card, Edition, Enhancement, Seal
from balatro_rl.engine.consumables import PlanetType, planet
from balatro_rl.engine.engine import reset, Verb
from balatro_rl.engine.hands import HandType
from balatro_rl.envs.actions import encode_action, legal_mask
from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.obs import (
    CARD_FEAT, OBS_SHAPES, encode, consum_vocab_id, _ENH0, _EDITION0, _SEAL0,
    _DEBUFF_I, _FACEDOWN_I, N_BOSS,
)
from balatro_rl.agent.spec import dummy_obs


def C(rank, suit=0, **mods):
    return Card(rank=rank, suit=suit, **mods)


# ============================================================================
# shapes
# ============================================================================

def test_dummy_obs_matches_shapes_and_has_new_fields():
    obs = dummy_obs(3)
    for k, shape in OBS_SHAPES.items():
        assert obs[k].shape == (3,) + shape, k
    for k in ("boss_onehot", "consum_types", "consum_mask"):
        assert k in obs


def test_encode_matches_obs_shapes():
    o = encode(reset(seed=0))
    for k, shape in OBS_SHAPES.items():
        assert o[k].shape == shape, k


# ============================================================================
# plain-game parity: new features 0, existing features preserved
# ============================================================================

def test_plain_card_vec_keeps_rank_suit_and_zeroes_new_feats():
    o = encode(reset(seed=0))
    card0 = reset(seed=0).hand[0]
    v = o["hand"][0]
    assert v[card0.rank - 2] == 1.0 and v[13 + card0.suit] == 1.0     # rank/suit unchanged
    assert v[_ENH0 + Enhancement.NONE] == 1.0                         # NONE enhancement one-hot
    assert v[_EDITION0 + Edition.NONE] == 1.0 and v[_SEAL0 + Seal.NONE] == 1.0
    assert v[_DEBUFF_I] == 0.0 and v[_FACEDOWN_I] == 0.0


def test_plain_game_boss_and_consumables_are_empty():
    o = encode(reset(seed=0))
    assert o["boss_onehot"][int(BossEffect.NONE)] == 1.0 and o["boss_onehot"].sum() == 1.0
    assert o["consum_mask"].sum() == 0.0
    assert o["global"][16] == 0.0 and o["global"][18] == 0.0          # no consumables, no boss


# ============================================================================
# enhancement / edition / seal one-hots
# ============================================================================

def test_card_mods_encode_into_one_hots():
    st = dataclasses.replace(
        reset(seed=0),
        hand=(C(7, 0, enhancement=Enhancement.GLASS, edition=Edition.FOIL, seal=Seal.GOLD),))
    v = encode(st)["hand"][0]
    assert v[_ENH0 + Enhancement.GLASS] == 1.0
    assert v[_EDITION0 + Edition.FOIL] == 1.0
    assert v[_SEAL0 + Seal.GOLD] == 1.0


# ============================================================================
# boss: one-hot + per-card is_debuffed
# ============================================================================

def test_boss_onehot_and_card_debuff_flag():
    st = dataclasses.replace(
        reset(seed=0), boss=int(BossEffect.THE_CLUB),
        hand=(C(7, 2), C(7, 1)))                                      # club, heart (0=S,1=H,2=C,3=D)
    o = encode(st)
    assert o["boss_onehot"][int(BossEffect.THE_CLUB)] == 1.0
    assert o["global"][18] == 1.0                                     # boss-active flag
    assert o["hand"][0][_DEBUFF_I] == 1.0                            # club debuffed
    assert o["hand"][1][_DEBUFF_I] == 0.0                            # heart not


def test_boss_onehot_size():
    assert OBS_SHAPES["boss_onehot"] == (N_BOSS,) and N_BOSS == len(BossEffect)


# ============================================================================
# consumables: stream + global counts
# ============================================================================

def test_consumable_stream_encodes_owned():
    st = dataclasses.replace(reset(seed=0), consumables=(planet(PlanetType.MERCURY),))
    o = encode(st)
    assert o["consum_types"][0] == consum_vocab_id(planet(PlanetType.MERCURY))
    assert o["consum_mask"][0] == 1.0 and o["consum_mask"][1] == 0.0
    assert o["global"][16] == 1.0 and o["global"][17] == 2.0          # #consumables, slots


# ============================================================================
# USE action round-trips through the env
# ============================================================================

def test_use_action_is_legal_and_applies_through_env():
    env = BalatroEnv()
    env.reset(0)
    env.state = dataclasses.replace(env.state, consumables=(planet(PlanetType.MERCURY),))
    use_id = encode_action(Verb.USE, 0)
    assert legal_mask(env.state)[use_id]
    _obs, _r, _done, info, _m = env.step(use_id)
    assert info["verb"] == "use"
    assert env.state.levels[int(HandType.PAIR)] == 2 and env.state.consumables == ()

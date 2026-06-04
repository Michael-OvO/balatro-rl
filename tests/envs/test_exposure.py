"""Phase D exposure: the training env can seed enhancements + grant Planet consumables so
the retrain actually encounters the new content. Default off -> byte-identical plain game.
"""
import numpy as np

from balatro_rl.engine.cards import Enhancement
from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.exposure import make_exposure
from balatro_rl.envs.obs import _ENH0
from balatro_rl.envs.actions import encode_action
from balatro_rl.engine.engine import Verb


# ============================================================================
# make_exposure
# ============================================================================

def test_default_is_noop():
    assert make_exposure(0, 0.0, 0) == (None, ())


def test_enhance_rate_1_mods_all_cards_deterministically():
    mods_a, _ = make_exposure(3, enhance_rate=1.0)
    mods_b, _ = make_exposure(3, enhance_rate=1.0)
    assert mods_a == mods_b                       # deterministic per seed
    assert len(mods_a) == 52 and all("enhancement" in m for m in mods_a.values())
    assert all(1 <= m["enhancement"] <= 8 for m in mods_a.values())   # a real enhancement


def test_enhance_rate_is_roughly_proportional():
    mods, _ = make_exposure(7, enhance_rate=0.25)
    assert 4 <= len(mods) <= 30                    # ~13 of 52, with slack


def test_grant_planets_capped_at_slots():
    _, cons = make_exposure(1, grant_planets=5)
    assert len(cons) == 2                          # capped at the 2 consumable slots


def test_different_seeds_differ():
    a, _ = make_exposure(1, enhance_rate=0.3)
    b, _ = make_exposure(2, enhance_rate=0.3)
    assert a != b


# ============================================================================
# BalatroEnv wiring
# ============================================================================

def test_env_default_is_plain_byte_compatible():
    env = BalatroEnv()
    env.reset(0)
    assert all(c.enhancement == 0 for c in env.state.master_deck)
    assert env.state.consumables == ()


def test_env_enhance_rate_seeds_enhanced_cards_visible_in_obs():
    env = BalatroEnv(enhance_rate=1.0)
    obs, _mask = env.reset(0)
    assert any(c.enhancement != 0 for c in env.state.master_deck)
    # the agent SEES it: at least one hand card has a non-NONE enhancement one-hot bit set
    enh_bits = obs["hand"][:, _ENH0 + 1:_ENH0 + 9]    # enhancement bits past NONE
    assert enh_bits.sum() > 0


def test_env_grant_planets_makes_use_legal():
    env = BalatroEnv(grant_planets=1)
    _obs, mask = env.reset(0)
    assert len(env.state.consumables) == 1
    assert mask[encode_action(Verb.USE, 0)]            # the agent can USE it


def test_train_smoke_with_exposure_runs():
    from balatro_rl.agent.train import train, TrainConfig
    from balatro_rl.agent.metrics_logger import NullLogger
    # a tiny run with exposure on -> the wider obs/USE path is exercised end-to-end
    train(TrainConfig(num_updates=2, num_envs=8, num_steps=32, d_model=16,
                      num_minibatches=2, update_epochs=1,
                      enhance_rate=0.3, grant_planets=1, enable_bosses=True),
          logger=NullLogger())

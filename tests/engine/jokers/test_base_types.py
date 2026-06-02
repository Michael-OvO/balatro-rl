from balatro_rl.engine.jokers.base import (
    Effect, NO_EFFECT, RuleFlags, NO_RULES, JokerType, JokerState,
)


def test_effect_defaults_are_identity():
    assert (NO_EFFECT.chips, NO_EFFECT.mult, NO_EFFECT.xmult) == (0, 0.0, 1.0)


def test_effect_is_frozen():
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        Effect().chips = 5


def test_rule_flags_default_off():
    assert NO_RULES == RuleFlags(splash=False, all_face=False)


def test_joker_type_has_proof_set():
    names = {jt.name for jt in JokerType}
    assert {"JOKER", "CAVENDISH", "GREEDY", "SCARY_FACE", "PHOTOGRAPH", "BARON",
            "HACK", "RIDE_THE_BUS", "SPLASH", "PAREIDOLIA", "BLUEPRINT"} <= names


def test_joker_state_defaults():
    js = JokerState(type=JokerType.JOKER)
    assert js.edition == 0 and js.counter == 0.0

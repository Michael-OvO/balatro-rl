import pytest

from balatro_rl.engine.jokers.base import (
    JokerEffect, JokerType, JokerState, Effect, RuleFlags,
    register, REGISTRY, aggregate_rules, resolve_providers, ScoreContext,
)


# Minimal stubs so the registry/resolve tests are self-contained (the Plan-2
# joker library is not imported here). Registered per-test via an autouse
# fixture (NOT at module/collection time) so they don't contaminate the
# REGISTRY snapshot baseline of other test modules; the conftest fixture then
# restores REGISTRY after each test.
class _StubJoker(JokerEffect):
    def independent(self, ctx, js):
        return Effect(mult=4)


class _StubBlueprint(JokerEffect):
    pass


@pytest.fixture(autouse=True)
def _register_stubs():
    REGISTRY[JokerType.JOKER] = _StubJoker()
    REGISTRY[JokerType.BLUEPRINT] = _StubBlueprint()
    yield


def test_default_hooks_are_noops():
    eff = JokerEffect()
    js = JokerState(type=JokerType.JOKER)
    assert eff.independent(None, js) == Effect()
    assert eff.on_score(None, None, 0, js) == Effect()
    assert eff.on_held(None, None, js) == Effect()
    assert eff.retrigger(None, None, js) == 0
    assert eff.rules() == RuleFlags()
    assert eff.copyable is True


def test_register_populates_registry():
    @register(JokerType.JOKER)
    class _J(JokerEffect):
        def independent(self, ctx, js):
            return Effect(mult=4)
    assert isinstance(REGISTRY[JokerType.JOKER], _J)
    assert REGISTRY[JokerType.JOKER].independent(None, JokerState(JokerType.JOKER)).mult == 4


def test_aggregate_rules_ors_flags():
    class _Splash(JokerEffect):
        def rules(self):
            return RuleFlags(splash=True)
    REGISTRY[JokerType.SPLASH] = _Splash()
    flags = aggregate_rules((JokerState(JokerType.SPLASH), JokerState(JokerType.JOKER)))
    assert flags.splash is True and flags.all_face is False


def test_resolve_providers_passes_through_non_copy():
    provs = resolve_providers((JokerState(JokerType.JOKER),))
    assert len(provs) == 1
    eff, js = provs[0]
    assert js.type == JokerType.JOKER


def test_blueprint_resolves_to_right_neighbor():
    # Blueprint (slot 0) left of Joker (slot 1) -> copies Joker's effect.
    class _BP(JokerEffect):
        pass
    REGISTRY[JokerType.BLUEPRINT] = _BP()
    jokers = (JokerState(JokerType.BLUEPRINT), JokerState(JokerType.JOKER))
    provs = resolve_providers(jokers)
    # Two providers: Blueprint-as-Joker, and Joker itself.
    assert provs[0][0] is REGISTRY[JokerType.JOKER]   # blueprint copies Joker's effect
    assert provs[1][0] is REGISTRY[JokerType.JOKER]


def test_blueprint_at_rightmost_contributes_nothing():
    jokers = (JokerState(JokerType.JOKER), JokerState(JokerType.BLUEPRINT))
    provs = resolve_providers(jokers)
    assert len(provs) == 1  # blueprint has no right neighbor -> dropped
    assert provs[0][1].type == JokerType.JOKER


def test_score_context_is_mutable():
    ctx = ScoreContext(chips=10, mult=2.0)
    ctx.chips += 5
    ctx.mult *= 3
    assert ctx.chips == 15 and ctx.mult == 6.0

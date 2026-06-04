"""The wiki-art resolver is cache-agnostic: with no local cache every lookup returns None
(viewer falls back to CSS); with a cache it returns base64 data URIs. These tests assert the
CONTRACT (None or a data: URI) so they pass with or without the gitignored cache present."""
from balatro_rl.viz import assets
from balatro_rl.engine.consumables import ConsumableKind, PlanetType


def _ok(v):
    return v is None or (isinstance(v, str) and v.startswith("data:image/png;base64,"))


def test_available_is_bool():
    assert isinstance(assets.available(), bool)


def test_all_resolvers_return_none_or_data_uri():
    assert _ok(assets.card(13, 1))
    assert _ok(assets.joker(1))
    assert _ok(assets.enhancement(4))
    assert _ok(assets.seal(1))
    assert _ok(assets.edition(1))
    assert _ok(assets.boss(7))
    assert _ok(assets.planet(int(PlanetType.MERCURY)))


def test_zero_ids_resolve_to_none():
    # NONE enhancement/seal/edition/boss are 0 -> no image.
    assert assets.enhancement(0) is None and assets.seal(0) is None
    assert assets.edition(0) is None and assets.boss(0) is None


def test_consumable_dispatches_on_kind():
    assert _ok(assets.consumable(int(ConsumableKind.PLANET), int(PlanetType.MERCURY)))
    assert assets.consumable(int(ConsumableKind.TAROT), 1) is None     # tarot art not mapped


def test_resolvers_do_not_raise_on_unknown_ids():
    for v in (assets.card(99, 9), assets.joker(99999), assets.boss(999), assets.planet(999)):
        assert _ok(v)

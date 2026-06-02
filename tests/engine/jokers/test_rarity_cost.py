from balatro_rl.engine.jokers.base import Rarity, JokerType, JokerState, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401


def test_rarity_enum():
    assert [r.name for r in Rarity] == ["COMMON", "UNCOMMON", "RARE", "LEGENDARY"]


def test_joker_state_has_sell_bonus_default():
    assert JokerState(type=JokerType.JOKER).sell_bonus == 0


def test_existing_jokers_declare_rarity_and_cost():
    # wiki: docs/reference/jokers.md
    expected = {
        JokerType.JOKER: (Rarity.COMMON, 2),
        JokerType.GREEDY: (Rarity.COMMON, 5),
        JokerType.SCARY_FACE: (Rarity.COMMON, 4),
        JokerType.PHOTOGRAPH: (Rarity.COMMON, 5),
        JokerType.CAVENDISH: (Rarity.COMMON, 4),
        JokerType.SPLASH: (Rarity.COMMON, 3),
        JokerType.RIDE_THE_BUS: (Rarity.COMMON, 6),
        JokerType.HACK: (Rarity.UNCOMMON, 6),
        JokerType.PAREIDOLIA: (Rarity.UNCOMMON, 5),
        JokerType.BARON: (Rarity.RARE, 8),
        JokerType.BLUEPRINT: (Rarity.RARE, 10),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt

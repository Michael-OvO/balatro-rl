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


def test_batch4_jokers_declare_rarity_and_cost():
    # wiki: /w/Faceless_Joker /w/Green_Joker /w/Ramen
    expected = {
        JokerType.FACELESS_JOKER: (Rarity.COMMON, 4),
        JokerType.GREEN_JOKER: (Rarity.COMMON, 4),
        JokerType.RAMEN: (Rarity.UNCOMMON, 6),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt


def test_batch5_jokers_declare_rarity_and_cost():
    # wiki: /w/Misprint /w/Bloodstone /w/Ancient_Joker /w/The_Idol /w/Mail-In_Rebate
    expected = {
        JokerType.MISPRINT: (Rarity.COMMON, 4),
        JokerType.BLOODSTONE: (Rarity.UNCOMMON, 7),
        JokerType.ANCIENT_JOKER: (Rarity.RARE, 8),
        JokerType.THE_IDOL: (Rarity.UNCOMMON, 6),
        JokerType.MAIL_IN_REBATE: (Rarity.COMMON, 4),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt


def test_batch6_jokers_declare_rarity_and_cost():
    # wiki: /w/Supernova /w/Card_Sharp /w/Obelisk
    expected = {
        JokerType.SUPERNOVA: (Rarity.COMMON, 5),
        JokerType.CARD_SHARP: (Rarity.UNCOMMON, 6),
        JokerType.OBELISK: (Rarity.RARE, 8),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt


def test_batch1_jokers_declare_rarity_and_cost():
    # wiki: docs/reference/jokers.md
    expected = {
        JokerType.LUSTY: (Rarity.COMMON, 5),
        JokerType.WRATHFUL: (Rarity.COMMON, 5),
        JokerType.GLUTTONOUS: (Rarity.COMMON, 5),
        JokerType.JOLLY: (Rarity.COMMON, 3),
        JokerType.ZANY: (Rarity.COMMON, 4),
        JokerType.MAD: (Rarity.COMMON, 4),
        JokerType.CRAZY: (Rarity.COMMON, 4),
        JokerType.DROLL: (Rarity.COMMON, 4),
        JokerType.SLY: (Rarity.COMMON, 3),
        JokerType.WILY: (Rarity.COMMON, 4),
        JokerType.CLEVER: (Rarity.COMMON, 4),
        JokerType.DEVIOUS: (Rarity.COMMON, 4),
        JokerType.CRAFTY: (Rarity.COMMON, 4),
        JokerType.HALF: (Rarity.COMMON, 5),
        JokerType.FIBONACCI: (Rarity.UNCOMMON, 8),
        JokerType.GROS_MICHEL: (Rarity.COMMON, 5),
        JokerType.EVEN_STEVEN: (Rarity.COMMON, 4),
        JokerType.ODD_TODD: (Rarity.COMMON, 4),
        JokerType.SCHOLAR: (Rarity.COMMON, 4),
        JokerType.RUNNER: (Rarity.COMMON, 5),
        JokerType.ICE_CREAM: (Rarity.COMMON, 5),
        JokerType.WALKIE_TALKIE: (Rarity.COMMON, 4),
        JokerType.SMILEY_FACE: (Rarity.COMMON, 4),
        JokerType.SOCK_AND_BUSKIN: (Rarity.UNCOMMON, 6),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt

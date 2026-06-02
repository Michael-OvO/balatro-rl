import pytest

from balatro_rl.engine.blinds import required_score, ANTE_BASE, BLIND_MULT


def test_ante1_blinds():
    assert required_score(1, 0) == 300   # small  = 1.0x
    assert required_score(1, 1) == 450   # big    = 1.5x
    assert required_score(1, 2) == 600   # boss   = 2.0x


def test_ante8_boss():
    assert required_score(8, 2) == 100_000  # 50_000 * 2.0


def test_base_table_matches_spec():
    assert ANTE_BASE == {1: 300, 2: 800, 3: 2000, 4: 5000,
                         5: 11000, 6: 20000, 7: 35000, 8: 50000}
    assert BLIND_MULT == (1.0, 1.5, 2.0)


def test_invalid_blind_index_raises():
    with pytest.raises(IndexError):
        required_score(1, 3)

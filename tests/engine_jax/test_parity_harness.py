"""Test the parity harness field extractors and deck reconstruction.

Validates python_core_fields and deck_from_python against a seeded Python
engine reset. No JAX engine yet — this is the oracle side only.
"""
from balatro_rl.engine import engine
from tests.engine_jax.parity_util import python_core_fields, deck_from_python


def test_python_extractors():
    gs = engine.reset(0, 0.2, None, False)
    f = python_core_fields(gs)
    assert f["ante"] == 1 and f["hands_left"] == 4 and f["discards_left"] == 3
    assert len(f["hand"]) == 8
    ranks, suits = deck_from_python(gs)
    assert len(ranks) == 52


def test_python_core_fields_types():
    """All extracted values must be plain Python types (int/bool/list/tuple)."""
    gs = engine.reset(0, 0.2, None, False)
    f = python_core_fields(gs)
    assert isinstance(f["ante"], int)
    assert isinstance(f["blind_index"], int)
    assert isinstance(f["round_score"], int)
    assert isinstance(f["required"], int)
    assert isinstance(f["hands_left"], int)
    assert isinstance(f["discards_left"], int)
    assert isinstance(f["hand_size"], int)
    assert isinstance(f["money"], int)
    assert isinstance(f["phase"], int)
    assert isinstance(f["done"], bool)
    assert isinstance(f["won"], bool)
    assert isinstance(f["hand"], list)
    assert isinstance(f["levels"], tuple)


def test_python_core_fields_initial_values():
    """Verify known initial values after reset(seed=0, scale=0.2)."""
    gs = engine.reset(0, 0.2, None, False)
    f = python_core_fields(gs)

    # Initial state invariants (from engine.py constants).
    assert f["ante"] == 1, f"expected ante=1, got {f['ante']}"
    assert f["blind_index"] == 0, f"expected blind_index=0, got {f['blind_index']}"
    assert f["round_score"] == 0, f"expected round_score=0, got {f['round_score']}"
    assert f["hands_left"] == 4, f"expected hands_left=4, got {f['hands_left']}"
    assert f["discards_left"] == 3, f"expected discards_left=3, got {f['discards_left']}"
    assert f["hand_size"] == 8, f"expected hand_size=8, got {f['hand_size']}"
    assert f["money"] == 4, f"expected money=4, got {f['money']}"
    assert f["phase"] == 0, f"expected phase=PLAYING(0), got {f['phase']}"
    assert f["done"] is False
    assert f["won"] is False

    # All 12 hand-type levels start at 1.
    assert len(f["levels"]) == 12
    assert all(lv == 1 for lv in f["levels"]), f"levels not all 1: {f['levels']}"

    # Hand contains 8 cards; each is a (rank, suit) tuple with valid values.
    assert len(f["hand"]) == 8
    for rank, suit in f["hand"]:
        assert 2 <= rank <= 14, f"rank out of range: {rank}"
        assert 0 <= suit <= 3, f"suit out of range: {suit}"


def test_deck_from_python_length_and_coverage():
    """deck_from_python returns exactly 52 cards covering the full shuffled deck."""
    gs = engine.reset(0, 0.2, None, False)
    ranks, suits = deck_from_python(gs)

    assert len(ranks) == 52, f"ranks length {len(ranks)}"
    assert len(suits) == 52, f"suits length {len(suits)}"

    # Every value in valid range.
    for r in ranks:
        assert 2 <= r <= 14, f"rank out of range: {r}"
    for s in suits:
        assert 0 <= s <= 3, f"suit out of range: {s}"

    # A standard deck has each (rank, suit) pair exactly once.
    cards = list(zip(ranks, suits))
    assert len(set(cards)) == 52, "duplicate (rank, suit) pairs in reconstructed deck"


def test_deck_from_python_hand_prefix():
    """draw_order[0:8] must equal the hand cards (in hand-tuple order)."""
    gs = engine.reset(0, 0.2, None, False)
    ranks, suits = deck_from_python(gs)

    for i, card in enumerate(gs.hand):
        assert ranks[i] == int(card.rank), (
            f"Position {i}: rank mismatch — draw_order={ranks[i]}, hand={card.rank}"
        )
        assert suits[i] == int(card.suit), (
            f"Position {i}: suit mismatch — draw_order={suits[i]}, hand={card.suit}"
        )


def test_deck_from_python_deck_suffix():
    """draw_order[8:52] must equal gs.deck in front-to-back order."""
    gs = engine.reset(0, 0.2, None, False)
    ranks, suits = deck_from_python(gs)

    for j, card in enumerate(gs.deck):
        i = 8 + j
        assert ranks[i] == int(card.rank), (
            f"Deck position {j} (draw_order[{i}]): rank mismatch — "
            f"draw_order={ranks[i]}, deck={card.rank}"
        )
        assert suits[i] == int(card.suit), (
            f"Deck position {j} (draw_order[{i}]): suit mismatch — "
            f"draw_order={suits[i]}, deck={card.suit}"
        )


def test_deck_from_python_different_seeds():
    """Different seeds produce different draw orders."""
    ranks0, _ = deck_from_python(engine.reset(0, 0.2, None, False))
    ranks1, _ = deck_from_python(engine.reset(1, 0.2, None, False))
    assert ranks0 != ranks1, "Seeds 0 and 1 produced identical draw orders"

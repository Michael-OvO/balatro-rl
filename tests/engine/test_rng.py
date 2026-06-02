from balatro_rl.engine.rng import RNG


def test_same_seed_same_sequence():
    a = RNG.from_seed(42)
    b = RNG.from_seed(42)
    seq_a, seq_b = [], []
    for _ in range(5):
        x, a = a.randint(0, 100)
        y, b = b.randint(0, 100)
        seq_a.append(x)
        seq_b.append(y)
    assert seq_a == seq_b


def test_different_seeds_differ():
    a = RNG.from_seed(1)
    b = RNG.from_seed(2)
    xa, _ = a.randint(0, 1_000_000)
    xb, _ = b.randint(0, 1_000_000)
    assert xa != xb


def test_random_in_unit_interval():
    rng = RNG.from_seed(7)
    for _ in range(100):
        x, rng = rng.random()
        assert 0.0 <= x < 1.0


def test_randint_inclusive_bounds():
    rng = RNG.from_seed(7)
    seen = set()
    for _ in range(500):
        x, rng = rng.randint(0, 3)
        seen.add(x)
    assert seen == {0, 1, 2, 3}


def test_shuffle_is_deterministic_and_a_permutation():
    items = list(range(10))
    s1, _ = RNG.from_seed(123).shuffle(items)
    s2, _ = RNG.from_seed(123).shuffle(items)
    assert s1 == s2
    assert sorted(s1) == items
    assert s1 != items  # extremely unlikely to be identity for seed 123

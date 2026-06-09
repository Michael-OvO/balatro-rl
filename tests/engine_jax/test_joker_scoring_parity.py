"""Gate A: component parity of score_with_jokers vs engine.scoring.score_play."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

BALATRO_RUN_SLOW = os.environ.get("BALATRO_RUN_SLOW") == "1"

from balatro_rl.engine.cards import Card
from balatro_rl.engine.jokers.base import JokerState
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine_jax.jokers import score_with_jokers
from balatro_rl.engine_jax.scoring import score_core

# Compile the kernel ONCE at module load: every padded shape is static
# ([5]/[8]/[12]/[MAX_JOKERS]), so all harness calls hit the same executable.
_SWJ = jax.jit(score_with_jokers)


def _empty_loadout():
    from balatro_rl.envs.actions import MAX_JOKERS
    return jnp.zeros((MAX_JOKERS,), dtype=jnp.int32)


def _pad5(ranks, suits):
    r = np.zeros(5, np.int8); s = np.zeros(5, np.int8); m = np.zeros(5, bool)
    for i, (rk, su) in enumerate(zip(ranks, suits)):
        r[i] = rk; s[i] = su; m[i] = True
    return jnp.asarray(r), jnp.asarray(s), jnp.asarray(m)


def _oracle(played, jokers, held=(), levels=(), money=0, discards_left=0,
            deck_count=0, hand_plays_run=(), hand_plays_round=()):
    js = tuple(JokerState(type=j) for j in jokers)
    cards = [Card(rank=r, suit=s) for r, s in played]
    heldc = [Card(rank=r, suit=s) for r, s in held]
    return score_play(cards, js, tuple(heldc), joker_slots=5, money=money,
                      hands_left=0, discards_left=discards_left, deck_count=deck_count,
                      hand_plays_run=tuple(hand_plays_run), hand_plays_round=tuple(hand_plays_round),
                      levels=tuple(levels))


def _kernel(played, jokers, held=(), levels=None, money=0, discards_left=0,
            deck_count=0, hand_plays_run=None, hand_plays_round=None):
    from balatro_rl.envs.actions import MAX_JOKERS
    pr, ps, pm = _pad5([r for r, _ in played], [s for _, s in played])
    hr = np.zeros(8, np.int8); hs = np.zeros(8, np.int8); hm = np.zeros(8, bool)
    for i, (r, s) in enumerate(held):
        hr[i] = r; hs[i] = s; hm[i] = True
    jk = np.zeros(MAX_JOKERS, np.int32)
    for i, j in enumerate(jokers):
        jk[i] = j
    lv = jnp.ones(12, jnp.int32) if levels is None else jnp.asarray(levels, jnp.int32)
    hpr = jnp.zeros(12, jnp.int32) if hand_plays_run is None else jnp.asarray(hand_plays_run, jnp.int32)
    hpo = jnp.zeros(12, jnp.int32) if hand_plays_round is None else jnp.asarray(hand_plays_round, jnp.int32)
    return _SWJ(pr, ps, pm, jnp.asarray(hr), jnp.asarray(hs), jnp.asarray(hm),
                lv, jnp.asarray(jk),
                money=jnp.int32(money), discards_left=jnp.int32(discards_left),
                deck_count=jnp.int32(deck_count),
                hand_plays_run=hpr, hand_plays_round=hpo)


def _assert_match(played, jokers, **kw):
    o = _oracle(played, jokers, **kw)
    k = _kernel(played, jokers, **kw)
    assert (int(o.hand_type), int(o.chips), int(round(o.mult * 1000))) == \
           (int(k[0]), int(k[1]), int(round(float(k[2]) * 1000))), (played, jokers, kw, o, k)
    assert o.score == int(k[3]), (played, jokers, kw, o.score, int(k[3]))


def test_independent_batch1():
    # JOKER +4 mult on a pair of Aces
    _assert_match([(14,0),(14,1)], [1])
    # Jolly +8 mult (pair) ; Sly +50 chips (pair)
    _assert_match([(14,0),(14,1),(7,2)], [6]); _assert_match([(14,0),(14,1),(7,2)], [11])
    # The Duo x2 (pair) ; The Trio x3 (trips)
    _assert_match([(9,0),(9,1)], [131]); _assert_match([(9,0),(9,1),(9,2)], [132])
    # Abstract +3 per joker (here 2 jokers -> +6) ; Banner +30 per discard
    _assert_match([(2,0)], [34, 1]); _assert_match([(2,0)], [22], discards_left=3)
    # Bull +2 per $ ; Blue +2 per deck card ; Half +20 if <=3 cards (and no-fire on 4 cards)
    _assert_match([(2,0)], [93], money=7); _assert_match([(2,0)], [53], deck_count=44)
    _assert_match([(2,0),(3,0),(4,0)], [16])
    _assert_match([(2,0),(3,0),(4,0),(5,1)], [16])              # 4 played -> Half no-fire
    # Mystic +15 if 0 discards (and no-fire with discards left)
    _assert_match([(2,0)], [23], discards_left=0)
    _assert_match([(2,0)], [23], discards_left=2)               # Mystic no-fire
    # Supernova +(plays_run+1) ; Card Sharp x3 if played this round (and no-fire at 0)
    _assert_match([(2,0),(2,1)], [43], hand_plays_run=[0,2,0,0,0,0,0,0,0,0,0,0])
    _assert_match([(2,0),(2,1)], [62], hand_plays_round=[0,1,0,0,0,0,0,0,0,0,0,0])
    _assert_match([(2,0),(2,1)], [62])                          # plays_round=0 -> Card Sharp no-fire
    # Joker Stencil x(empty_slots+1): 1 joker, 4 empty -> x5
    _assert_match([(2,0)], [17])
    # Flush jokers: Droll +10, The Tribe x2, Crafty +80
    flush = [(2,1),(5,1),(8,1),(11,1),(13,1)]
    _assert_match(flush, [10]); _assert_match(flush, [135]); _assert_match(flush, [15])
    # Straight jokers: Crazy +12, The Order x3, Devious +100
    straight = [(2,0),(3,1),(4,2),(5,3),(6,0)]
    _assert_match(straight, [9]); _assert_match(straight, [134]); _assert_match(straight, [14])
    # Two-pair: Mad +10, Clever +80 ; Trips: Zany +12, Wily +100 ; Quads: The Family x4
    _assert_match([(2,0),(2,1),(3,2),(3,3)], [8]); _assert_match([(2,0),(2,1),(3,2),(3,3)], [13])
    _assert_match([(4,0),(4,1),(4,2)], [7]); _assert_match([(4,0),(4,1),(4,2)], [12])
    _assert_match([(4,0),(4,1),(4,2),(4,3)], [133])
    # Seeing Double x2: pair 3♣3♥ -> both score -> Club+other -> FIRES;
    # High Card 2♣3♥ -> only the 3♥ scores (no scoring Club) -> no-fire.
    _assert_match([(3,2),(3,1)], [128]); _assert_match([(2,2),(3,1)], [128])
    # Flower Pot x3 (all four suits)
    _assert_match([(2,0),(3,1),(4,2),(5,3),(6,0)], [122])
    # Blackboard x3 (all held spade/club) ; vacuous when none held
    _assert_match([(2,0)], [48]); _assert_match([(2,0)], [48], held=[(9,0),(9,2)])


def test_on_score_batch2():
    # suit +mult: Greedy(♦+3), Lusty(♥+3), Wrathful(♠+3), Gluttonous(♣+3), Onyx(♣+7)
    _assert_match([(5,3),(7,3)], [2]); _assert_match([(5,1),(7,1)], [3])
    _assert_match([(5,0),(7,0)], [4]); _assert_match([(5,2),(7,2)], [5]); _assert_match([(5,2),(7,2)], [119])
    # suit +chips: Arrowhead(♠+50)
    _assert_match([(5,0),(7,0)], [118])
    # face: Scary +30 chips, Smiley +5 mult (on K/Q)
    _assert_match([(13,0),(12,1)], [33]); _assert_match([(13,0),(12,1)], [104])
    # rank: Fibonacci, Even Steven, Odd Todd, Scholar, Walkie Talkie
    _assert_match([(2,0),(3,1)], [31]); _assert_match([(2,0),(4,1)], [39])
    _assert_match([(3,0),(5,1)], [40]); _assert_match([(14,0),(14,1)], [41]); _assert_match([(10,0),(4,1)], [101])
    # Photograph x2 on first scoring face only (two faces -> still single x2)
    _assert_match([(13,0),(12,1)], [78])
    # Hack retriggers 2-5 (each adds its rank chips again); pair to make them score
    _assert_match([(3,0),(3,1)], [36])
    # Sock & Buskin retriggers faces; pair of Kings
    _assert_match([(13,0),(13,1)], [109])
    # Photograph + Sock&Buskin: first-face card retriggered -> x2 applies twice
    _assert_match([(13,0),(13,1)], [78, 109])
    # ordering: [The Duo x2, Joker +4] vs [Joker +4, The Duo x2] differ; both match oracle
    _assert_match([(14,0),(14,1)], [131, 1]); _assert_match([(14,0),(14,1)], [1, 131])


def test_empty_loadout_reduces_to_score_core():
    """With no jokers, score_with_jokers == score_core for random plain hands."""
    rng = np.random.default_rng(0)
    h_r = jnp.zeros(8, jnp.int8); h_s = jnp.zeros(8, jnp.int8); h_m = jnp.zeros(8, bool)
    levels = jnp.ones(12, jnp.int32)
    jk = _empty_loadout()
    swj = jax.jit(score_with_jokers)  # compile once; shapes are static across iterations
    for _ in range(300):
        n = rng.integers(1, 6)
        ranks = rng.integers(2, 15, size=n); suits = rng.integers(0, 4, size=n)
        pr, ps, pm = _pad5(ranks, suits)
        ht0, c0, m0, sc0 = score_core(pr, ps, pm, levels)
        ht1, c1, m1, sc1 = swj(pr, ps, pm, h_r, h_s, h_m, levels, jk,
                               money=jnp.int32(0), discards_left=jnp.int32(0), deck_count=jnp.int32(0),
                               hand_plays_run=jnp.zeros(12, jnp.int32), hand_plays_round=jnp.zeros(12, jnp.int32))
        assert (int(ht0), int(c0), int(sc0)) == (int(ht1), int(c1), int(sc1))
        assert int(sc1) == int(jnp.floor(c1.astype(jnp.float32) * m1))


def test_score_with_jokers_jit_vmap_batches():
    # build a tiny batch of 3 plain hands, empty loadout
    pr = jnp.array([[14,14,5,0,0],[2,3,4,5,6],[10,10,10,2,2]], jnp.int32)
    ps = jnp.array([[0,1,2,0,0],[0,0,0,0,0],[0,1,2,3,0]], jnp.int32)
    pm = jnp.array([[1,1,1,0,0],[1,1,1,1,1],[1,1,1,1,1]], bool)
    hr = jnp.zeros((3,8), jnp.int32); hs = jnp.zeros((3,8), jnp.int32); hm = jnp.zeros((3,8), bool)
    lv = jnp.ones((3,12), jnp.int32)
    from balatro_rl.envs.actions import MAX_JOKERS
    jk = jnp.zeros((3, MAX_JOKERS), jnp.int32)
    z = jnp.zeros(3, jnp.int32); z12 = jnp.zeros((3,12), jnp.int32)
    out = jax.vmap(lambda *a: score_with_jokers(
        a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
        money=a[8],discards_left=a[9],deck_count=a[10],
        hand_plays_run=a[11],hand_plays_round=a[12]))(
        pr,ps,pm,hr,hs,hm,lv,jk,z,z,z,z12,z12)
    assert out[3].shape == (3,)  # score per env


def test_baron_held():
    # Baron x1.5 per held King; play a pair, hold two Kings -> x1.5*1.5
    _assert_match([(9,0),(9,1)], [72], held=[(13,0),(13,2)])
    _assert_match([(9,0),(9,1)], [72], held=[(13,0),(9,2)])   # one King


# --- full randomized parity over the whole in-scope set ---
from balatro_rl.engine_jax.jokers import INSCOPE_IDS, N_INSCOPE

_FIRED = set()    # dense ids observed firing across the CI corpus (coverage)
_CI_DONE = set()  # CI seeds completed (guards the _FIRED assertion below)
_CI_SEEDS = range(200)


def _random_case(rng):
    n = int(rng.integers(1, 6))
    deck = rng.permutation([(r, s) for r in range(2, 15) for s in range(4)])
    played = [tuple(int(x) for x in deck[i]) for i in range(n)]
    nheld = int(rng.integers(0, 6))
    held = [tuple(int(x) for x in deck[n + i]) for i in range(nheld)]
    k = int(rng.integers(0, 6))
    jokers = [int(rng.choice(INSCOPE_IDS)) for _ in range(k)]
    levels = [int(rng.integers(1, 4)) for _ in range(12)]
    money = int(rng.integers(0, 30)); discards = int(rng.integers(0, 4)); deck_count = int(rng.integers(0, 45))
    hpr = [int(rng.integers(0, 4)) for _ in range(12)]; hpo = [int(rng.integers(0, 3)) for _ in range(12)]
    return dict(played=played, jokers=jokers, held=held, levels=levels, money=money,
                discards_left=discards, deck_count=deck_count, hand_plays_run=hpr, hand_plays_round=hpo)


@pytest.mark.parametrize("seed", _CI_SEEDS)
def test_random_parity_ci(seed):
    rng = np.random.default_rng(seed)
    case = _random_case(rng)
    _assert_match(**case)
    from balatro_rl.engine_jax.jokers import _dense_np
    for j in case["jokers"]:
        _FIRED.add(int(_dense_np[j]))
    _CI_DONE.add(seed)


def test_coverage_every_inscope_joker_appears():
    # Drive enough cases to hit all ids; assert all in-scope ids appear in loadouts.
    rng = np.random.default_rng(12345)
    seen = set()
    for _ in range(4000):
        case = _random_case(rng)
        for j in case["jokers"]:
            seen.add(j)
        if len(seen) == N_INSCOPE:
            break
    assert set(INSCOPE_IDS) <= seen, set(INSCOPE_IDS) - seen
    # And when the full CI sweep ran first (normal file-order run), every dense id
    # must have gone through the kernel-vs-oracle parity assertion at least once.
    if len(_CI_DONE) == len(_CI_SEEDS):
        missing = set(range(1, N_INSCOPE + 1)) - _FIRED
        assert not missing, missing


def test_golden_values_oracle_free():
    # Hand-computed: pair of Aces (PAIR base 10c/2m), Aces score 11+11=22 -> 32c.
    # Joker +4 mult -> mult 6 ; The Duo x2 (pair) -> applied in slot order.
    # Slot order [Joker(+4), Duo(x2)]: (2+4)*2 = 12 mult -> 32*12 = 384.
    o = _kernel([(14,0),(14,1)], [1, 131]); assert int(o[3]) == 384
    # Slot order [Duo(x2), Joker(+4)]: (2*2)+4 = 8 mult -> 32*8 = 256.
    o = _kernel([(14,0),(14,1)], [131, 1]); assert int(o[3]) == 256


@pytest.mark.parametrize("order", [(1,131,6), (6,1,131), (131,6,1)])
def test_fold_order_matches_oracle(order):
    _assert_match([(14,0),(14,1)], list(order))


def test_negative_control_gate_has_teeth():
    # Order sensitivity: [Joker(+4), Duo(x2)] != [Duo(x2), Joker(+4)] (384 vs 256). If the
    # kernel were order-insensitive (a "sum-then-multiply" bug) these would be equal and the
    # episode/golden gates would silently pass a wrong kernel. They must differ.
    a = int(_kernel([(14,0),(14,1)], [1, 131])[3])
    b = int(_kernel([(14,0),(14,1)], [131, 1])[3])
    assert a != b and (a, b) == (384, 256)


def test_out_of_scope_id_is_noop():
    # A deferred joker id (e.g. RIDE_THE_BUS=44) must behave as an empty slot.
    base = _kernel([(14,0),(14,1)], [])
    with_oos = _kernel([(14,0),(14,1)], [44])
    assert int(base[3]) == int(with_oos[3])


def test_max_retrigger_path_parity():
    # Exercises the static unroll bound (1 + MAX_JOKERS passes) and verifies via parity:
    # 5 Hacks all retrigger a played 3 -> 5 retriggers -> 6 passes.
    _assert_match([(3,0),(3,1)], [36, 36, 36, 36, 36])
    # Pareidolia makes a low card count as a face, so Hack AND Sock & Buskin both fire on a 3
    # -> +2 retriggers on that card; parity must still hold.
    _assert_match([(3,0),(3,1)], [36, 109, 37])


def test_planet_levels_parity():
    # Spec §8.C: a leveled loadout (Planet upgrades) + jokers scores at parity. Level the
    # PAIR hand type (index 1) to 3 and add Joker; the kernel must honor levels[ht] exactly.
    lv = [1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    _assert_match([(14,0),(14,1)], [1], levels=lv)
    _assert_match([(9,0),(9,1),(9,2)], [132], levels=[1,1,1,4,1,1,1,1,1,1,1,1])  # trips lvl 4 + The Trio


@pytest.mark.slow
@pytest.mark.skipif(not BALATRO_RUN_SLOW, reason="set BALATRO_RUN_SLOW=1")
def test_random_parity_1000():
    rng = np.random.default_rng(2024)
    for _ in range(1000):
        _assert_match(**_random_case(rng))

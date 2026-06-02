"""Portable, seedable PRNG (splitmix64), threaded purely as immutable state.

Chosen over Python's `random`/`numpy.random` because splitmix64 is trivial to
re-implement bit-for-bit in Rust/C, which preserves cross-implementation parity
when (if) the hot path is ported. Every call returns a NEW RNG — never mutates.
"""
from __future__ import annotations

import dataclasses

_MASK64 = (1 << 64) - 1


@dataclasses.dataclass(frozen=True, slots=True)
class RNG:
    state: int

    @staticmethod
    def from_seed(seed: int) -> "RNG":
        return RNG(state=seed & _MASK64)

    def _next_u64(self) -> tuple[int, "RNG"]:
        s = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z = (z ^ (z >> 31)) & _MASK64
        return z, RNG(state=s)

    def random(self) -> tuple[float, "RNG"]:
        """Uniform float in [0, 1)."""
        z, rng = self._next_u64()
        return (z >> 11) / float(1 << 53), rng

    def randint(self, lo: int, hi: int) -> tuple[int, "RNG"]:
        """Uniform integer in the inclusive range [lo, hi]."""
        z, rng = self._next_u64()
        n = hi - lo + 1
        return lo + (z % n), rng

    def shuffle(self, items: list) -> tuple[list, "RNG"]:
        """Fisher-Yates. Returns a new shuffled list and the advanced RNG."""
        arr = list(items)
        rng = self
        for i in range(len(arr) - 1, 0, -1):
            j, rng = rng.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr, rng

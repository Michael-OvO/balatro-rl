"""ReqScaleCurriculum: ramp the blind-difficulty knob (req_scale) from easy -> real-game as
the agent's rolling blind-clear rate rises.

Why M2 needs it: GRPO learns from reward variance WITHIN a group of rollouts on the same
seed. A frozen 8B loses at ante 1 every game (M1: win_rate 0), so at req_scale=1.0 every
rollout has near-identical low reward and the group-relative advantage ~ 0 -> no gradient.
Starting at a low req_scale makes blinds trivially clearable, so some rollouts clear and
some don't -> reward variance -> a usable signal. As the rolling clear-rate clears a target
over a full window, bump req_scale toward 1.0. Mirrors the E5 PPO closed-loop ramp.
"""
from __future__ import annotations


class ReqScaleCurriculum:
    def __init__(self, start: float = 0.1, end: float = 1.0, step: float = 0.05,
                 clear_rate_target: float = 0.7, window: int = 64):
        if not 0.0 < start <= end:
            raise ValueError(f"need 0 < start ({start}) <= end ({end})")
        self._scale = float(start)
        self._end = float(end)
        self._step = float(step)
        self._target = float(clear_rate_target)
        self._window = int(window)
        self._recent: list[float] = []   # per finished episode: 1.0 if it cleared >=1 blind else 0.0

    @property
    def current(self) -> float:
        return self._scale

    def record(self, cleared: bool) -> None:
        """Record one finished episode's outcome (did it clear at least one blind)."""
        self._recent.append(1.0 if cleared else 0.0)
        if len(self._recent) > self._window:
            self._recent = self._recent[-self._window:]

    def maybe_ramp(self) -> float:
        """Bump req_scale one step if the rolling clear-rate over a FULL window meets the
        target (and we're not already at the end). Resets the window after a ramp so the
        next bump waits for fresh evidence at the harder setting. Returns the new req_scale."""
        if len(self._recent) >= self._window and self._scale < self._end:
            if sum(self._recent) / len(self._recent) >= self._target:
                self._scale = min(self._end, round(self._scale + self._step, 4))
                self._recent = []
        return self._scale

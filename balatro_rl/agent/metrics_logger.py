"""Tiny metrics-logger abstraction. NullLogger (default; collects history in-memory,
test-friendly), ConsoleLogger (streams human-readable progress to stdout so a run
is legible live), TrackioLogger (lazy trackio import -> the live dashboard), and
MultiLogger (fan out to several at once, e.g. console + trackio).

All loggers share one duck-typed interface: `log(metrics: dict, step=None)` and
`finish()`. `train()` calls `log()` once per update (loss/* + train/* keys) and
again on each eval (eval/* keys); the logger decides what to do with each.
"""
from __future__ import annotations

import sys


class NullLogger:
    """No external sink; records (step, metrics) for inspection/tests."""

    def __init__(self):
        self.history: list[tuple] = []

    def log(self, metrics: dict, step=None):
        self.history.append((step, dict(metrics)))

    def finish(self):
        pass


class ConsoleLogger:
    """Streams progress to stdout so training is legible live (and a backgrounded
    log fills as it trains). Prints a compact line per update — throttled to every
    `every` updates so long runs stay readable — and a prominent line on every eval
    (never throttled; evals are infrequent and the metric that matters). Also keeps
    `history` like NullLogger, so it can stand in for one in scripts/tests.

    Formatting degrades gracefully: known keys (loss/*, train/mean_reward, eval/*)
    get a curated layout; anything else falls back to a generic `k=v` join, so the
    logger never breaks if the metric set changes.
    """

    _EVAL_LAYOUT = (
        ("eval/mean_ante", "ante", "{:.2f}"),
        ("eval/max_ante", "max", "{:.0f}"),
        ("eval/win_rate", "win", "{:.3f}"),
        ("eval/mean_run_chips", "chips", "{:.1f}"),
        ("eval/mean_ep_len", "len", "{:.1f}"),
    )

    def __init__(self, every: int = 1, stream=None):
        self.every = max(1, int(every))
        self.stream = stream if stream is not None else sys.stdout
        self.history: list[tuple] = []

    def log(self, metrics: dict, step=None):
        self.history.append((step, dict(metrics)))
        if any(k.startswith("eval/") for k in metrics):
            self._emit("  eval @ " + self._step(step), self._eval_parts(metrics))
        elif step is None or step % self.every == 0:
            self._emit("update " + self._step(step), self._update_parts(metrics))

    def finish(self):
        self.stream.flush()

    @staticmethod
    def _step(step) -> str:
        return "    ?" if step is None else f"{step:>5}"

    def _emit(self, prefix: str, parts: list[str]):
        print(f"{prefix} | " + "  ".join(parts), file=self.stream, flush=True)

    @staticmethod
    def _update_parts(m: dict) -> list[str]:
        parts = []
        if "loss/total" in m:
            parts.append(f"loss {m['loss/total']:.3f}")
        sub = [f"{k.split('/')[-1]} {m[k]:.3f}"
               for k in ("loss/policy", "loss/value", "loss/entropy") if k in m]
        if sub:
            parts.append("(" + " ".join(sub) + ")")
        if "train/mean_reward" in m:
            parts.append(f"reward {m['train/mean_reward']:.4f}")
        return parts or [ConsoleLogger._generic(m)]

    @classmethod
    def _eval_parts(cls, m: dict) -> list[str]:
        parts = []
        for key, label, fmt in cls._EVAL_LAYOUT:
            if key in m:
                parts.append(f"{label} {fmt.format(m[key])}")
        return parts or [cls._generic(m)]

    @staticmethod
    def _generic(m: dict) -> str:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        return "  ".join(f"{k}={fmt(v)}" for k, v in m.items())


class TrackioLogger:
    """Logs to a local-first Trackio run (see docs/reference/observability.md).
    View later with: `trackio show --project <project>`."""

    def __init__(self, project: str, name=None, config=None):
        import trackio
        self._trackio = trackio
        trackio.init(project=project, name=name, config=config)

    def log(self, metrics: dict, step=None):
        self._trackio.log(metrics, step=step)

    def finish(self):
        self._trackio.finish()


class MultiLogger:
    """Fan a single stream of metrics out to several loggers at once — e.g.
    `MultiLogger(ConsoleLogger(), TrackioLogger(...))` to watch progress in the
    terminal while also feeding the dashboard. `None` entries are dropped, so a
    caller can conditionally include a sink without branching."""

    def __init__(self, *loggers):
        self.loggers = [lg for lg in loggers if lg is not None]

    def log(self, metrics: dict, step=None):
        for lg in self.loggers:
            lg.log(metrics, step=step)

    def finish(self):
        for lg in self.loggers:
            lg.finish()

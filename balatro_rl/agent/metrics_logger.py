"""Tiny metrics-logger abstraction. NullLogger (default; collects history in-memory,
test-friendly) and TrackioLogger (lazy trackio import -> the live dashboard).
"""
from __future__ import annotations


class NullLogger:
    """No external sink; records (step, metrics) for inspection/tests."""

    def __init__(self):
        self.history: list[tuple] = []

    def log(self, metrics: dict, step=None):
        self.history.append((step, dict(metrics)))

    def finish(self):
        pass


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

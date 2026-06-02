import pytest

from balatro_rl.engine.jokers.base import REGISTRY


@pytest.fixture(autouse=True)
def _isolate_joker_registry():
    """Snapshot REGISTRY before each test and restore it after, so per-test
    registrations (stubs/fakes) never contaminate other tests."""
    saved = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)

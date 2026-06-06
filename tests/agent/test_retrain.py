"""build_config() reads the swept knobs (seed / lr / curr_floor) from the environment at call time,
so a parallel sweep (scripts/ppo_sweep.sh) varies them per process without code edits. d_model is
deliberately not swept — a warm-started run must shape-match its resumed checkpoint."""
from balatro_rl.agent.retrain import build_config


def test_swept_knobs_default(monkeypatch):
    for k in ("BALATRO_SEED", "BALATRO_LR", "BALATRO_CURR_FLOOR"):
        monkeypatch.delenv(k, raising=False)
    cfg = build_config()
    assert cfg.seed == 0
    assert abs(cfg.lr - 3e-4) < 1e-12
    assert abs(cfg.curr_floor - 0.2) < 1e-12


def test_swept_knobs_from_env(monkeypatch):
    monkeypatch.setenv("BALATRO_SEED", "3")
    monkeypatch.setenv("BALATRO_LR", "6e-4")
    monkeypatch.setenv("BALATRO_CURR_FLOOR", "0.4")
    cfg = build_config()
    assert cfg.seed == 3
    assert abs(cfg.lr - 6e-4) < 1e-12
    assert abs(cfg.curr_floor - 0.4) < 1e-12

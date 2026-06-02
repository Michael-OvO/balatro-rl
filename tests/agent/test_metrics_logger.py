from balatro_rl.agent.metrics_logger import NullLogger, TrackioLogger


def test_null_logger_collects_history():
    lg = NullLogger()
    lg.log({"loss": 1.0}, step=0)
    lg.log({"loss": 0.5, "eval/win_rate": 0.1}, step=1)
    lg.finish()
    assert len(lg.history) == 2
    assert lg.history[0] == (0, {"loss": 1.0})
    assert lg.history[1][1]["eval/win_rate"] == 0.1


def test_trackio_logger_logs_without_crashing(tmp_path, monkeypatch):
    # Use a temp dir so we never touch ~/.cache; trackio logging is non-blocking.
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path))
    lg = TrackioLogger(project="balatro-rl-test", name="unit", config={"lr": 3e-4})
    lg.log({"loss": 1.0}, step=0)
    lg.log({"loss": 0.5}, step=1)
    lg.finish()   # must not raise

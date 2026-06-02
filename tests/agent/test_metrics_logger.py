from balatro_rl.agent.metrics_logger import (
    ConsoleLogger,
    MultiLogger,
    NullLogger,
    TrackioLogger,
)


def test_null_logger_collects_history():
    lg = NullLogger()
    lg.log({"loss": 1.0}, step=0)
    lg.log({"loss": 0.5, "eval/win_rate": 0.1}, step=1)
    lg.finish()
    assert len(lg.history) == 2
    assert lg.history[0] == (0, {"loss": 1.0})
    assert lg.history[1][1]["eval/win_rate"] == 0.1


def test_console_logger_prints_update_and_eval(capsys):
    lg = ConsoleLogger(every=1)
    lg.log({"loss/total": 1.234, "loss/policy": 0.1, "loss/value": 1.0,
            "loss/entropy": 0.9, "train/mean_reward": 0.05}, step=0)
    lg.log({"eval/mean_ante": 1.5, "eval/max_ante": 2.0, "eval/win_rate": 0.25,
            "eval/mean_run_chips": 800.0, "eval/mean_ep_len": 9.0}, step=7)
    out = capsys.readouterr().out
    # update line: curated loss + reward, tagged with step
    assert "update     0" in out
    assert "loss 1.234" in out
    assert "reward 0.0500" in out
    # eval line: prominent, never throttled, distinct prefix
    assert "eval @     7" in out
    assert "ante 1.50" in out and "win 0.250" in out and "chips 800.0" in out
    # also collects history like NullLogger
    assert len(lg.history) == 2


def test_console_logger_throttles_updates_but_never_evals(capsys):
    lg = ConsoleLogger(every=10)
    for i in range(25):
        lg.log({"loss/total": float(i), "train/mean_reward": 0.0}, step=i)
    out = capsys.readouterr().out
    # only steps 0, 10, 20 emit an update line; eval would bypass the throttle
    assert out.count("update") == 3
    lg.log({"eval/win_rate": 0.5}, step=3)   # 3 % 10 != 0, still prints
    assert "eval @     3" in capsys.readouterr().out
    # history records every call regardless of throttling
    assert len(lg.history) == 26


def test_console_logger_falls_back_for_unknown_metrics(capsys):
    lg = ConsoleLogger(every=1)
    lg.log({"custom/thing": 42, "other": 1.5}, step=0)
    out = capsys.readouterr().out
    assert "custom/thing=42" in out and "other=1.5000" in out


def test_multi_logger_fans_out_and_finishes():
    a, b = NullLogger(), NullLogger()
    lg = MultiLogger(a, None, b)   # None entries are dropped
    lg.log({"loss/total": 1.0}, step=0)
    lg.log({"loss/total": 0.5}, step=1)
    lg.finish()
    assert len(a.history) == 2 and len(b.history) == 2
    assert a.history[1] == (1, {"loss/total": 0.5})


def test_trackio_logger_logs_without_crashing(tmp_path, monkeypatch):
    # Use a temp dir so we never touch ~/.cache; trackio logging is non-blocking.
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path))
    lg = TrackioLogger(project="balatro-rl-test", name="unit", config={"lr": 3e-4})
    lg.log({"loss": 1.0}, step=0)
    lg.log({"loss": 0.5}, step=1)
    lg.finish()   # must not raise

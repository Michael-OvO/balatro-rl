import numpy as np
from balatro_rl.agent.train import train, TrainConfig
from balatro_rl.agent.metrics_logger import NullLogger


def _cfg(**over):
    cfg = TrainConfig(num_envs=4, num_steps=16, num_updates=3, d_model=32,
                      num_minibatches=2, update_epochs=2, reward_name="max_depth", seed=0)
    cfg.eval_interval = 1
    cfg.eval_seeds = [0, 1]
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_train_records_eval_history():
    logger = NullLogger()
    result = train(_cfg(), logger=logger)
    # one eval per update (eval_interval=1, 3 updates)
    assert len(result.eval_history) == 3
    for m in result.eval_history:
        assert "eval/win_rate" in m and "eval/mean_ante" in m
        assert all(np.isfinite(v) for v in m.values())
    # logger captured both per-update losses and eval metrics
    keys_logged = set().union(*[d.keys() for _step, d in logger.history])
    assert "loss/total" in keys_logged and "eval/win_rate" in keys_logged


def test_train_default_logger_is_null():
    # No logger passed -> training still runs (uses an internal NullLogger).
    result = train(_cfg(eval_interval=0))   # eval off -> empty history
    assert result.eval_history == []


def test_eval_reports_loop_robust_blinds_metric():
    result = train(_cfg(), logger=NullLogger())
    m = result.eval_history[-1]
    for k in ("eval/mean_blinds_cleared", "eval/max_blinds_cleared", "eval/blind1_clear_rate"):
        assert k in m and np.isfinite(m[k])
    assert m["eval/max_blinds_cleared"] >= m["eval/mean_blinds_cleared"]
    assert 0.0 <= m["eval/blind1_clear_rate"] <= 1.0


def test_ent_coef_fixed_float_logs_constant():
    logger = NullLogger()
    train(_cfg(ent_coef=0.02, eval_interval=0), logger=logger)
    ecs = [d["train/ent_coef"] for _s, d in logger.history if "train/ent_coef" in d]
    assert len(ecs) == 3 and all(abs(e - 0.02) < 1e-12 for e in ecs)


def test_ent_coef_schedule_evaluated_per_update_from_index_zero():
    sched = lambda u: 0.02 * max(0.0, 1.0 - u / 700.0)
    logger = NullLogger()
    train(_cfg(ent_coef=sched, eval_interval=0), logger=logger)
    ecs = [d["train/ent_coef"] for _s, d in logger.history if "train/ent_coef" in d]
    assert abs(ecs[0] - sched(0)) < 1e-12   # FIRST update uses index 0 (off-by-one guard)
    assert abs(ecs[1] - sched(1)) < 1e-12
    assert abs(ecs[2] - sched(2)) < 1e-12


def test_traced_scalar_ent_coef_does_not_retrace():
    """Guards train()'s technique: a 0-d jnp scalar jit-arg traces ONCE regardless of
    its value, so a per-update entropy schedule never recompiles the PPO update."""
    import jax
    import jax.numpy as jnp
    traces = {"n": 0}

    @jax.jit
    def f(x, c):
        traces["n"] += 1          # body runs only during tracing
        return x * c

    for v in (0.02, 0.001, 0.0, 0.5):
        f(jnp.ones(3), jnp.float32(v))
    assert traces["n"] == 1


def test_ramp_scale_logic():
    from balatro_rl.agent.train import _ramp_scale
    cfg = TrainConfig(curr_floor=0.2, ramp_clear_rate=0.7, ramp_step=0.05)
    assert _ramp_scale(0.2, 0.5, True, cfg) == 0.2              # below threshold -> hold
    assert _ramp_scale(0.2, 0.9, False, cfg) == 0.2            # can't bump (window/cooldown) -> hold
    assert abs(_ramp_scale(0.2, 0.9, True, cfg) - 0.25) < 1e-9  # bump allowed -> raise by ramp_step
    assert _ramp_scale(0.98, 0.9, True, cfg) == 1.0           # clamp at the real game
    assert _ramp_scale(1.0, 0.9, True, cfg) == 1.0            # already 1.0 -> stays


def test_train_logs_max_ante_and_scores():
    lg = NullLogger()
    train(TrainConfig(num_updates=2, num_envs=8, num_steps=32, d_model=16,
                      num_minibatches=2, update_epochs=1), logger=lg)
    keys = set().union(*[d.keys() for _s, d in lg.history])
    assert {"train/max_ante", "train/max_hand_score", "train/max_round_score"} <= keys
    antes = [d["train/max_ante"] for _s, d in lg.history if "train/max_ante" in d]
    assert antes and all(a >= 1 for a in antes)


def test_curriculum_smoke_manufactures_clears():
    lg = NullLogger()
    train(TrainConfig(num_updates=3, num_envs=16, num_steps=64, d_model=32,
                      num_minibatches=2, update_epochs=1, curr_floor=0.02), logger=lg)
    rs = [d["train/req_scale"] for _s, d in lg.history if "train/req_scale" in d]
    cr = [d["train/clear_rate"] for _s, d in lg.history if "train/clear_rate" in d]
    assert len(rs) == 3 and all(abs(r - 0.02) < 1e-9 for r in rs)   # ran at the curriculum floor
    assert max(cr) > 0.0                                            # tiny target manufactures clears

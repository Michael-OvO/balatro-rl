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

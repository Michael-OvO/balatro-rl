# Observability: Eval Metrics + Trackio Dashboard — Implementation Plan (Plan 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.
> Trackio API patterns: `docs/reference/observability.md` (verified).

**Goal:** Stop flying blind on *how well* the agent plays. Add **episode-eval metrics** (mean/max ante reached, win-rate, mean run-chips, episode length) computed by running the greedy policy to completion, and a **Trackio dashboard** that logs losses + eval metrics each update. This is the "easy to see stats quickly" half of the observability ask; the Gradio replay viewer is Plan 7.

**Architecture:** `agent/eval.py` runs the argmax policy over eval seeds → a metrics dict (pure, no external deps). `agent/metrics_logger.py` defines a tiny `Logger` protocol with `NullLogger` (collects history; default, test-friendly) and `TrackioLogger` (lazy `import trackio`). `train()` gains an `eval_interval` + a `logger`, logging losses every update and eval metrics every `eval_interval` updates; `TrainResult` gains `eval_history`. A `run.py` CLI wires Trackio for real runs.

**Tech Stack:** Python ≥3.11, jax/flax/optax/numpy, **trackio** (new, optional `viz` extra), pytest. Builds on `balatro_rl/agent/` (ActorCritic, train) + `balatro_rl/envs/` (BalatroEnv).

**Scope:** eval metrics + Trackio dashboard + train integration + CLI. **Deferred (Plan 7):** the Gradio replay viewer (board/score/π-value scrubber) + per-step agent-stat recording.

**Conventions:** repo `/Users/michael/Documents/GitHub/balatro-rl`; `python3 -m pytest`; commit per task (no co-author trailers); feature branch off `master`. Tests use small dims + `NullLogger` (never spawn the Trackio server/cache; the one TrackioLogger test uses a temp `TRACKIO_DIR`).

---

### Task 0: trackio dependency (viz extra)

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/agent/test_trackio_available.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_trackio_available.py
def test_trackio_importable():
    import trackio  # noqa: F401
    assert hasattr(trackio, "init") and hasattr(trackio, "log") and hasattr(trackio, "finish")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/agent/test_trackio_available.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trackio'`

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml`, add an optional `viz` extra (keeps trackio out of the core runtime dep):
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
viz = ["trackio>=0.2"]
```
(If `dev` already lists deps, keep them and add the `viz` line.) Then install: `pip install -e ".[dev,viz]"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/agent/test_trackio_available.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/agent/test_trackio_available.py
git commit -m "chore(agent): add trackio (viz extra) for the dashboard"
```

---

### Task 1: Episode-eval metrics

**Files:**
- Create: `balatro_rl/agent/eval.py`
- Test: `tests/agent/test_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_eval.py
import jax, jax.numpy as jnp
import numpy as np
from balatro_rl.agent.networks import ActorCritic
from balatro_rl.agent.spec import dummy_obs
from balatro_rl.agent.eval import evaluate
from balatro_rl.envs.actions import NUM_ACTIONS


def _params(d=32):
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d)
    p = net.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in dummy_obs(1).items()},
                 jnp.ones((1, NUM_ACTIONS), bool))
    return net, p


def test_evaluate_returns_metric_keys():
    net, p = _params()
    m = evaluate(net, p, seeds=[0, 1, 2], reward_name="max_depth")
    assert set(m.keys()) == {"eval/mean_ante", "eval/max_ante", "eval/win_rate",
                             "eval/mean_run_chips", "eval/mean_ep_len"}
    assert all(np.isfinite(v) for v in m.values())
    assert m["eval/mean_ante"] >= 1.0          # every run reaches at least ante 1
    assert 0.0 <= m["eval/win_rate"] <= 1.0


def test_evaluate_is_deterministic():
    net, p = _params()
    a = evaluate(net, p, seeds=[5, 6], reward_name="max_depth")
    b = evaluate(net, p, seeds=[5, 6], reward_name="max_depth")
    assert a == b                              # greedy policy + fixed seeds -> identical
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/agent/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: ...agent.eval`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/agent/eval.py
"""Greedy-policy evaluation: run the argmax policy to completion over fixed seeds
and report how WELL it plays (ante reached, win-rate, run chips). Deterministic.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..envs.balatro_env import BalatroEnv

_MAX_STEPS = 3000


def _batch(obs: dict):
    return {k: jnp.asarray(v)[None] for k, v in obs.items()}


def evaluate(net, params, seeds, reward_name: str = "shaped") -> dict:
    antes, wins, chips, lengths = [], [], [], []
    for seed in seeds:
        env = BalatroEnv(reward_name)
        obs, mask = env.reset(int(seed))
        run_chips, steps, done = 0, 0, False
        while not done and steps < _MAX_STEPS:
            logits, _ = net.apply(params, _batch(obs), jnp.asarray(mask)[None])
            a = int(jnp.argmax(logits[0]))
            obs, _reward, done, info, mask = env.step(a)
            if info.get("verb") == "play":
                run_chips += int(info.get("score", 0))
            steps += 1
        antes.append(env.state.ante)
        wins.append(1.0 if env.state.won else 0.0)
        chips.append(run_chips)
        lengths.append(steps)
    return {
        "eval/mean_ante": float(np.mean(antes)),
        "eval/max_ante": float(np.max(antes)),
        "eval/win_rate": float(np.mean(wins)),
        "eval/mean_run_chips": float(np.mean(chips)),
        "eval/mean_ep_len": float(np.mean(lengths)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/agent/test_eval.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/agent/eval.py tests/agent/test_eval.py
git commit -m "feat(agent): greedy-policy episode-eval metrics (ante/win-rate/chips)"
```

---

### Task 2: Metrics logger (Null + Trackio)

**Files:**
- Create: `balatro_rl/agent/metrics_logger.py`
- Test: `tests/agent/test_metrics_logger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_metrics_logger.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/agent/test_metrics_logger.py -v`
Expected: FAIL — `ModuleNotFoundError: ...agent.metrics_logger`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/agent/metrics_logger.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/agent/test_metrics_logger.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/agent/metrics_logger.py tests/agent/test_metrics_logger.py
git commit -m "feat(agent): metrics logger (NullLogger + TrackioLogger)"
```

---

### Task 3: Wire eval + logging into the training loop

**Files:**
- Modify: `balatro_rl/agent/train.py`
- Test: `tests/agent/test_train_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_train_eval.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/agent/test_train_eval.py -v`
Expected: FAIL — `TrainConfig` has no `eval_interval` / `train()` has no `logger`

- [ ] **Step 3: Write minimal implementation** (edit `balatro_rl/agent/train.py`)

Add fields to `TrainConfig` (after `seed`):
```python
    eval_interval: int = 0          # run greedy eval every N updates (0 = off)
    eval_seeds: tuple = (0, 1, 2, 3)
```

Add `eval_history` to `TrainResult`:
```python
    eval_history: list = dataclasses.field(default_factory=list)   # one eval-metrics dict per eval
```

Add imports near the top of `train.py`:
```python
from .eval import evaluate
from .metrics_logger import NullLogger
```

Change the `train` signature and wire logging. Replace `def train(cfg: TrainConfig) -> TrainResult:` with:
```python
def train(cfg: TrainConfig, logger=None) -> TrainResult:
    if logger is None:
        logger = NullLogger()
```

At the end of each update iteration (after `mean_returns.append(...)`), add per-update logging and periodic eval:
```python
        update_idx = len(losses) - 1
        logger.log({"loss/total": total, "loss/policy": pg, "loss/value": vl,
                    "loss/entropy": ent, "train/mean_reward": mean_returns[-1]}, step=update_idx)
        if cfg.eval_interval and (update_idx % cfg.eval_interval == 0):
            metrics = evaluate(net, ts.params, cfg.eval_seeds, cfg.reward_name)
            eval_history.append(metrics)
            logger.log(metrics, step=update_idx)
```

Initialize `eval_history = []` alongside `losses, mean_returns = [], []`, call `logger.finish()` before returning, and pass `eval_history=eval_history` into the returned `TrainResult(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/agent/test_train_eval.py tests/agent/test_train.py -v`
Expected: PASS (new eval tests + the Plan-5 train tests still green — `eval_history` defaults to `[]` so existing assertions hold)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/agent/train.py tests/agent/test_train_eval.py
git commit -m "feat(agent): eval metrics + metrics logging in the training loop"
```

---

### Task 4: Training CLI (Trackio dashboard) + full suite

**Files:**
- Create: `balatro_rl/agent/run.py`
- Test: `tests/agent/test_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_run.py
import numpy as np
from balatro_rl.agent.run import run_training
from balatro_rl.agent.train import TrainConfig
from balatro_rl.agent.metrics_logger import NullLogger


def test_run_training_with_null_logger():
    cfg = TrainConfig(num_envs=4, num_steps=16, num_updates=2, d_model=32,
                      num_minibatches=2, update_epochs=1, reward_name="max_depth", seed=0)
    cfg.eval_interval = 1
    cfg.eval_seeds = [0, 1]
    logger = NullLogger()
    result = run_training(cfg, logger=logger)
    assert len(result.losses) == 2
    assert len(result.eval_history) == 2
    assert any("eval/win_rate" in d for _s, d in logger.history)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/agent/test_run.py -v`
Expected: FAIL — `ModuleNotFoundError: ...agent.run`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/agent/run.py
"""Training entry point. `run_training(cfg, logger)` trains with a metrics logger;
the CLI uses a live Trackio dashboard. View it with: `trackio show --project <project>`.
"""
from __future__ import annotations

import dataclasses

from .metrics_logger import NullLogger, TrackioLogger
from .train import TrainConfig, TrainResult, train


def run_training(cfg: TrainConfig, logger=None) -> TrainResult:
    if logger is None:
        logger = NullLogger()
    return train(cfg, logger=logger)


def main():
    cfg = TrainConfig(num_updates=50, num_envs=64, num_steps=128, eval_interval=5)
    logger = TrackioLogger(project="balatro-rl", name="ppo", config=dataclasses.asdict(cfg))
    result = run_training(cfg, logger=logger)
    print(f"done: {len(result.losses)} updates; "
          f"last eval = {result.eval_history[-1] if result.eval_history else 'n/a'}")
    print("view the dashboard with:  trackio show --project balatro-rl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the FULL suite**

Run: `python3 -m pytest -q`
Expected: ALL tests pass (Plans 1–6). (`python3 -m balatro_rl.agent.run` would launch a real 50-update Trackio run — do NOT run it in the test suite; the unit test uses NullLogger.)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/agent/run.py tests/agent/test_run.py
git commit -m "feat(agent): training CLI with Trackio dashboard (run_training)"
```

---

## Self-Review

**1. Spec coverage (design spec §7 dashboard + the "is it learning to play" gap):**
- Episode-eval metrics (ante/win-rate/run-chips/length) → Task 1 ✓
- Metrics logger abstraction (Null + Trackio) → Task 2 ✓
- Eval + logging wired into the training loop → Task 3 ✓
- CLI with the live Trackio dashboard → Task 4 ✓
- **Deferred (Plan 7):** the Gradio replay viewer (board/score/π-value scrubber), per-step agent-stat recording, and richer board rendering.

**2. Placeholder scan:** none — every step has complete code + concrete assertions. The TrackioLogger test uses a temp `TRACKIO_DIR` and relies on Trackio's documented non-blocking behavior (never crashes the caller).

**3. Type consistency:** `evaluate(net, params, seeds, reward_name) -> dict` with the 5 `eval/*` keys used identically in eval/train/tests; `Logger.log(metrics, step)` / `.finish()` consistent across NullLogger/TrackioLogger/train/run; `TrainConfig` (now with `eval_interval`, `eval_seeds`) and `TrainResult` (now with `eval_history`) consistent across train/run/tests; `run_training(cfg, logger) -> TrainResult` consistent.

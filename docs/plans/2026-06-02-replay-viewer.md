# Replay Viewer — Implementation Plan (Plan 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.
> Gradio API patterns: `docs/reference/observability.md` (verified).

**Goal:** The "watch it play" half of observability — a **Gradio replay viewer** that scrubs through a recorded agent episode showing, per step: the board (jokers, hand, blind/score, shop), the score breakdown of a played hand, and the agent's **top action-probabilities + value estimate**. Plus a recorder that captures those per-step beliefs, and a CLI to train-then-record an episode.

**Architecture:** `viz/replay_data.py` (pure, testable): `action_label`, `render_board(state)`, `replay_states(seed, actions)` (engine determinism → all intermediate states), `record_agent_episode(net, params, seed)` → a list of per-step dicts (board + action + reward + value + top-k probs), and JSON save/load. `viz/viewer.py`: a thin `gr.Blocks` app whose `gr.Slider.release` calls a pure `render_step(idx, steps)` → (board, score, prob-table); most logic is in `render_step` (tested without launching Gradio). `viz/record.py`: a CLI that trains briefly then records an episode to JSON.

**Tech Stack:** Python ≥3.11, jax/flax/optax/numpy/trackio, **gradio** (new, added to `viz` extra), pytest. Builds on `balatro_rl/engine/` (engine, render, cards, jokers), `balatro_rl/envs/` (BalatroEnv, actions.decode), `balatro_rl/agent/` (ActorCritic, value_decode, train).

**Scope:** recorder + render functions + Gradio scrubber + record CLI. **Deferred:** pixel/HTML card art (text board is fine); side-by-side sim-vs-real parity view; loading from checkpoints (none yet — the CLI trains then records in-process).

**Conventions:** repo `/Users/michael/Documents/GitHub/balatro-rl`; `python3 -m pytest`; commit per task (no co-author trailers); feature branch off `master`. Tests cover the pure functions; the Gradio app is tested for construction only (never launched in tests).

---

### Task 0: gradio dependency + viz package

**Files:**
- Modify: `pyproject.toml`
- Create: `balatro_rl/viz/__init__.py`
- Test: `tests/viz/__init__.py`, `tests/viz/test_viz_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/viz/test_viz_smoke.py
def test_gradio_and_viz_import():
    import gradio as gr  # noqa: F401
    import balatro_rl.viz  # noqa: F401
    assert hasattr(gr, "Blocks")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/viz/test_viz_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError` (gradio and/or balatro_rl.viz)

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml`, add gradio to the existing `viz` extra:
```toml
viz = ["trackio>=0.2", "gradio>=5.0"]
```
Create empty `balatro_rl/viz/__init__.py` and `tests/viz/__init__.py`. Install: `pip install -e ".[dev,viz]"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/viz/test_viz_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml balatro_rl/viz/ tests/viz/
git commit -m "chore(viz): add gradio (viz extra) and viz package"
```

---

### Task 1: Replay data — labels, board render, state replay, episode recording

**Files:**
- Create: `balatro_rl/viz/replay_data.py`
- Test: `tests/viz/test_replay_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/viz/test_replay_data.py
import jax, jax.numpy as jnp
import numpy as np
from balatro_rl.engine.engine import reset, Verb
from balatro_rl.envs.actions import encode_action
from balatro_rl.agent.networks import ActorCritic
from balatro_rl.agent.spec import dummy_obs
from balatro_rl.envs.actions import NUM_ACTIONS
from balatro_rl.viz.replay_data import (
    action_label, render_board, replay_states, record_agent_episode, save_episode, load_episode,
)


def test_action_label_covers_verbs():
    assert "PLAY" in action_label(encode_action(Verb.PLAY, (0, 1)))
    assert action_label(encode_action(Verb.REROLL, 0)) == "REROLL"
    assert action_label(encode_action(Verb.LEAVE_SHOP, 0)) == "LEAVE SHOP"
    assert "BUY" in action_label(encode_action(Verb.BUY, 1))


def test_render_board_has_key_fields():
    txt = render_board(reset(seed=1))
    assert "Ante 1" in txt and "Hand:" in txt and "Jokers:" in txt and "/300" in txt


def test_replay_states_reconstructs_deterministically():
    seed = 7
    # a short scripted action sequence (discard then play)
    s = reset(seed)
    a0 = encode_action(Verb.DISCARD, (0,))
    a1 = encode_action(Verb.PLAY, (0, 1))
    states = replay_states(seed, [a0, a1])
    assert len(states) == 3                      # before a0, before a1, terminal-ish
    # re-running yields identical states (engine determinism)
    states2 = replay_states(seed, [a0, a1])
    assert all(x.hand == y.hand and x.round_score == y.round_score
               for x, y in zip(states, states2))


def _net_params(d=32):
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d)
    p = net.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in dummy_obs(1).items()},
                 jnp.ones((1, NUM_ACTIONS), bool))
    return net, p


def test_record_agent_episode_step_dicts():
    net, p = _net_params()
    steps = record_agent_episode(net, p, seed=3, reward_name="max_depth")
    assert len(steps) > 0
    s0 = steps[0]
    assert set(["t", "ante", "blind", "phase", "money", "board", "action_id",
                "action_label", "reward", "value", "top_probs"]).issubset(s0.keys())
    # top_probs is a list of [label, prob]; probs are valid
    assert all(0.0 <= p_ <= 1.0 for _lbl, p_ in s0["top_probs"])
    # recorded actions replay back to the same terminal state
    actions = [st["action_id"] for st in steps]
    assert replay_states(3, actions)[-1].done


def test_record_is_deterministic_greedy(tmp_path):
    net, p = _net_params()
    a = record_agent_episode(net, p, seed=5, reward_name="max_depth")
    b = record_agent_episode(net, p, seed=5, reward_name="max_depth")
    assert [s["action_id"] for s in a] == [s["action_id"] for s in b]
    path = tmp_path / "ep.json"
    save_episode(a, path)
    assert [s["action_id"] for s in load_episode(path)] == [s["action_id"] for s in a]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/viz/test_replay_data.py -v`
Expected: FAIL — `ModuleNotFoundError: ...viz.replay_data` (also confirms `encode_action` is importable from `envs.actions` — it is, since Plan-4's review promoted it)

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/viz/replay_data.py
"""Replay data: reconstruct states from (seed, actions), render a text board, and
record a per-step episode (board + action + reward + value + top action-probs) for
the Gradio viewer. Pure/testable; the engine's determinism makes replay exact.
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from ..agent.value_head import value_decode
from ..engine import engine
from ..engine.cards import card_str
from ..engine.engine import Verb
from ..engine.jokers.base import JokerType
from ..engine.shop import joker_cost
from ..engine.state import GameState, Phase
from ..envs.actions import decode
from ..envs.balatro_env import BalatroEnv

_PHASE = {0: "PLAYING", 1: "WON", 2: "LOST", 3: "SHOP"}
_MAX_STEPS = 3000


def action_label(action_id: int) -> str:
    verb, arg = decode(int(action_id))
    if verb in (Verb.PLAY, Verb.DISCARD):
        return f"{verb.name} cards {tuple(arg)}"
    if verb == Verb.BUY:
        return f"BUY offer {arg}"
    if verb == Verb.SELL:
        return f"SELL joker {arg}"
    if verb == Verb.REROLL:
        return "REROLL"
    if verb == Verb.REORDER:
        return f"REORDER {tuple(arg)}"
    if verb == Verb.LEAVE_SHOP:
        return "LEAVE SHOP"
    return verb.name


def render_board(state: GameState) -> str:
    jokers = " | ".join(JokerType(j.type).name for j in state.jokers) or "—"
    hand = " ".join(card_str(c) for c in state.hand) or "—"
    lines = [
        f"Ante {state.ante}  blind {state.blind_index}  [{_PHASE.get(int(state.phase), state.phase)}]",
        f"score {state.round_score}/{state.required}   hands {state.hands_left}  "
        f"discards {state.discards_left}   ${state.money}",
        f"Jokers: {jokers}",
        f"Hand:   {hand}",
    ]
    if int(state.phase) == int(Phase.SHOP) and state.shop_offers:
        offers = "  ".join(f"[{JokerType(o.type).name} ${joker_cost(o.type)}]" for o in state.shop_offers)
        lines.append(f"Shop:   {offers}")
    return "\n".join(lines)


def replay_states(seed: int, actions: list[int]) -> list[GameState]:
    """States before each action plus the final state (engine is pure-deterministic)."""
    state = engine.reset(int(seed))
    states = [state]
    for a in actions:
        if state.done:
            break
        state, _ = engine.step(state, decode(int(a)))
        states.append(state)
    return states


def _b(obs: dict):
    return {k: jnp.asarray(v)[None] for k, v in obs.items()}


def record_agent_episode(net, params, seed: int, reward_name: str = "shaped",
                         topk: int = 6, greedy: bool = True) -> list[dict]:
    apply = jax.jit(net.apply)
    env = BalatroEnv(reward_name)
    obs, mask = env.reset(int(seed))
    key = jax.random.PRNGKey(int(seed))
    steps: list[dict] = []
    done = False
    while not done and len(steps) < _MAX_STEPS:
        state = env.state
        logits, value_logits = apply(params, _b(obs), jnp.asarray(mask)[None])
        probs = np.asarray(jax.nn.softmax(logits[0]))
        value = float(np.asarray(value_decode(value_logits))[0])
        if greedy:
            a = int(np.argmax(np.asarray(logits[0])))
        else:
            key, sub = jax.random.split(key)
            from ..agent.ppo import sample_action
            a = int(np.asarray(sample_action(logits, sub))[0])
        legal = np.flatnonzero(np.asarray(mask))
        order = legal[np.argsort(probs[legal])[::-1][:topk]]
        top = [[action_label(int(i)), float(probs[i])] for i in order]
        board = render_board(state)
        obs, reward, done, info, mask = env.step(a)
        steps.append({
            "t": len(steps), "ante": int(state.ante), "blind": int(state.blind_index),
            "phase": _PHASE.get(int(state.phase)), "money": int(state.money),
            "board": board, "action_id": a, "action_label": action_label(a),
            "reward": float(reward), "value": value,
            "score": info.get("score"), "hand_type": info.get("hand_type"),
            "chips": info.get("chips"), "mult": info.get("mult"),
            "top_probs": top,
        })
    return steps


def save_episode(steps: list[dict], path) -> None:
    with open(path, "w") as f:
        json.dump(steps, f)


def load_episode(path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/viz/test_replay_data.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/viz/replay_data.py tests/viz/test_replay_data.py
git commit -m "feat(viz): episode recorder + board render + deterministic state replay"
```

---

### Task 2: Gradio replay viewer

**Files:**
- Create: `balatro_rl/viz/viewer.py`
- Test: `tests/viz/test_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/viz/test_viewer.py
from balatro_rl.viz.viewer import render_step, build_demo


_STEPS = [
    {"ante": 1, "blind": 0, "phase": "PLAYING", "money": 4,
     "board": "Ante 1  blind 0\nHand: K♠ K♦",
     "action_label": "PLAY cards (0, 1)", "reward": 0.21, "value": 9800.0,
     "score": 405, "hand_type": 1, "chips": 30, "mult": 13.5,
     "top_probs": [["PLAY cards (0, 1)", 0.62], ["DISCARD cards (4,)", 0.10]]},
    {"ante": 1, "blind": 0, "phase": "SHOP", "money": 12,
     "board": "Shop: [Joker $2]", "action_label": "BUY offer 0", "reward": 0.0, "value": 50.0,
     "score": None, "hand_type": None, "chips": None, "mult": None,
     "top_probs": [["BUY offer 0", 0.8], ["LEAVE SHOP", 0.2]]},
]


def test_render_step_play_includes_score_breakdown():
    board, score, probs = render_step(0, _STEPS)
    assert "Ante 1" in board
    assert "405" in score and "13.5" in score        # the play's score breakdown
    assert "value:" in score.lower()
    assert probs == [["PLAY cards (0, 1)", "0.620"], ["DISCARD cards (4,)", "0.100"]]


def test_render_step_shop_no_score_breakdown():
    board, score, probs = render_step(1, _STEPS)
    assert "BUY offer 0" in score
    assert "played" not in score.lower()             # not a play step -> no hand breakdown
    assert probs[0] == ["BUY offer 0", "0.800"]


def test_render_step_empty_is_safe():
    board, score, probs = render_step(0, [])
    assert probs == []


def test_build_demo_constructs():
    demo = build_demo()                               # builds the gr.Blocks app (does not launch)
    import gradio as gr
    assert isinstance(demo, gr.Blocks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/viz/test_viewer.py -v`
Expected: FAIL — `ModuleNotFoundError: ...viz.viewer`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/viz/viewer.py
"""Gradio replay viewer: a slider scrubs through a recorded episode (JSON list of
step dicts from replay_data.record_agent_episode), rendering board + score
breakdown + the agent's top action-probabilities. render_step is pure (tested
without launching Gradio); the gr.Blocks wiring is thin.
"""
from __future__ import annotations

import json


def render_step(step_index: int, steps: list[dict]):
    """(step_index, steps) -> (board_markdown, score_html, prob_rows)."""
    if not steps:
        return "_load an episode_", "", []
    s = steps[int(step_index)]
    board_md = (f"### Ante {s['ante']} · blind {s['blind']} · {s['phase']} · ${s['money']}\n"
                f"```\n{s['board']}\n```")
    parts = [f"<b>action:</b> {s['action_label']}",
             f"<b>reward:</b> {s['reward']:.3f}",
             f"<b>value:</b> {s['value']:.1f}"]
    if s.get("score") is not None:
        parts.append(f"<b>played:</b> hand_type {s['hand_type']} · "
                     f"{s['chips']} × {s['mult']:.2f} = {s['score']}")
    score_html = "<br>".join(parts)
    prob_rows = [[lbl, f"{p:.3f}"] for lbl, p in s["top_probs"]]
    return board_md, score_html, prob_rows


def parse_file(filepath, _state):
    import gradio as gr
    with open(filepath) as f:        # filepath is a NamedString (str path)
        steps = json.load(f)
    n = len(steps)
    board, score, probs = render_step(0, steps)
    slider = gr.Slider(minimum=0, maximum=max(n - 1, 0), step=1, value=0,
                       interactive=True, label=f"Step (0–{n - 1})")
    return steps, slider, board, score, probs


def build_demo():
    import gradio as gr
    with gr.Blocks(title="Balatro RL — Replay") as demo:
        gr.Markdown("# Balatro RL — Replay Viewer")
        traj = gr.State(value=[])
        upload = gr.UploadButton("Load episode (.json)", file_types=[".json"], file_count="single")
        slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Step", interactive=True)
        with gr.Row():
            board = gr.Markdown(label="Board")
            score = gr.HTML(label="Agent")
            probs = gr.Dataframe(headers=["action", "prob"], datatype=["str", "str"],
                                 row_count=1, column_count=2, interactive=False,
                                 label="policy (top actions)")
        upload.upload(parse_file, inputs=[upload, traj],
                      outputs=[traj, slider, board, score, probs])
        slider.release(render_step, inputs=[slider, traj], outputs=[board, score, probs])
    return demo


def main():
    build_demo().launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/viz/test_viewer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/viz/viewer.py tests/viz/test_viewer.py
git commit -m "feat(viz): Gradio replay-scrubber viewer (board + score + policy)"
```

---

### Task 3: record CLI (train → record → JSON) + full suite

**Files:**
- Create: `balatro_rl/viz/record.py`
- Test: `tests/viz/test_record.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/viz/test_record.py
from balatro_rl.viz.record import record_demo
from balatro_rl.viz.replay_data import load_episode, replay_states


def test_record_demo_trains_and_writes_episode(tmp_path):
    out = tmp_path / "episode.json"
    steps = record_demo(out_path=str(out), train_updates=2, seed=0,
                        num_envs=4, num_steps=16, d_model=32)
    assert len(steps) > 0
    loaded = load_episode(out)
    assert [s["action_id"] for s in loaded] == [s["action_id"] for s in steps]
    # recorded actions replay back to a terminal state
    assert replay_states(0, [s["action_id"] for s in loaded])[-1].done
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/viz/test_record.py -v`
Expected: FAIL — `ModuleNotFoundError: ...viz.record`

- [ ] **Step 3: Write minimal implementation**

```python
# balatro_rl/viz/record.py
"""CLI: train a short PPO run in-process, then record the greedy agent's episode
to JSON for the replay viewer. (No checkpoint format yet — train and record in
one process.)  Usage: python -m balatro_rl.viz.record [out.json]
"""
from __future__ import annotations

from ..agent.networks import ActorCritic
from ..agent.train import TrainConfig, train
from ..envs.actions import NUM_ACTIONS
from .replay_data import record_agent_episode, save_episode


def record_demo(out_path: str = "episode.json", train_updates: int = 20, seed: int = 0,
                num_envs: int = 16, num_steps: int = 64, d_model: int = 128,
                reward_name: str = "shaped") -> list[dict]:
    cfg = TrainConfig(num_updates=train_updates, num_envs=num_envs, num_steps=num_steps,
                      d_model=d_model, reward_name=reward_name, seed=seed)
    result = train(cfg)
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d_model)
    steps = record_agent_episode(net, result.params, seed=seed, reward_name=reward_name)
    save_episode(steps, out_path)
    return steps


def main():
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "episode.json"
    steps = record_demo(out_path=out)
    print(f"recorded {len(steps)} steps -> {out}")
    print(f"view it:  python -m balatro_rl.viz.viewer   (then upload {out})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the FULL suite**

Run: `python3 -m pytest -q`
Expected: ALL tests pass (Plans 1–7).

- [ ] **Step 5: Commit**

```bash
git add balatro_rl/viz/record.py tests/viz/test_record.py
git commit -m "feat(viz): record CLI (train -> record greedy episode -> JSON)"
```

---

## Self-Review

**1. Spec coverage (design spec §7.4 replay viewer + the "watch it play" ask):**
- Per-step episode recording (board + action + reward + value + top action-probs) → Task 1 ✓
- Deterministic state replay from `(seed, actions)` → Task 1 ✓
- Text board render (jokers/hand/blind/shop) → Task 1 ✓
- Gradio scrubber (slider → board + score breakdown + policy table) → Task 2 ✓
- Train→record CLI → Task 3 ✓
- **Deferred:** card art/HTML prettiness; sim-vs-real parity side-by-side; checkpoint loading.

**2. Placeholder scan:** none — every step has complete code + concrete assertions. The Gradio app is exercised via the pure `render_step` (4 tests) + a `build_demo()` construction check; it is never `launch()`ed in tests.

**3. Type consistency:** step-dict schema (keys `t/ante/blind/phase/money/board/action_id/action_label/reward/value/score/hand_type/chips/mult/top_probs`) is produced by `record_agent_episode` and consumed by `render_step` identically; `replay_states(seed, actions)->[GameState]`, `record_agent_episode(net, params, seed, ...)->[dict]`, `save_episode/load_episode`, `render_step(idx, steps)->(md, html, rows)` consistent across replay_data/viewer/record/tests; uses `encode_action` (public, from Plan-4's review) in tests and `decode` in code.

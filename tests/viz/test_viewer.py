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

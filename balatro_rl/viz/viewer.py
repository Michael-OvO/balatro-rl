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

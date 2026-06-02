# Observability Stack — Code Reference

Verified (2026) Trackio + Gradio API patterns for the dashboard and replay viewer. Local-first; no HF account needed.

## Trackio (v0.26, `pip install trackio`)
Drop-in wandb API; non-blocking (failures degrade to warnings, never crash training); local SQLite at `~/.cache/huggingface/trackio/` (override `TRACKIO_DIR`).
```python
trackio.init(project, name=None, config=None, group=None, resume="never", ...) -> Run
trackio.log(metrics: dict, step=None)        # step auto-increments if None
trackio.finish()                             # no args; flushes/saves
trackio.show(project=None, host="127.0.0.1", server_port=None, share=None)  # or CLI: `trackio show --project <p>`
```
Minimal use: `init(project="balatro-rl", name=..., config={...})` → per-update `log({"loss/policy":..., "eval/win_rate":...})` → `finish()`. View: `trackio show --project balatro-rl` (default http://127.0.0.1:7860).
- `dataset_id` deprecated → `bucket_id`. Theme via `TRACKIO_THEME` env (not `show(theme=)`). SQLite schema is beta.

## Gradio 5.x (`pip install gradio`) — replay scrubber
```python
import gradio as gr
with gr.Blocks() as demo:
    traj_state = gr.State(value=[])                      # holds any Python obj; per-session
    upload = gr.UploadButton("Load .json", file_types=[".json"], file_count="single")
    step_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Step", interactive=True)
    board = gr.Markdown(); score = gr.HTML()
    probs = gr.Dataframe(headers=["action", "prob"], datatype=["str", "str"],
                         row_count=1, column_count=2, interactive=False)   # row_count int; column_count (NOT col_count)
    upload.upload(fn=parse_file, inputs=[upload, traj_state],
                  outputs=[traj_state, step_slider, board, score, probs])
    step_slider.release(fn=render_step, inputs=[step_slider, traj_state],  # .release fires once per drag
                        outputs=[board, score, probs])
demo.launch(server_name="127.0.0.1", server_port=7860)   # "0.0.0.0" for LAN
```
Wiring rules: `inputs=`/`outputs=` are component(s); the callback returns a tuple matching `outputs` length/order. Slider value is **float → `int()`**. Upload handler with `type="filepath"` (default) receives a `NamedString` path → pass straight to `open()`. Reconfigure a component on load by **returning a fresh instance** (`gr.Slider(minimum=..., maximum=...)`). `gr.Dataframe.value` accepts a list-of-lists.
- Deprecated (removed in 6.0): `row_count=(n,"dynamic")` tuple form, `col_count`, `.style()`.

## OPEN / UNCERTAIN
- Trackio CLI run-management flags (`list`/`get alerts`/`query`) exist but exact syntax uncertain; HTTP `/api/*` documented. `trackio.alert(...)`, `freeze()`, `import_csv()` exist.
- Gradio `row_limits=(min,max)` is in the signature but not yet implemented — use `row_count=int`.

def test_gradio_and_viz_import():
    import gradio as gr  # noqa: F401
    import balatro_rl.viz  # noqa: F401
    assert hasattr(gr, "Blocks")

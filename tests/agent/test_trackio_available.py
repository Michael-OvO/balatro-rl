def test_trackio_importable():
    import trackio  # noqa: F401
    assert hasattr(trackio, "init") and hasattr(trackio, "log") and hasattr(trackio, "finish")

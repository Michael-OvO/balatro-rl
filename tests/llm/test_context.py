from balatro_rl.llm.context import ConversationContext


def test_first_render_has_system_then_user():
    ctx = ConversationContext(system_prompt="RULES", window_turns=3)
    msgs = ctx.render("OBS-0")
    assert msgs[0] == {"role": "system", "content": "RULES"}
    assert msgs[-1]["role"] == "user" and "OBS-0" in msgs[-1]["content"]


def test_window_bounds_number_of_turns():
    ctx = ConversationContext(system_prompt="RULES", window_turns=2)
    for t in range(5):
        ctx.render(f"OBS-{t}")
        ctx.update(assistant_reply=f"REPLY-{t}", observation="")
    msgs = ctx.render("OBS-5")
    # system + at most window_turns*(user,assistant) history + current user.
    assert len(msgs) <= 1 + 2 * 2 + 1
    # oldest turns are dropped; only recent replies survive verbatim.
    blob = "\n".join(m["content"] for m in msgs)
    assert "REPLY-0" not in blob
    assert "REPLY-4" in blob


def test_dropped_turns_are_summarized_not_silently_lost():
    ctx = ConversationContext(system_prompt="RULES", window_turns=1)
    for t in range(3):
        ctx.render(f"OBS-{t}")
        ctx.update(assistant_reply=f"REPLY-{t}", observation="")
    msgs = ctx.render("OBS-3")
    blob = "\n".join(m["content"] for m in msgs)
    assert "earlier turns" in blob.lower()

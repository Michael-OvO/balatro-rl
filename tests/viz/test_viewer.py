import os

from balatro_rl.viz.viewer import (
    build_demo,
    build_probs_html,
    build_reel,
    card_html,
    diff_hands,
    key_event_choices,
    list_episodes,
    render_focus,
    render_step,
    _delta_row,
)


def _card(r, s):
    return {"rank": r, "suit": s, "enh": 0, "ed": 0, "seal": 0}


# kings + low spades; after PLAYing cards 0,1,2 the engine keeps 3.. and draws 3 fresh
_H0 = [_card(13, 0), _card(13, 1), _card(13, 2), _card(2, 0), _card(3, 0),
       _card(4, 0), _card(5, 0), _card(6, 0)]
_H1 = _H0[3:] + [_card(9, 1), _card(10, 2), _card(11, 3)]   # kept 5 + 3 newly drawn


def _play(t, hand, selected, **over):
    s = {"t": t, "ante": 1, "blind": 0, "phase": "PLAYING", "money": 4, "board": "txt",
         "action_id": 0, "action_label": f"PLAY cards {tuple(selected)}", "reward": 0.05,
         "value": 9.0, "score": 60, "hand_type": 1, "chips": 30, "mult": 2.0,
         "top_probs": [[f"PLAY cards {tuple(selected)}", 0.6], ["DISCARD cards (4,)", 0.1]],
         "schema": 2, "verb": "PLAY", "selected": list(selected), "hand": hand,
         "scoring_idx": [], "round_score": 0, "required": 300, "hands_left": 4,
         "discards_left": 3, "jokers": [], "shop_offers": [], "hand_reset": False, "earned": None}
    s.update(over)
    return s


_NEW = [_play(0, _H0, (0, 1, 2)), _play(1, _H1, (0, 1, 2, 3, 4), round_score=60, hands_left=3)]

# legacy episode (no structured 'hand') -> must fall back to the text board
_OLD = [{"ante": 1, "blind": 0, "phase": "PLAYING", "money": 4,
         "board": "Ante 1 blind 0\nHand: K♠ K♦", "action_label": "PLAY cards (0, 1)",
         "reward": 0.21, "value": 9800.0, "score": 405, "hand_type": 1, "chips": 30, "mult": 13.5,
         "top_probs": [["PLAY cards (0, 1)", 0.62], ["DISCARD cards (4,)", 0.10]]}]


def test_card_html_color_rank_and_state():
    blk = card_html(_card(13, 0))                      # K of spades -> black
    assert "K" in blk and "&spades;" in blk and "blk" in blk
    red = card_html(_card(14, 1))                      # A of hearts -> red
    assert "A" in red and "&hearts;" in red and "red" in red
    assert "c-played" in card_html(_card(2, 0), state="played") and "PLAY" in card_html(_card(2, 0), state="played")
    assert "c-new" in card_html(_card(2, 0), state="new") and "NEW" in card_html(_card(2, 0), state="new")
    assert "width:40px" in card_html(_card(2, 0), small=True)


def test_diff_hands_normal_play_is_multiset_before_vs_before():
    left, drawn, note = diff_hands(_NEW[0], _NEW[1])
    assert note is None
    assert {(c["rank"], c["suit"]) for c in left} == {(13, 0), (13, 1), (13, 2)}      # the 3 played
    assert {(c["rank"], c["suit"]) for c in drawn} == {(9, 1), (10, 2), (11, 3)}      # the 3 drawn


def test_diff_hands_reorder_is_a_noop():
    a = _play(0, _H0, ())
    b = _play(1, list(reversed(_H0)), ())          # same multiset, different order
    left, drawn, note = diff_hands(a, b)
    assert left == [] and drawn == [] and note is None


def test_diff_hands_handles_duplicates():
    prev = _play(0, [_card(7, 0), _card(7, 0), _card(8, 1)], ())
    cur = _play(1, [_card(7, 0), _card(8, 1)], ())   # drop ONE of the duplicate 7s
    left, drawn, _ = diff_hands(prev, cur)
    assert len(left) == 1 and (left[0]["rank"], left[0]["suit"]) == (7, 0)   # exactly one, not both


def test_diff_hands_suppresses_blind_redraw():
    a = _play(0, _H0, ())
    b = _play(1, _H1, (), hand_reset=True)
    assert diff_hands(a, b) == ([], [], "new blind: full redraw")


def test_diff_hands_missing_prev_is_safe():
    assert diff_hands(None, _NEW[0]) == ([], [], None)


def test_render_focus_play_shows_banner_progress_and_highlight():
    out = render_focus(0, _NEW)
    assert "PLAY" in out and "PAIR" in out          # hand_type 1 named
    assert "c-played" in out                        # selected cards highlighted
    assert "/ 300 required" in out                  # progress bar context
    assert "&times; 2.00 = <b>60</b>" in out        # chips x mult = score


def test_render_focus_diff_panel_lists_drawn_cards():
    out = render_focus(1, _NEW)
    assert "Newly drawn" in out and "Left the hand" in out


def test_render_focus_falls_back_for_legacy_episode():
    out = render_focus(0, _OLD)
    assert "Ante 1 blind 0" in out and "legacy episode" in out and 'class="pre"' in out
    assert '<div class="card' not in out            # no card tiles rendered (only CSS defines .card)


def test_build_reel_one_tick_per_step_marks_current():
    out = build_reel(1, _NEW)
    assert out.count('<span class="tk') == 2
    assert 'class="tk cur"' in out                  # index 1 is current


def test_key_event_choices_picks_terminal_and_clears():
    lost = [_play(0, _H0, ()), {"t": 1, "phase": "LOST", "ante": 1}]
    assert key_event_choices(lost) == [("step 1: LOST", 1)]


def test_build_probs_html_bolds_chosen():
    out = build_probs_html(0, _NEW)
    assert "pbar-lbl chosen" in out                 # the argmax action is bolded
    assert "PLAY cards (0, 1, 2)" in out


def test_delta_row_direction_and_missing():
    assert "up" in _delta_row("x", 1, 3) and "+2" in _delta_row("x", 1, 3)
    assert "down" in _delta_row("x", 3, 1) and "-2" in _delta_row("x", 3, 1)
    assert "zero" in _delta_row("x", 2, 2)
    assert "&mdash;" in _delta_row("x", None, 5)     # missing value -> em-dash zero row


def test_render_step_empty_is_safe_and_returns_triple():
    out = render_step(0, [])
    assert isinstance(out, tuple) and len(out) == 3 and all(isinstance(x, str) for x in out)
    out2 = render_step(1, _NEW)
    assert len(out2) == 3 and all(isinstance(x, str) for x in out2)


def test_render_focus_terminal_frame_shows_outcome():
    term = _play(2, _H1, (), verb="TERMINAL", phase="LOST", action_label="LOST",
                 score=None, hand_type=None, round_score=115, hands_left=0)
    out = render_focus(0, [term])
    assert "LOST" in out and "115 / 300" in out and "v-loss" in out


def test_build_demo_constructs():
    demo = build_demo()
    import gradio as gr
    assert isinstance(demo, gr.Blocks)


def test_list_episodes_excludes_logs_and_summary(tmp_path):
    (tmp_path / "a.episode.json").write_text("[]")
    (tmp_path / "b.episode.json").write_text("[]")
    (tmp_path / "shaped_ent010__seed0.log").write_text("update 0 | loss ...")
    (tmp_path / "summary.json").write_text("{}")
    names = {os.path.basename(p) for p in list_episodes(str(tmp_path))}
    assert names == {"a.episode.json", "b.episode.json"}

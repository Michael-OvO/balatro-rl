"""Rendering-only tests for the enriched viewer (no JAX). Each test feeds a synthetic
step dict to a pure render_* helper and asserts the new UI surfaces the right data:
joker effect text, the score-breakdown tally, card modifier badges, the boss banner,
and the consumables panel. Legacy steps missing the new keys must still render."""
from balatro_rl.viz.viewer import (
    _boss_banner,
    _consumables_html,
    _jokers_html,
    _score_trace_html,
    card_html,
    render_focus,
)


def _card(r, s, enh=0, ed=0, seal=0):
    return {"rank": r, "suit": s, "enh": enh, "ed": ed, "seal": seal}


def _play(**over):
    s = {"t": 0, "ante": 1, "blind": 0, "phase": "PLAYING", "money": 4, "board": "txt",
         "action_id": 0, "action_label": "PLAY cards (0, 1)", "reward": 0.05, "value": 9.0,
         "score": 60, "hand_type": 1, "chips": 30, "mult": 2.0, "top_probs": [],
         "schema": 2, "verb": "PLAY", "selected": [0, 1],
         "hand": [_card(13, 0), _card(13, 1)], "scoring_idx": [], "round_score": 0,
         "required": 300, "hands_left": 4, "discards_left": 3, "jokers": [],
         "shop_offers": [], "hand_reset": False, "earned": None,
         "boss": {}, "consumables": [], "score_trace": []}
    s.update(over)
    return s


# --- 1. joker abilities visible ---------------------------------------------- #
def test_jokers_html_shows_effect_description():
    s = _play(jokers=[{"type": 1, "name": "GREEDY", "desc": "+3 Mult per Diamond played",
                       "counter": 0.0, "edition": 0, "sell": 2}])
    out = _jokers_html(s, None)
    assert "GREEDY" in out
    assert "+3 Mult per Diamond played" in out          # effect visible at a glance
    assert "jk-desc" in out


def test_jokers_html_keeps_scaling_counter():
    s = _play(jokers=[{"type": 2, "name": "RIDE_THE_BUS", "desc": "x1 Mult, scales",
                       "counter": 7.0, "edition": 0, "sell": 3}])
    out = _jokers_html(s, None)
    assert "&times;7" in out and "jk-cnt" in out


# --- 2. score-breakdown panel (centerpiece) ---------------------------------- #
_TRACE = [
    {"label": "Pair base", "chips": 10, "mult": 2},
    {"label": "K♠", "chips": 20, "mult": 2},
    {"label": "K♥", "chips": 30, "mult": 2},
    {"label": "Greedy Joker", "chips": 30, "mult": 6},
    {"label": "Glass card", "chips": 30, "mult": 12},
]


def test_score_trace_renders_labels_and_final_score():
    out = _score_trace_html(_play(score_trace=_TRACE))
    for entry in _TRACE:
        assert entry["label"] in out                    # every contribution labelled
    assert "Score breakdown" in out
    assert "trace-row final" in out
    assert ">360<" in out                               # 30 x 12 = 360 final score
    assert "= " in out                                  # chips x mult = score form


def test_score_trace_shows_deltas():
    out = _score_trace_html(_play(score_trace=_TRACE))
    assert "+10 chips" in out                            # 10 -> 20 chips on first scored card
    assert "&times;3 mult" in out                        # Greedy bumps mult 2 -> 6 (x3)
    assert "&times;2 mult" in out                        # Glass card doubles mult (6 -> 12)


def test_score_trace_empty_for_non_play():
    assert _score_trace_html(_play(verb="DISCARD", score_trace=[])) == ""
    assert _score_trace_html(_play(score_trace=[])) == ""


# --- 3. card modifier badges ------------------------------------------------- #
def test_card_html_shows_glass_foil_gold_seal_badges():
    out = card_html(_card(13, 0, enh=4, ed=1, seal=1))   # GLASS / FOIL / GOLD seal
    assert "Glass" in out and "bd-enh" in out
    assert "Foil" in out and "bd-ed" in out
    assert "Gold seal" in out and "bd-seal" in out
    assert "badges" in out


def test_card_html_no_badges_when_plain():
    out = card_html(_card(13, 0))
    assert "badge" not in out                            # plain card -> no modifier tags


def test_card_html_legacy_card_without_mod_keys_is_safe():
    out = card_html({"rank": 5, "suit": 2})              # no enh/ed/seal keys at all
    assert "badge" not in out and "card" in out


# --- 4. boss banner ---------------------------------------------------------- #
def test_boss_banner_shows_name_and_effect():
    s = _play(boss={"id": 1, "name": "The Club",
                    "desc": "All Club cards are debuffed"})
    out = _boss_banner(s)
    assert "The Club" in out
    assert "All Club cards are debuffed" in out
    assert "boss-name" in out and "boss-desc" in out


def test_boss_banner_empty_without_boss():
    assert _boss_banner(_play(boss={})) == ""
    assert _boss_banner({"verb": "PLAY"}) == ""          # legacy step lacks the key


# --- 5. consumables panel ---------------------------------------------------- #
def test_consumables_panel_shows_name_and_effect():
    s = _play(consumables=[{"kind": 0, "type_id": 0, "name": "Mercury",
                            "desc": "Level up Pair: +1 Mult and +15 Chips"}])
    out = _consumables_html(s)
    assert "Mercury" in out
    assert "Level up Pair: +1 Mult and +15 Chips" in out
    assert "con-name" in out and "Consumables" in out


def test_consumables_panel_empty_when_none():
    assert _consumables_html(_play(consumables=[])) == ""
    assert _consumables_html({"verb": "PLAY"}) == ""     # legacy step lacks the key


# --- integration: render_focus surfaces all of the above --------------------- #
def test_render_focus_play_includes_all_enrichments():
    s = _play(
        score_trace=_TRACE,
        jokers=[{"type": 1, "name": "GREEDY", "desc": "+3 Mult per Diamond played",
                 "counter": 0.0, "edition": 0, "sell": 2}],
        consumables=[{"kind": 0, "type_id": 0, "name": "Mercury", "desc": "Level up Pair"}],
        boss={"id": 1, "name": "The Club", "desc": "All Club cards are debuffed"},
        hand=[_card(13, 0, enh=4), _card(13, 1)],
    )
    out = render_focus(0, [s])
    assert "+3 Mult per Diamond played" in out           # joker ability
    assert "Score breakdown" in out and ">360<" in out   # score tally + final
    assert "Glass" in out and "bd-enh" in out            # card badge in the hand
    assert "The Club" in out and "boss-name" in out      # boss banner
    assert "Mercury" in out and "con-name" in out        # consumables panel


def test_render_focus_legacy_step_without_new_keys_is_safe():
    # a schema-v2 step that predates the Phase-D keys: must still render the hand.
    s = _play()
    for k in ("boss", "consumables", "score_trace"):
        s.pop(k)
    out = render_focus(0, [s])
    assert "PAIR" in out                                 # banner still renders
    # none of the new panels render their *content* (the CSS classes live in <style>):
    assert '<div class="boss">' not in out               # no boss banner div
    assert '<div class="consums">' not in out            # no consumables panel
    assert '<div class="trace">' not in out              # no score tally

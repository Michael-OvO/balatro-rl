"""Gradio replay viewer: scrub a recorded episode and SEE what the agent did.

Per step it renders (all in gr.HTML, no text blob):
  - a big ACTION banner (PLAY -> hand type, DISCARD, BUY, blind-cleared, ...),
  - a progress bar (round_score / required) + hands/discards/$,
  - the hand as real card tiles with the action's cards highlighted in place,
  - a "what changed" diff panel: cards that LEFT (played/discarded) and were newly
    DRAWN (engine-correct before-vs-before multiset diff), plus score/$/hands deltas,
  - a colored timeline reel + a "jump to key event" dropdown for fast navigation,
  - the policy's top action-probs as a bar chart with the chosen action bolded.

The render_* functions are PURE (no gradio import) and substring-snapshot-testable.
Old episodes lacking the structured `hand` field fall back to the legacy text board.
"""
from __future__ import annotations

import glob
import html
import json
import os
from collections import Counter

# The picker lists episodes from here (where the sweep writes <run>.episode.json).
EPISODE_DIR = os.environ.get("BALATRO_EPISODE_DIR", "/tmp/sweep_out")


def list_episodes(dirpath: str = EPISODE_DIR) -> list[str]:
    """Recorded-episode JSONs in ``dirpath`` (``*.episode.json``), newest first.
    Training ``*.log`` and ``summary.json`` are excluded — they hold scalars, not a
    game trajectory, so the replay viewer cannot render them."""
    return sorted(glob.glob(os.path.join(dirpath, "*.episode.json")),
                  key=os.path.getmtime, reverse=True)


def _choices():
    return [(os.path.basename(p), p) for p in list_episodes()]


# ----------------------------------------------------------------------------- #
# Pure rendering (no gradio import). Mirrors cards.py glyph/colour rules.
# ----------------------------------------------------------------------------- #
_SUIT_GLYPH = {0: "&spades;", 1: "&hearts;", 2: "&clubs;", 3: "&diams;"}
_RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}
_HAND_TYPE_NAMES = {0: "High Card", 1: "Pair", 2: "Two Pair", 3: "Three of a Kind",
                    4: "Straight", 5: "Flush", 6: "Full House", 7: "Four of a Kind",
                    8: "Straight Flush", 9: "Five of a Kind", 10: "Flush House",
                    11: "Flush Five"}
# verb -> (banner css class, reel tick colour, tick height px)
_VERB_STYLE = {"PLAY": ("v-play", "#2f7fe0", 22), "DISCARD": ("v-disc", "#6b7280", 16),
               "BUY": ("v-buy", "#8b5cf6", 16), "SELL": ("v-sell", "#b45309", 16),
               "REROLL": ("v-shop", "#7c5cff", 14), "REORDER": ("v-shop", "#7c5cff", 14),
               "LEAVE_SHOP": ("v-shop", "#7c5cff", 14)}

_STYLE = """<style>
.bv{font-family:ui-sans-serif,system-ui,Arial;color:#e8e8ea}
.bv-banner{display:flex;align-items:center;gap:14px;padding:11px 16px;border-radius:9px;
  font-size:20px;font-weight:800;color:#fff !important;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.bv-banner .sub{font-size:14px;font-weight:600;opacity:.95}
.v-play{background:#2f7fe0}.v-disc{background:#6b7280}.v-clear{background:#f59e0b}
.v-buy{background:#8b5cf6}.v-sell{background:#b45309}.v-shop{background:#7c5cff}
.v-win{background:#22a957}.v-loss{background:#d6453a}.v-other{background:#475569}
.bv-prog-wrap{margin:9px 0 2px}
.bv-prog-label{font-size:12px;font-weight:700;color:#9aa0b0;margin-bottom:3px}
.bv-prog{height:16px;background:#27272a;border-radius:8px;overflow:hidden}
.bv-prog-fill{height:100%;background:linear-gradient(90deg,#ffb02e,#ff6f3c)}
.bv-prog-fill.done{background:#22a957}
.bv-prog-num{font-size:11px;color:#9aa0b0;margin-top:2px}
.bv-cols{display:flex;gap:16px;margin-top:11px}
.bv-table{flex:3}.bv-diff{flex:2;background:#171a23;border-radius:9px;padding:9px 11px}
.bv-sec{font-size:11px;font-weight:800;letter-spacing:.05em;color:#8b90a0;
  text-transform:uppercase;margin:4px 0 6px}
.bv-row{display:flex;gap:7px;flex-wrap:wrap;align-items:flex-end;min-height:84px}
.card{position:relative;width:52px;height:72px;border-radius:7px;background:#fff;
  border:1px solid #ccc;box-shadow:0 1px 3px rgba(0,0,0,.4);display:flex;
  flex-direction:column;justify-content:space-between;padding:4px 6px;box-sizing:border-box}
.card .r{font-size:14px;font-weight:800;line-height:1}
.card .big{font-size:24px;text-align:center;line-height:1}
.card .br{font-size:12px;font-weight:800;text-align:right;line-height:1}
.red{color:#d6453a}.blk{color:#141414}
.c-played{border:3px solid #2f7fe0;transform:translateY(-9px)}
.c-disc{border:3px solid #6b7280;transform:translateY(-5px);opacity:.55}
.c-new{border:3px solid #22a957;box-shadow:0 0 8px rgba(34,169,87,.55)}
.c-left{filter:grayscale(1);border:2px dashed #888;opacity:.7}
.c-noscore{opacity:.4}
.ribbon{position:absolute;top:-8px;left:50%;transform:translateX(-50%);font-size:8px;
  font-weight:800;color:#fff;padding:1px 5px;border-radius:6px}
.rb-play{background:#2f7fe0}.rb-disc{background:#6b7280}.rb-new{background:#22a957}
.jokers{display:flex;gap:6px;margin-bottom:9px;flex-wrap:wrap}
.jk{font-size:11px;font-weight:700;background:#3a2d5c;color:#fff !important;padding:3px 8px;border-radius:6px}
.jk.add{background:#15803d}.jk.rem{background:#7f1d1d;text-decoration:line-through;opacity:.65}
.shop{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.offer{font-size:12px;font-weight:700;background:#2a2540;color:#cbb9ff;
  padding:5px 9px;border-radius:7px;border:1px solid #4c3f7a}
.offer.target{border:2px solid #f59e0b;color:#ffe2ad}
.delta{display:flex;justify-content:space-between;font-size:13px;padding:2px 0;border-bottom:1px solid #262a36}
.up{color:#22c55e !important;font-weight:800}.down{color:#ef6a5e !important;font-weight:800}.zero{color:#9aa0b0 !important}
.note{font-size:12px;color:#9aa0b0;font-style:italic;padding:6px 0}
.reel{display:flex;gap:2px;align-items:flex-end;height:38px;padding:6px 4px;
  background:#0f1115;border-radius:8px;overflow-x:auto}
.tk{width:8px;border-radius:2px;flex:0 0 auto}
.tk.cur{box-shadow:0 0 0 2px #fff,0 0 6px currentColor;width:10px}
.legend{font-size:11px;color:#9aa0b0;margin:4px 2px}
.legend span{margin-right:8px}
.probs{font-size:12px}
.pbar-row{display:flex;align-items:center;gap:8px;margin:3px 0}
.pbar-lbl{width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#cbd0da}
.pbar-lbl.chosen{font-weight:800;color:#fff}
.pbar-track{flex:1;height:11px;background:#27272a;border-radius:6px;overflow:hidden}
.pbar-fill{height:100%;background:#475569}
.pbar-fill.chosen{background:#2f7fe0}
.pbar-num{width:46px;text-align:right;color:#9aa0b0}
.pre{white-space:pre;font-family:ui-monospace,Menlo,monospace;font-size:12px;
  background:#0f1115;color:#d4d4d8;padding:10px;border-radius:8px;overflow-x:auto}
</style>"""


def _rank_txt(rank):
    return _RANK_NAMES.get(rank, str(rank))


def _is_red(suit):
    return suit in (1, 3)


def _ckey(c):   # identity for the multiset diff; tolerant of missing modifier keys
    return (c["rank"], c["suit"], c.get("enh", 0), c.get("ed", 0), c.get("seal", 0))


def card_html(card: dict, *, state: str = "held", small: bool = False) -> str:
    """One self-contained card tile. state in held|played|disc|new|left.

    Colours are set INLINE (not via the .red/.blk classes) because Gradio's theme
    applies a descendant text-colour that overrides class-based colour; inline wins.
    The suit glyph carries VS15 (&#xFE0E;) to force monochrome TEXT, not an emoji."""
    col = "red" if _is_red(card["suit"]) else "blk"
    hexc = "#d6453a" if col == "red" else "#141414"
    r = _rank_txt(card["rank"])
    g = _SUIT_GLYPH[card["suit"]] + "&#xFE0E;"
    cls = {"played": " c-played", "disc": " c-disc", "new": " c-new", "left": " c-left"}.get(state, "")
    rib = {"played": '<span class="ribbon rb-play">PLAY</span>',
           "disc": '<span class="ribbon rb-disc">DISC</span>',
           "new": '<span class="ribbon rb-new">NEW</span>'}.get(state, "")
    cstyle = "background:#fff !important" + (";width:40px;height:56px" if small else "")
    tcol = f"color:{hexc} !important"
    return (f'<div class="card{cls}" style="{cstyle}">{rib}'
            f'<div class="r {col}" style="{tcol}">{r}</div>'
            f'<div class="big {col}" style="{tcol}">{g}</div>'
            f'<div class="br {col}" style="{tcol}">{r}{g}</div></div>')


def _delta_row(label, prev, cur, *, money=False):
    if prev is None or cur is None:
        return f'<div class="delta"><span>{label}</span><span class="zero">&mdash;</span></div>'
    d = cur - prev
    cls = "up" if d > 0 else ("down" if d < 0 else "zero")
    arrow = "&uarr;" if d > 0 else ("&darr;" if d < 0 else "")
    pre = "$" if money else ""
    sign = f"+{d}" if d > 0 else (str(d) if d < 0 else "0")
    return (f'<div class="delta"><span>{label}</span>'
            f'<span class="{cls}">{sign} {arrow} ({pre}{prev}&rarr;{pre}{cur})</span></div>')


def diff_hands(prev_step: dict, cur_step: dict):
    """PURE before-vs-before diff -> (left_cards, drawn_cards, suppressed_note|None).
    left = cards in prev hand absent from cur hand (played/discarded by the PREVIOUS
    action); drawn = cards in cur hand absent from prev. Counter over _ckey handles
    duplicates and makes REORDER a no-op. Suppressed on a fresh-blind redraw."""
    if prev_step is None or "hand" not in prev_step or "hand" not in cur_step:
        return [], [], None
    if prev_step.get("hand_reset") or cur_step.get("hand_reset"):
        return [], [], "new blind: full redraw"
    pc = Counter(_ckey(c) for c in prev_step["hand"])
    cc = Counter(_ckey(c) for c in cur_step["hand"])
    left_keys, drawn_keys = pc - cc, cc - pc

    def _expand(step, keys):
        out, used = [], Counter()
        for c in step["hand"]:
            k = _ckey(c)
            if keys[k] - used[k] > 0:
                out.append(c)
                used[k] += 1
        return out

    return _expand(prev_step, left_keys), _expand(cur_step, drawn_keys), None


def _banner(s):
    verb = s.get("verb", "")
    if verb == "TERMINAL":
        won = s.get("phase") == "WON"
        cls = "v-win" if won else "v-loss"
        head = "&#127942; WON THE RUN" if won else "&#128128; RUN OVER &mdash; LOST"
        sub = f'final score {s.get("round_score", "?")} / {s.get("required", "?")} required'
        return f'<div class="bv-banner {cls}"><span>{head}</span><span class="sub">{sub}</span></div>'
    cls = _VERB_STYLE.get(verb, ("v-other", "#475569", 16))[0]
    if s.get("score") is not None:
        ht = _HAND_TYPE_NAMES.get(s.get("hand_type"), f"hand {s.get('hand_type')}")
        chips, mult = s.get("chips"), s.get("mult")
        cleared = s.get("round_score", 0) + (s.get("score") or 0) >= s.get("required", 1 << 30)
        if cleared:
            cls = "v-clear"
        sub = f'{chips} &times; {mult:.2f} = <b>{s["score"]}</b> chips'
        if cleared:
            sub += ' &middot; <b>BLIND CLEARED &check;</b>'
        head = f'&#9654; PLAY &rarr; {html.escape(ht.upper())}'
    elif verb == "DISCARD":
        head = f'&#9851; DISCARD {len(s.get("selected", []))} cards'
        sub = ""
    elif verb == "BUY":
        head = f'&#128722; {html.escape(s.get("action_label", ""))}'
        sub = (f'-${(-(s["earned"]))}' if s.get("earned") else "")
    else:
        head = html.escape(s.get("action_label", verb))
        sub = ""
    return f'<div class="bv-banner {cls}"><span>{head}</span><span class="sub">{sub}</span></div>'


def _progress(s):
    rs, req = s.get("round_score"), s.get("required")
    if rs is None or req is None or req <= 0:
        return ""
    pct = max(0, min(100, int(100 * rs / req)))
    done = " done" if rs >= req else ""
    return (f'<div class="bv-prog-wrap"><div class="bv-prog-label">'
            f'Ante {s["ante"]} &middot; blind {s["blind"]}</div>'
            f'<div class="bv-prog"><div class="bv-prog-fill{done}" style="width:{pct}%"></div></div>'
            f'<div class="bv-prog-num">round score {rs} / {req} required &middot; '
            f'hands {s.get("hands_left", "?")} &middot; discards {s.get("discards_left", "?")} '
            f'&middot; ${s["money"]}</div></div>')


def _jokers_html(cur, prev):
    cur_j = cur.get("jokers", [])
    prev_j = (prev or {}).get("jokers", [])
    cur_names = Counter(j["name"] for j in cur_j)
    prev_names = Counter(j["name"] for j in prev_j)
    chips = []
    for j in cur_j:
        c = f' &times;{j["counter"]:.0f}' if j.get("counter") else ""
        cls = " add" if cur_names[j["name"]] > prev_names[j["name"]] else ""
        chips.append(f'<span class="jk{cls}">{html.escape(j["name"])}{c}</span>')
    for name in (prev_names - cur_names).elements():   # sold / lost since prev step
        chips.append(f'<span class="jk rem">{html.escape(name)}</span>')
    if not chips:
        chips = ['<span class="jk" style="opacity:.5">no jokers</span>']
    return '<div class="bv-sec">Jokers</div><div class="jokers">' + "".join(chips) + "</div>"


def _hand_html(s):
    """Current hand with the action's cards highlighted in place (intent for THIS step)."""
    sel = set(s.get("selected", []))
    score_idx = set(s.get("scoring_idx") or [])
    verb = s.get("verb")
    tiles = []
    for i, c in enumerate(s["hand"]):
        if i in sel and verb == "PLAY":
            extra = "" if (not score_idx or i in score_idx) else " c-noscore"
            tiles.append(card_html(c, state="played").replace(
                'class="card c-played"', f'class="card c-played{extra}"'))
        elif i in sel and verb == "DISCARD":
            tiles.append(card_html(c, state="disc"))
        else:
            tiles.append(card_html(c, state="held"))
    cap = ("this step PLAYs the highlighted cards" if verb == "PLAY" else
           "this step DISCARDs the highlighted cards" if verb == "DISCARD" else "current hand")
    return (f'<div class="bv-sec">Hand &middot; {cap}</div>'
            f'<div class="bv-row">' + "".join(tiles) + "</div>")


def _shop_html(s):
    offers = s.get("shop_offers", [])
    if not offers:
        return ""
    chips = "".join(f'<span class="offer">{html.escape(o["name"])} ${o["cost"]}</span>' for o in offers)
    return '<div class="bv-sec">Shop offers</div><div class="shop">' + chips + "</div>"


def _diff_panel(cur, prev):
    left, drawn, suppressed = diff_hands(prev, cur)
    blocks = ['<div class="bv-sec">What changed' + (f' (vs step {prev["t"]})' if prev else "") + '</div>']
    if suppressed:
        blocks.append(f'<div class="note">{suppressed}</div>')
    elif "hand" not in cur or prev is None:
        blocks.append('<div class="note">card-level diff unavailable for this step</div>')
    elif not left and not drawn:
        blocks.append('<div class="note">hand unchanged (shop / reorder action)</div>')
    else:
        prev_verb = (prev or {}).get("verb")
        blocks.append('<div class="bv-sec">Left the hand'
                      + (f' &middot; {prev_verb.lower()}' if prev_verb in ("PLAY", "DISCARD") else "")
                      + '</div><div class="bv-row" style="min-height:64px">'
                      + "".join(card_html(c, state="left", small=True) for c in left) + "</div>")
        blocks.append('<div class="bv-sec">Newly drawn</div>'
                      '<div class="bv-row" style="min-height:64px">'
                      + "".join(card_html(c, state="new", small=True) for c in drawn) + "</div>")
    if prev is not None:
        blocks.append('<div class="bv-sec">Deltas</div>')
        blocks.append(_delta_row("score", prev.get("round_score"), cur.get("round_score")))
        blocks.append(_delta_row("hands left", prev.get("hands_left"), cur.get("hands_left")))
        blocks.append(_delta_row("discards left", prev.get("discards_left"), cur.get("discards_left")))
        blocks.append(_delta_row("money", prev.get("money"), cur.get("money"), money=True))
    r = cur.get("reward")
    if r is not None:
        rc = "up" if r > 0 else ("down" if r < 0 else "zero")
        blocks.append(f'<div class="delta"><span>reward</span><span class="{rc}">{r:+.3f}</span></div>')
    return "".join(blocks)


def render_focus(step_index: int, steps: list[dict]) -> str:
    """PURE: composed focus HTML (banner+progress+hand/shop+diff). Falls back to the
    text board for OLD episodes lacking 'hand'. Returns the full string incl. <style>."""
    if not steps:
        return _STYLE + '<div class="bv"><div class="note">load an episode</div></div>'
    i = int(step_index)
    s = steps[i]
    prev = steps[i - 1] if i > 0 else None
    if "hand" not in s:   # OLD episode -> graceful fallback to the text blob
        head = (f'<div class="bv-banner v-other"><span>{html.escape(s.get("action_label", ""))}'
                f'</span><span class="sub">reward {s["reward"]:+.3f} &middot; '
                f'value {s["value"]:.1f}</span></div>')
        return (_STYLE + '<div class="bv">' + head
                + '<div class="bv-sec">Board (legacy episode)</div>'
                + f'<div class="pre">{html.escape(s["board"])}</div></div>')
    in_shop = s.get("phase") == "SHOP" or bool(s.get("shop_offers"))
    table = _jokers_html(s, prev) + (_shop_html(s) if in_shop else _hand_html(s))
    body = (_banner(s) + _progress(s)
            + '<div class="bv-cols"><div class="bv-table">' + table + '</div>'
            + '<div class="bv-diff">' + _diff_panel(s, prev) + '</div></div>')
    return _STYLE + '<div class="bv">' + body + "</div>"


def build_reel(step_index: int, steps: list[dict]) -> str:
    """PURE: display-only colored timeline (one tick per step)."""
    if not steps:
        return _STYLE + '<div class="reel"></div>'
    cur = int(step_index)
    ticks = []
    for j, s in enumerate(steps):
        verb = s.get("verb", "")
        color = _VERB_STYLE.get(verb, ("v-other", "#475569", 14))[1]
        h = _VERB_STYLE.get(verb, (None, None, 14))[2]
        if s.get("phase") == "WON":
            color, h = "#22a957", 26
        elif s.get("phase") == "LOST":
            color, h = "#d6453a", 26
        elif s.get("score") is not None and (s.get("round_score", 0) + (s.get("score") or 0)
                                             >= s.get("required", 1 << 30)):
            color, h = "#f59e0b", 26
        cur_cls = " cur" if j == cur else ""
        ticks.append(f'<span class="tk{cur_cls}" title="step {j} &middot; {verb}" '
                     f'style="height:{h}px;background:{color};color:{color}"></span>')
    legend = ('<div class="legend">step %d / %d &nbsp;'
              '<span style="color:#2f7fe0">&#9646;play</span>'
              '<span style="color:#6b7280">&#9646;discard</span>'
              '<span style="color:#f59e0b">&#9646;clear</span>'
              '<span style="color:#7c5cff">&#9646;shop</span>'
              '<span style="color:#22a957">&#9646;win</span>'
              '<span style="color:#d6453a">&#9646;loss</span></div>') % (cur, len(steps) - 1)
    return _STYLE + '<div class="reel">' + "".join(ticks) + "</div>" + legend


def build_probs_html(step_index: int, steps: list[dict]) -> str:
    """PURE: policy top-probs as an HTML bar list; the chosen action is bolded."""
    if not steps:
        return ""
    s = steps[int(step_index)]
    chosen = s.get("action_label")
    rows = []
    for lbl, p in s.get("top_probs", []):
        c = " chosen" if lbl == chosen else ""
        w = max(1, int(100 * p))
        rows.append(f'<div class="pbar-row"><div class="pbar-lbl{c}">{html.escape(lbl)}</div>'
                    f'<div class="pbar-track"><div class="pbar-fill{c}" style="width:{w}%"></div></div>'
                    f'<div class="pbar-num">{p:.3f}</div></div>')
    return _STYLE + '<div class="bv probs">' + "".join(rows) + "</div>"


def key_event_choices(steps: list[dict]):
    """PURE: (label, index) list of notable steps for the jump dropdown."""
    out = []
    for j, s in enumerate(steps):
        ph = s.get("phase")
        if ph == "WON":
            out.append((f"step {j}: WON", j))
        elif ph == "LOST":
            out.append((f"step {j}: LOST", j))
        elif s.get("score") is not None and (s.get("round_score", 0) + (s.get("score") or 0)
                                             >= s.get("required", 1 << 30)):
            out.append((f"step {j}: blind cleared (ante {s['ante']})", j))
        elif s.get("verb") == "BUY":
            out.append((f"step {j}: {s.get('action_label', 'BUY')}", j))
    return out


def render_step(step_index: int, steps: list[dict]):
    """Back-compat PURE entrypoint: (focus_html, reel_html, probs_html)."""
    return (render_focus(step_index, steps),
            build_reel(step_index, steps),
            build_probs_html(step_index, steps))


# ----------------------------------------------------------------------------- #
# Gradio wiring (thin; all nav funnels through the pure renderers above).
# ----------------------------------------------------------------------------- #
def _clamp(i, n):
    return max(0, min(int(i), n - 1)) if n else 0


def _render_all(i, steps):
    return render_focus(i, steps), build_reel(i, steps), build_probs_html(i, steps)


def parse_file(filepath, _state):
    import gradio as gr
    with open(filepath) as f:
        steps = json.load(f)
    n = len(steps)
    focus, reel, probs = _render_all(0, steps)
    slider = gr.Slider(minimum=0, maximum=max(n - 1, 0), step=1, value=0,
                       interactive=True, label=f"Step (0–{max(n - 1, 0)})")
    jump = gr.Dropdown(choices=key_event_choices(steps), value=None, label="jump to key event")
    return steps, 0, slider, jump, focus, reel, probs


def load_selected(path, state):
    import gradio as gr
    if not path:
        return (state, 0, gr.Slider(), gr.Dropdown(choices=[]),
                render_focus(0, []), build_reel(0, []), "")
    return parse_file(path, state)


def refresh_episode_list():
    import gradio as gr
    return gr.Dropdown(choices=_choices())


def _go(new_i, steps):
    import gradio as gr
    i = _clamp(new_i, len(steps))
    focus, reel, probs = _render_all(i, steps)
    return i, gr.Slider(value=i), focus, reel, probs


def go_prev(cur, steps):
    return _go(cur - 1, steps)


def go_next(cur, steps):
    return _go(cur + 1, steps)


def go_slider(val, steps):
    i = _clamp(val, len(steps))
    focus, reel, probs = _render_all(i, steps)
    return i, focus, reel, probs


def go_jump(choice, cur, steps):
    target = cur if choice is None else int(choice)
    return _go(target, steps)


_KEYS_JS = """() => { document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  document.getElementById('bv-prev')?.click();
  if (e.key === 'ArrowRight') document.getElementById('bv-next')?.click();
}); }"""


def build_demo():
    import gradio as gr
    with gr.Blocks(title="Balatro RL — Replay") as demo:
        gr.Markdown("# Balatro RL — Replay Viewer")
        gr.Markdown(f"Pick a recorded episode from `{EPISODE_DIR}` (refresh after new runs) "
                    "or upload a `.json`. Old episodes without structured hands fall back to text.")
        traj = gr.State(value=[])
        cur_idx = gr.State(value=0)
        with gr.Row():
            picker = gr.Dropdown(choices=_choices(), value=None, scale=4, label="recorded episode")
            refresh = gr.Button("🔄 refresh", scale=1)
            upload = gr.UploadButton("⬆ upload .json", file_types=[".json"],
                                     file_count="single", scale=1)
        reel = gr.HTML(label="timeline")
        with gr.Row():
            prev_b = gr.Button("◀ Prev", elem_id="bv-prev", scale=1)
            slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="Step",
                               interactive=True, scale=4)
            next_b = gr.Button("Next ▶", elem_id="bv-next", scale=1)
            jump = gr.Dropdown(choices=[], value=None, label="jump to key event", scale=3)
        focus = gr.HTML(label="step")
        with gr.Accordion("agent internals (policy / value)", open=False):
            probs = gr.HTML(label="policy (top actions)")

        load_outs = [traj, cur_idx, slider, jump, focus, reel, probs]
        picker.change(load_selected, inputs=[picker, traj], outputs=load_outs)
        upload.upload(parse_file, inputs=[upload, traj], outputs=load_outs)
        refresh.click(refresh_episode_list, outputs=[picker])

        nav_outs = [cur_idx, slider, focus, reel, probs]
        prev_b.click(go_prev, inputs=[cur_idx, traj], outputs=nav_outs)
        next_b.click(go_next, inputs=[cur_idx, traj], outputs=nav_outs)
        jump.change(go_jump, inputs=[jump, cur_idx, traj], outputs=nav_outs)
        slider.release(go_slider, inputs=[slider, traj], outputs=[cur_idx, focus, reel, probs])

        demo.load(None, None, None, js=_KEYS_JS)   # optional keyboard sugar over the buttons
    return demo


def main():
    build_demo().launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    main()

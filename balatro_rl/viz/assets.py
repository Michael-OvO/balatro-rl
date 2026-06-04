"""Resolve our game content (cards / jokers / enhancements / seals / editions / bosses /
planets) to embeddable image data URIs from the locally-fetched wiki art cache (see
fetch_wiki_assets.py). Returns None when an asset is absent so the viewer falls back to its
CSS tiles -- so the viewer works with or without the cache, and CI (no cache) is unaffected.

Images are returned as base64 `data:` URIs (cached in memory): self-contained, so they work
both in the Gradio app and in standalone HTML previews with no static-file serving config.
"""
from __future__ import annotations

import base64
import functools
import json
import os

_DIR = os.path.join(os.path.dirname(__file__), "assets", "wiki")
_MANIFEST_PATH = os.path.join(_DIR, "manifest.json")


@functools.lru_cache(maxsize=1)
def _manifest() -> dict:
    try:
        with open(_MANIFEST_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


@functools.lru_cache(maxsize=1024)
def _uri(key: str) -> str | None:
    fname = _manifest().get(key)
    if not fname:
        return None
    try:
        with open(os.path.join(_DIR, fname), "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


def available() -> bool:
    """True if the local art cache is populated (run fetch_wiki_assets to populate)."""
    return bool(_manifest())


def card(rank: int, suit: int) -> str | None:
    return _uri(f"card:{int(rank)}:{int(suit)}")


def joker(type_id: int) -> str | None:
    return _uri(f"joker:{int(type_id)}")


def enhancement(enh: int) -> str | None:
    return _uri(f"enh:{int(enh)}") if enh else None


def seal(s: int) -> str | None:
    return _uri(f"seal:{int(s)}") if s else None


def edition(ed: int) -> str | None:
    return _uri(f"ed:{int(ed)}") if ed else None


def boss(boss_id: int) -> str | None:
    return _uri(f"boss:{int(boss_id)}") if boss_id else None


def planet(type_id: int) -> str | None:
    return _uri(f"planet:{int(type_id)}")


def consumable(kind: int, type_id: int) -> str | None:
    from ..engine.consumables import ConsumableKind
    return planet(type_id) if kind == ConsumableKind.PLANET else None

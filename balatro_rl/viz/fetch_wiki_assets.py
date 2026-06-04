"""Download the Balatro art assets we map to (cards / jokers / enhancements / seals /
editions / bosses / planets) from the fan wiki (balatrowiki.org, a MediaWiki) into a LOCAL,
GIT-IGNORED cache, plus a manifest the viewer reads. The repo ships this script + the
integration code, NOT the art -- the assets are copyrighted (c LocalThunk / Playstack), so
they stay out of the public repo; run this once locally to populate the cache.

    python -m balatro_rl.viz.fetch_wiki_assets

Filenames are resolved by probing the MediaWiki API (which File:<Name>.png exist), so we
never guess wrong: joker names come from each joker's `# wiki: /w/<Page>` link in library.py
(e.g. Ride_the_Bus, not Ride_The_Bus), with title-cased fallbacks; other categories use
known name patterns + candidates. Missing assets are simply skipped (the viewer falls back
to its CSS tiles).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

from ..engine.bosses import BossEffect
from ..engine.cards import Edition, Enhancement, Seal
from ..engine.consumables import PlanetType
from ..engine.jokers.base import JokerType, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401  (populate REGISTRY)

API = "https://balatrowiki.org/api.php"
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "wiki")
MANIFEST = os.path.join(ASSET_DIR, "manifest.json")
_UA = "balatro-rl-viewer-asset-fetch (personal research; contact via github Michael-OvO/balatro-rl)"

_RANK_NAME = {**{n: str(n) for n in range(2, 11)}, 11: "Jack", 12: "Queen", 13: "King", 14: "Ace"}
_SUIT_NAME = {0: "Spades", 1: "Hearts", 2: "Clubs", 3: "Diamonds"}


def _titlecase(enum_name: str) -> str:
    """GREEDY -> Greedy, SCARY_FACE -> Scary_Face (small words lowercased, MediaWiki-style)."""
    small = {"the", "of", "in", "to", "a", "and"}
    parts = enum_name.lower().split("_")
    return "_".join(p if (i and p in small) else p.capitalize() for i, p in enumerate(parts))


def _joker_wiki_pages() -> dict[int, str]:
    """Map JokerType id -> exact wiki page name from each `# wiki: /w/<Page>` link in library.py."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "engine", "jokers", "library.py")).read()
    out: dict[int, str] = {}
    for m in re.finditer(r"@register\(JokerType\.(\w+)\).*?#\s*wiki:\s*/w/(\S+)", src, re.S):
        name, page = m.group(1), m.group(2)
        if name in JokerType.__members__:
            out[int(JokerType[name])] = page
    return out


def _candidates() -> dict[str, list[str]]:
    """asset_key -> ordered candidate File base names (no extension)."""
    cand: dict[str, list[str]] = {}
    for rank, rn in _RANK_NAME.items():
        for suit, sn in _SUIT_NAME.items():
            cand[f"card:{rank}:{suit}"] = [f"{rn}_of_{sn}"]
    pages = _joker_wiki_pages()
    for jt in REGISTRY:
        jid = int(jt)
        tc = _titlecase(jt.name)
        # wiki page name (exact) first, then title-cased fallbacks; de-duped, order-preserving.
        cand[f"joker:{jid}"] = list(dict.fromkeys(
            p for p in (pages.get(jid), tc, f"{tc}_Joker") if p))
    for e in Enhancement:
        if e != Enhancement.NONE:
            cand[f"enh:{int(e)}"] = [f"{_titlecase(e.name)}_Card"]
    for s in Seal:
        if s != Seal.NONE:
            cand[f"seal:{int(s)}"] = [f"{_titlecase(s.name)}_Seal"]
    cand[f"ed:{int(Edition.FOIL)}"] = ["Foil"]
    cand[f"ed:{int(Edition.HOLO)}"] = ["Holographic", "Holo"]
    cand[f"ed:{int(Edition.POLY)}"] = ["Polychrome", "Poly"]
    for b in BossEffect:
        if b != BossEffect.NONE:
            cand[f"boss:{int(b)}"] = [_titlecase(b.name)]
    for p in PlanetType:
        cand[f"planet:{int(p)}"] = [_titlecase(p.name)]
    return cand


def _api_resolve(filebases: list[str]) -> dict[str, str]:
    """File base name -> direct image URL, for those that exist (batched imageinfo query)."""
    urls: dict[str, str] = {}
    uniq = sorted(set(filebases))
    for i in range(0, len(uniq), 40):
        batch = uniq[i:i + 40]
        titles = "|".join(f"File:{b}.png" for b in batch)
        q = urllib.parse.urlencode({"action": "query", "prop": "imageinfo",
                                    "iiprop": "url", "titles": titles, "format": "json"})
        req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": _UA})
        data = json.load(urllib.request.urlopen(req, timeout=30))
        for p in data.get("query", {}).get("pages", {}).values():
            if "imageinfo" in p:
                # MediaWiki normalizes underscores to spaces in titles -> map back to match candidates.
                base = p["title"].split(":", 1)[1].rsplit(".", 1)[0].replace(" ", "_")
                urls[base] = p["imageinfo"][0]["url"].split("?")[0]
        time.sleep(0.3)
    return urls


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)
    cand = _candidates()
    all_bases = [b for cs in cand.values() for b in cs]
    print(f"resolving {len(set(all_bases))} candidate filenames via the wiki API...")
    url_for = _api_resolve(all_bases)

    manifest: dict[str, str] = {}
    missing: list[str] = []
    for key, cs in cand.items():
        base = next((b for b in cs if b in url_for), None)
        if base is None:
            missing.append(key)
            continue
        fname = f"{base}.png"
        dest = os.path.join(ASSET_DIR, fname)
        if not os.path.exists(dest):
            req = urllib.request.Request(url_for[base], headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
                f.write(r.read())
            time.sleep(0.15)
        manifest[key] = fname
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=0, sort_keys=True)
    print(f"downloaded/cached {len(manifest)} assets -> {ASSET_DIR}")
    if missing:
        print(f"{len(missing)} not found on the wiki (viewer falls back to CSS): "
              + ", ".join(sorted(missing)[:20]) + (" ..." if len(missing) > 20 else ""))


if __name__ == "__main__":
    main()

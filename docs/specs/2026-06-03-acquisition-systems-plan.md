# Acquisition / Deckbuilding Systems Plan

**Goal:** make the environment a faithful *full* Balatro by adding the **acquisition meta** —
the systems through which the agent obtains and shapes its deck/jokers/consumables. The
mechanics-vs-wiki audit confirmed the existing engine is numerically correct but missing
these systems entirely (shop offers only jokers; consumables can be USEd but never obtained;
no packs, no vouchers). Without them the agent can buy jokers but can't *build a deck*, which
caps strategic depth (the original agent topped out at ante 7).

**Strategy: ENGINE-FIRST, ONE RETRAIN LAST** — same approach that worked for Phases B/C/D.
Build each system in the engine with the agent blind (tested via direct engine calls, obs
unchanged), then do **one** coordinated obs/action widening + the (GPU) retrain at the end.

## Phases

### E1 — Shop generalization + consumable offers
- A typed `ShopItem(kind, payload, cost)` where kind ∈ {JOKER, TAROT, PLANET, SPECTRAL}.
  `generate_offers` rolls each card slot's KIND by the wiki composition (~Joker 71.4% /
  Tarot 14.3% / Planet 14.3% at base; Spectral only from the Ghost voucher), then the
  specific item. Consumables cost **$3** in the shop (verify on wiki).
- `_shop_step` BUY: joker → `jokers`; consumable → `consumables` (respect `consumable_slots`).
- Obs stays joker-only for now (encoder ignores non-joker offers) — agent blind until E5.

### E2 — Tarot cards (+ the USE-with-targets two-step)
- Implement the 22 Tarots (`consumables.py` `apply_consumable` for TAROT): enhance N selected
  cards (The Magician→Lucky, Empress→Mult, Hierophant→Bonus, Hermit→money, Sun/Moon/Star/
  World→suit conversion, Death→copy, Strength→rank+1, etc.), wiki-verified.
- USE-with-targets: a `pending_use` two-step (USE selects the consumable, then the agent
  selects target cards) — engine supports `step((USE, i), target_idx)` for now; the agent
  wiring (action) comes in E5.

### E3 — Booster packs
- `generate_offers` adds **2 pack slots** (Arcana/Celestial/Standard/Buffoon/Spectral, sizes
  Normal/Jumbo/Mega). BUY a pack → enter an OPEN sub-phase: pick K of M revealed cards
  (K=1-2, M=3-5 by size), add picks to deck/consumables/jokers, then resume the shop.

### E4 — Vouchers
- A voucher slot in the shop; `vouchers.py` with the ~32 vouchers + effects: Overstock
  (+1 card slot), Overstock Plus (+1 more), Seed Money/Money Tree (interest cap → $10/$20),
  Grabber/Nacho Tong (+hands), Wasteful/Recyclomancy (+discards), Hone/Glow Up (edition
  rates), Reroll Surplus/Glut (reroll cost), Crystal Ball (+consumable slot), Telescope/
  Observatory (planets), etc. Persistent per-run modifiers on `GameState`.

### E5 — THE RETRAIN: coordinated obs/action widening + GPU
- One obs revision: shop offer KIND + payload encoding, pack-open stream, owned vouchers,
  consumable-target selection (reuse the play-subset selection). Network: extend the shop
  head to score buy-consumable/buy-pack/buy-voucher; add open-pack + use-with-targets heads.
- Retrain on the full env (GPU). Pause for user confirmation before the training run.

## Out of scope (deferred, per the audit)
- Endless mode (float-precision scoring above 2^53), deck/stake selection (single Red/White
  is a legitimate scope), the 4 finisher-boss effects, Spectral cards initially (E-later),
  blind-skip + Tags (medium), the remaining jokers toward the full roster, Negative edition
  / extra slots.

## Notes
- Verify every value against balatrowiki.org before implementing (shop composition rates,
  consumable costs, Tarot effects, pack sizes/contents, voucher effects).
- Keep all existing tests green; the engine stays correct throughout (the agent is blind to
  each new system until E5, so no obs/training breakage mid-build).

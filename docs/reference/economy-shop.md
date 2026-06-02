# Balatro Economy & Shop Mechanics — Implementation Reference

All values verified against [balatrowiki.org](https://balatrowiki.org) via a fan-out + adversarial-verify research pass (corrections applied; see "OPEN / UNCERTAIN").

---

## 1. Starting money
- Most decks: **$4** ([/w/Money](https://balatrowiki.org/w/Money)). Yellow Deck: **$14** ($4 + $10).

## 2. End-of-round payout (cash-out)
Cash-out = blind reward + interest + leftover-hand money (+ leftover-discard money on some decks) + tag/joker payouts.

**Blind clear reward** ([/w/Blinds_and_Antes](https://balatrowiki.org/w/Blinds_and_Antes)):
| Blind | Reward |
|---|---|
| Small | **$3** |
| Big | **$4** |
| Boss (regular) | **$5** |
| Showdown/Finisher (Ante 8+) | **$8** |

- Red Stake+: Small Blind reward = **$0** (others unaffected).

**Leftover hands / discards** ([/w/Money](https://balatrowiki.org/w/Money)):
- Standard decks: **$1 per unused hand**, $0 per unused discard.
- Green Deck: $2/hand, $1/discard, **no interest**.
- The Omelette: no leftover money AND all blinds give no reward; Mad World: no leftover money + no interest (blind rewards normal).

**Interest** ([/w/Interest](https://balatrowiki.org/w/Interest)):
- **+$1 per $5 held**, default cap **$5/round** (at $25 held). `interest = min(money // 5, cap)`.
- Cap raised: Seed Money → $10, Money Tree → $20. `To the Moon` joker adds a *separate* +$1/$5 (don't fold into base). No interest on Green/Omelette/Mad World decks.

## 3. Round → cash-out → shop → next-blind flow
Each ante = Small → Big → Boss. Small/Big skippable (claim a **Tag**); Boss mandatory.
1. Blind-select: PLAY, or (Small/Big) SKIP → grants a Tag, no round/shop, advance.
2. Play hands to reach the score target.
3. Clear → cash-out (reward + interest + leftover hands).
4. Shop (buy/sell/reroll).
5. Leave shop → next blind's select screen. Beating the Boss advances to Ante N+1.
- Voucher slot restocks only after defeating a Boss (once per ante). First shop of a run guarantees one Normal Buffoon Pack.

## 4. Shop slot layout ([/w/The_Shop](https://balatrowiki.org/w/The_Shop))
| Slot | Default | Expansion |
|---|---|---|
| Card slots | **2** | Overstock → 3, Overstock Plus → 4 |
| Booster-pack slots | **2** | (fixed) |
| Voucher slot | **1** | (restocks after each Boss) |

**Card-slot pool weights:** Joker **20** (~71.4%), Tarot **4** (~14.3%), Planet **4** (~14.3%); Playing card **4** only with Magic Trick voucher; Spectral **2** only with Ghost Deck. When a Joker rolls, rarity = **Common 70% / Uncommon 25% / Rare 5%** (Legendary never in shop — Soul only).

## 5. Buy prices ([/w/The_Shop](https://balatrowiki.org/w/The_Shop))
- Jokers: Common $1–6, Uncommon $4–8, Rare $7–10, Legendary $20. *(Per-joker base costs are in `docs/reference/jokers.md`; use those as deterministic prices.)*
- Playing card $1; Tarot $3; Planet $3; Spectral $4; Voucher $10.
- Edition surcharge (joker & card): Foil +$2, Holographic +$3, Polychrome +$5, Negative +$5 (jokers).
- Discount vouchers (multiply all shop prices incl. vouchers): Clearance Sale ×0.75, Liquidation ×0.5.

## 6. Reroll & sell ([/w/The_Shop](https://balatrowiki.org/w/The_Shop), [/w/Reroll](https://balatrowiki.org/w/Reroll))
**Reroll:** base **$5**, **+$1 each reroll**, **resets to $5** each new shop. Refreshes only card slots (not packs/voucher). `reroll_cost = max(0, base + rerolls_done)`, base ∈ {5, 3 (Reroll Surplus), 1 (Reroll Glut)}. Chaos the Clown = 1 free reroll/shop; D6 Tag → next shop starts rerolls at $0.

**Sell value:** `sell = max(1, floor(buy_cost / 2)) + egg_giftcard_bonus`. Egg: +$3 sell value at end of each round (accumulates). Gift Card: +$1 sell value to every joker & consumable at end of each round.

## 7. Vouchers ([/w/Vouchers](https://balatrowiki.org/w/Vouchers)) — base $10, upgrade a separate $10 (needs base)
Overstock/Overstock Plus (+1/+1 card slot); Clearance Sale/Liquidation (25%/50% off); Reroll Surplus/Glut (−$2/−$2 reroll base); Seed Money/Money Tree (interest cap $10/$20); Tarot/Planet Merchant→Tycoon (2×/4× appearance); Hone/Glow Up (edition rate 2×/4×); Director's Cut/Retcon (reroll boss 1×/unlimited, $10); Magic Trick/Illusion (playing cards in shop / with mods). Resource vouchers: Crystal Ball/Omen Globe (+consumable slot), Grabber/Nacho Tong (+hand), Wasteful/Recyclomancy (+discard), Telescope/Observatory, Hieroglyph/Petroglyph (−ante, −hand/discard), Antimatter (+1 joker slot), Paint Brush/Palette (+hand size).

## 8. Booster packs ([/w/Booster_Packs](https://balatrowiki.org/w/Booster_Packs))
Types: Arcana (tarot), Celestial (planet), Standard (playing cards), Buffoon (jokers), Spectral. Prices by **tier only**: Normal **$4**, Jumbo **$6**, Mega **$8**.
| Tier | Cost | Standard/Arcana/Celestial (shown/pick) | Buffoon & Spectral (shown/pick) |
|---|---|---|---|
| Normal | $4 | 3 / 1 | 2 / 1 |
| Jumbo | $6 | 5 / 1 | 4 / 1 |
| Mega | $8 | 5 / 2 | 4 / 2 |
2 packs always offered; don't refresh on reroll; first shop guarantees a Normal Buffoon Pack.

## 9. Economy-relevant skip tags ([/w/Tags](https://balatrowiki.org/w/Tags))
Granted only by skipping Small/Big. Investment ($25 after next boss); Economy (double current money, max +$40); Handy ($1/hand played this run); Garbage ($1/unused discard this run); Speed ($5/blind skipped this run); Coupon (next shop's initial cards & packs free); D6 (next shop rerolls start $0).

## OPEN / UNCERTAIN (verifier-flagged)
1. Liquidation requiring Clearance Sale first is the standard tier structure but not stated verbatim (50%/$10 confirmed).
2. D6 Tag "$0 then +$1" scaling inferred from normal reroll mechanics.
3. Economy/Speed tag exact trigger-timing wording inferred.
4. "Spectral is rarest" loosely worded (lowest per-size weights; Buffoon Mega 0.15 also extremely rare).
5. Pack prices are by tier only ($4/$6/$8), uniform across all 5 types — no per-type price differences.

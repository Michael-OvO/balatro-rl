# Game Subsystems Build Plan (enhancements / bosses / consumables)

**Goal:** unblock the remaining subsystem-gated jokers (~42 of them, 62 → ~104) by building
the card-enhancement, boss-blind, and consumable subsystems.

**Chosen strategy: MECHANICS FIRST, RETRAIN LAST.** Build the engine mechanics with **no
obs/action change** (the validated ante-7 agent keeps running, blind to the new features),
then do **exactly one** coordinated obs/action widening + retrain at the end — which also
brings consumables online. This avoids the 3-retrain trap (enhancements + bosses both widen
the same per-card net kernel; all three touch the global vector).

Full design + critique: workflow `wf_0e0732b9-3b7`.

## Status

- **Phase A (P0 backbone)** — DONE (`dbcfed9`).
- **Phase B (enhancement/edition/seal scoring + jokers)** — DONE. B1 scoring (`903bca0`);
  B2a deck-reading/economy jokers + Steel/Stone Joker, Golden Ticket, Rough Gem, Business
  Card, Reserved Parking (`2522ef6`); B2b-i Glass Joker + Lucky Cat via HandEvents/
  on_hand_events (`895af0f`); B2b-ii Vampire + Midas Mask via the master-deck mutation
  channel (`49a2e48`). **80 jokers.**
- **Phase C (boss blinds, engine-only, gated `enable_bosses` default OFF)** — DONE.
  C0 backbone+selection+req-mult (`3ed7bb7`); C1 scoring debuffs + The Flint (`bad0c90`);
  C2 legal-mask + blind-setup (Water/Manacle/Needle/Psychic/Eye/Mouth) (`30c5441`);
  C3 draw/state (Hook/Serpent/Tooth/Ox) (`036ca73`). Debuff semantics corrected to
  wiki-true (debuffed card fully inert). **593 tests, zero regressions, all byte-compatible**
  — the validated ante-7 agent still runs unchanged on its checkpoint.
- **Phase D (THE RETRAIN)** — NOT STARTED. Pauses for user confirmation before training.

Deferred into Phase D (need agent obs / levels to matter): face-down bosses (House/Wheel/
Fish/Mark), The Pillar (cross-blind tracking), The Arm (level scoring), finisher effects
(Amber Acorn/Verdant Leaf/Crimson Heart/Cerulean Bell — Violet Vessel's 6x req is done),
BLUE/PURPLE seals (need consumables).

## Build order

### Phase A — P0 backbone (no agent impact, byte-compatible with the ante-7 checkpoint)
- `GameState.master_deck: tuple[Card,...]` = the 52 owned cards w/ their mod fields. `reset()`
  seeds it from `standard_deck()`; `reset()`/`_advance_blind()` reshuffle the working deck
  **from `master_deck`** instead of rebuilding `standard_deck()` (engine.py ~67,118). `_draw`
  already carries Card POD, so mods ride along.
- `ScoreResult.money_delta:int=0` and `destroyed_idx:tuple[int,...]=()` — thread money
  (Lucky/Gold-seal/Gold-enh) and card destruction (Glass) out of the pure scoring fold;
  `engine.step` applies them (`money += res.money_delta`; remove `destroyed_idx` from
  master_deck) — mirrors how `res.rng` is already written back.
- Relax `actions.py:86` `assert len(hand) <= MAX_HAND` → clamp/mask absent slots (so later
  hand-size effects don't trip it).
- All scores unchanged (mods all 0) → existing tests + ante-7 agent byte-identical.

### Phase B — Enhancement / Edition / Seal SCORING (engine-only, agent blind), +22 jokers
- `cards.py`: IntEnums `Enhancement(NONE,BONUS,MULT,WILD,GLASS,STEEL,GOLD,LUCKY,STONE)`,
  `Edition(NONE,FOIL,HOLO,POLY,NEGATIVE)`, `Seal(NONE,GOLD,RED,BLUE,PURPLE)` + `is_stone`,
  `scores_as_suit`.
- `score_play`: per-scored-card BONUS +30c, MULT +4m, GLASS ×2m (+1-in-4 shatter → destroyed_idx,
  rng via `roll,ctx.rng=ctx.rng.random()`), STONE +50c & force into scoring_idx (like Splash),
  LUCKY 1-in-5 +20m / 1-in-15 +$20 (→ money_delta); editions FOIL +50c / HOLO +10m / POLY ×1.5;
  held STEEL ×1.5m (on_held phase); RED seal +1 retrigger; GOLD seal +$3 on score. GOLD
  enhancement +$3/held at round end (`_cash_out`). Wild (any suit) + Stone (no rank/suit/face)
  in `hands.py` evaluate/contains/is_face.
- **Compose with bosses now:** a `debuffed_idx` card nullifies its own enhancement/edition/seal —
  own this skip in `score_play` so Phase C just supplies `debuffed_idx`.
- Acquired via **pre-seeded decks** (reset builds a master_deck with non-zero mods) — no
  consumables/packs dependency; self-validating. BLUE/PURPLE seals (→ create consumable) deferred
  to the consumables phase. **No obs change** (agent doesn't see mods yet — that's the final retrain).
- Verify every enhancement/edition/seal value against balatrowiki.org.

### Phase C — Boss blinds (engine-only, agent blind), +8 jokers
- `engine/bosses.py`: `BossEffect(IntEnum)` (NONE + ~23 standard; stub 5 finishers).
- `_advance_blind()` is the single selection point: when `blind_index==2`, roll a min-ante-eligible
  boss from `state.rng`; set required-mult override (Wall 4×/Needle 1×/Vessel 6× vs default 2.0).
- Three integration buckets: (1) **scoring** — compute `debuffed_idx` (suit bosses, The Plant=face,
  The Pillar) and skip chips/on_score/enhancement for those (Phase B's compose path); The Flint
  halves HAND_BASE; The Arm -1 level. (2) **legal-mask** — Water 0 discards, Manacle hand_size-1,
  Needle 1 hand, Psychic only size-5, Mouth/Eye filter by per-subset hand type (memoize!),
  Cerulean Bell forced index. (3) **draw/state** — Hook discards 2 held, Fish/Wheel/Mark/House
  face-down draws, Serpent draw-3. Matador/Luchador/Chicot set `boss_disabled`.
- DEFER finisher bosses (Amber Acorn/Crimson Heart/Verdant Leaf). **No obs change yet.**
- Gate boss selection behind a flag so exact ante-7 replays are preserved if needed.

### Phase D — THE RETRAIN: coordinated obs/action widening + consumables, +12 jokers
- **One** obs revision (everything that was invisible in B/C becomes visible, + consumables):
  - per-card `CARD_FEAT 17 → 38` (+enhancement 9, edition 5, seal 5, is_debuffed 1, is_face_down 1);
    rewrite `_card_vec`; obs hand `[8,38]`.
  - global: `+boss_onehot(~28)`, merge `GLOBAL_FEAT 16 → ~21` (boss_disabled, boss-mult,
    consumable_slots, len(consumables), pending_use).
  - consumable entity stream: `consum_types/consum_kind/consum_neg/consum_mask` (MAX_CONSUM=2).
  - `NUM_ACTIONS 465 → ~468` (MAX_CONSUM USE ids + 1 CANCEL), masked until used.
  - `networks.py`: input Dense fan-in 17→38 (auto); concat boss_onehot into g-block; `seg (3→4,d)`;
    separate `CONSUM_VOCAB` Embed + a 4th masked-pool consum_ctx stream; +3 logits + USE/CANCEL head.
- Consumables: GameState consumable slots; Tarot (enhance/transform), Planet (level a hand type via
  `levels[12]`), Spectral; shop-direct offers (packs out of scope). USE action targets selected
  cards via a `pending_use` two-step (reuse the play-subset selection for targets; strong
  pending_use feature + distinct seg tag so the net doesn't confuse play vs target).
- **Retrain** the card-aware agent on the new (wider) obs; re-validate ante-7 parity first on an
  all-mods-0 deck (zero/masked features must reproduce current behavior).

## Out of scope (flagged)
- Booster packs / vouchers subsystem (real in-run acquisition source for both enh + consumables).
- The 5 finisher bosses (reach into joker-provider resolution + sell path).
- These gate the remaining ~46 jokers (≈104 → 150).

## Key risks (from the critique)
- P0 is the un-owned shared backbone — ship it BEFORE any mod logic or mods are silently erased
  every blind and Glass/Lucky/Gold money is silently dropped (no error).
- Phase B must own the debuffed-card enhancement-skip so Phase C only supplies `debuffed_idx`.
- Relax the hand-size assert in P0 before any hand-size effect runs.
- Boss selection + Lucky/Glass rolls perturb the deterministic rng stream — gate boss selection
  behind a flag to preserve exact replays; document roll order vs Misprint/Bloodstone.
- Mouth/Eye bosses do per-subset `evaluate()` in `legal_mask` (hot loop) — memoize.

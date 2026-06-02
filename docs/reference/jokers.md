# Joker Reference (all 150)

Source: balatrowiki.org/w/Jokers (game v1.0.1o, 150 jokers). This is the canonical implementation checklist for the joker campaign.

## Type legend
`+c` chips · `+m` additive mult · `Xm` ×mult · `++` chips & mult · `!!` effect/utility · `...` retrigger · `+$` economy

## Activation legend
- **Indep.** — applies once during scoring, regardless of which cards scored (joker-slot order matters for +before-×).
- **On Scored** — per scoring card (cards that participate in the hand; debuffed/non-scoring cards don't trigger).
- **On Held** — per card held in hand (not played).
- **On Played** — when the hand is played (before/at scoring), independent of which cards score.
- **On Discard** — when cards are discarded.
- **On Blind Select** — when a blind is selected (not when skipped).
- **Mixed** — a scaling joker: a lifecycle event updates its internal counter, which then applies independently.
- **rule** — passive rule modifier consulted by hand detection / probabilities.
- **copy** — copies another joker's ability.
- **N/A** — passive/utility (hand size, discards, debt, economy at round end, etc.).

## Key terms (scoring semantics)
- **Contains vs Is**: "contains a Pair" = the hand includes that sub-hand (Three of a Kind contains a Pair; a 5-same-suit Four of a Kind contains a Flush). "is a Pair" = the whole hand classifies exactly as a Pair. Note: Four of a Kind does **not** contain Two Pair; Five of a Kind does **not** contain Full House.
- **Retrigger**: re-score a card, re-activating all on-score effects (other jokers, enhancements, editions, non-red seals, and the card's own chips). `Mime` retriggers only held-in-hand abilities (no chips/editions, since the card isn't scoring).
- **Debuffed**: joker (and its edition) can't trigger; keeps sell value; still targetable by random effects.
- **Create**: needs room (slot) for the created card, except `Perkeo` (negative consumables).

## Editions (on jokers)
Base · Foil (+50 Chips) · Holographic (+10 Mult) · Polychrome (×1.5 Mult) · Negative (+1 joker slot, takes no slot).
High-stake stickers: Eternal (can't sell/destroy), Perishable (debuffs after 5 rounds), Rental ($3/round).

## Rarities & generation
Common (61, 70%) · Uncommon (64, 25%) · Rare (20, 5%) · Legendary (5, only via The Soul). Default 5 joker slots.

---

## The 150

> Format: `N. Name — effect ($cost, Rarity, type, activation)[ · unlock: condition]`. All are "available from start" unless an unlock is noted. `(start: …)` shows a scaling joker's initial value.

1. **Joker** — +4 Mult ($2, Common, +m, Indep)
2. **Greedy Joker** — played ♦ cards give +3 Mult when scored ($5, Common, +m, On Scored)
3. **Lusty Joker** — played ♥ cards give +3 Mult when scored ($5, Common, +m, On Scored)
4. **Wrathful Joker** — played ♠ cards give +3 Mult when scored ($5, Common, +m, On Scored)
5. **Gluttonous Joker** — played ♣ cards give +3 Mult when scored ($5, Common, +m, On Scored)
6. **Jolly Joker** — +8 Mult if played hand contains a Pair ($3, Common, +m, Indep)
7. **Zany Joker** — +12 Mult if contains Three of a Kind ($4, Common, +m, Indep)
8. **Mad Joker** — +10 Mult if contains Two Pair ($4, Common, +m, Indep)
9. **Crazy Joker** — +12 Mult if contains a Straight ($4, Common, +m, Indep)
10. **Droll Joker** — +10 Mult if contains a Flush ($4, Common, +m, Indep)
11. **Sly Joker** — +50 Chips if contains a Pair ($3, Common, +c, Indep)
12. **Wily Joker** — +100 Chips if contains Three of a Kind ($4, Common, +c, Indep)
13. **Clever Joker** — +80 Chips if contains Two Pair ($4, Common, +c, Indep)
14. **Devious Joker** — +100 Chips if contains a Straight ($4, Common, +c, Indep)
15. **Crafty Joker** — +80 Chips if contains a Flush ($4, Common, +c, Indep)
16. **Half Joker** — +20 Mult if played hand has ≤3 cards ($5, Common, +m, Indep)
17. **Joker Stencil** — ×1 Mult per empty joker slot, including itself ($8, Uncommon, Xm, Indep)
18. **Four Fingers** — all Flushes and Straights can be made with 4 cards ($7, Uncommon, !!, rule)
19. **Mime** — retrigger all held-in-hand card abilities ($5, Uncommon, ..., On Held)
20. **Credit Card** — go up to −$20 in debt ($1, Common, +$, N/A)
21. **Ceremonial Dagger** — on blind select, destroy joker to the right, add 2× its sell value to this Mult (start: +0) ($6, Uncommon, +m, On Blind Select→Indep)
22. **Banner** — +30 Chips per remaining discard ($5, Common, +c, Indep)
23. **Mystic Summit** — +15 Mult when 0 discards remaining ($5, Common, +m, Indep)
24. **Marble Joker** — adds a Stone card to the deck on blind select ($6, Uncommon, !!, On Blind Select)
25. **Loyalty Card** — ×4 Mult every 6 hands played ($5, Uncommon, Xm, Indep)
26. **8 Ball** — 1 in 4 per played 8 to create a Tarot when scored (needs room) ($5, Common, !!, On Scored)
27. **Misprint** — +0–23 Mult (random each score) ($4, Common, +m, Indep)
28. **Dusk** — retrigger all played cards in the final hand of the round ($5, Uncommon, ..., On Scored)
29. **Raised Fist** — adds 2× rank of the lowest held card to Mult ($5, Common, +m, On Held)
30. **Chaos the Clown** — 1 free reroll per shop ($4, Common, !!, N/A)
31. **Fibonacci** — each played A, 2, 3, 5, or 8 gives +8 Mult when scored ($8, Uncommon, +m, On Scored)
32. **Steel Joker** — ×0.2 Mult per Steel card in full deck (start: ×1) ($7, Uncommon, Xm, Indep) · dep: enhancements
33. **Scary Face** — played face cards give +30 Chips when scored ($4, Common, +c, On Scored)
34. **Abstract Joker** — +3 Mult per joker card ($4, Common, +m, Indep)
35. **Delayed Gratification** — earn $2 per discard if no discards used by round end ($4, Common, +$, N/A)
36. **Hack** — retrigger each played 2, 3, 4, or 5 ($6, Uncommon, ..., On Scored)
37. **Pareidolia** — all cards are considered face cards ($5, Uncommon, !!, rule)
38. **Gros Michel** — +15 Mult; 1 in 6 destroyed at end of round ($5, Common, +m, Indep)
39. **Even Steven** — played even-rank cards give +4 Mult when scored ($4, Common, +m, On Scored)
40. **Odd Todd** — played odd-rank cards give +31 Chips when scored ($4, Common, +c, On Scored)
41. **Scholar** — played Aces give +20 Chips and +4 Mult when scored ($4, Common, ++, On Scored)
42. **Business Card** — played face cards 1 in 2 to give $2 when scored ($4, Common, +$, On Scored)
43. **Supernova** — adds number of times this poker hand has been played this run to Mult ($5, Common, +m, Indep)
44. **Ride the Bus** — +1 Mult per consecutive hand with no scoring face card; resets on a face (start: +0) ($6, Common, +m, Mixed)
45. **Space Joker** — 1 in 4 to upgrade the level of the played poker hand ($5, Uncommon, !!, On Played)
46. **Egg** — gains $3 sell value at end of round ($4, Common, +$, N/A)
47. **Burglar** — on blind select, +3 Hands and lose all discards ($6, Uncommon, !!, On Blind Select)
48. **Blackboard** — ×3 Mult if all held cards are ♠ or ♣ ($6, Uncommon, Xm, Indep/On Held)
49. **Runner** — gains +15 Chips if played hand contains a Straight (start: +0) ($5, Common, +c, Mixed)
50. **Ice Cream** — +100 Chips, −5 per hand played ($5, Common, +c, Indep)
51. **DNA** — if first hand of round is 1 card, add a permanent copy to deck and draw it ($8, Rare, !!, On Played)
52. **Splash** — every played card counts in scoring ($3, Common, !!, rule)
53. **Blue Joker** — +2 Chips per remaining card in deck ($5, Common, +c, Indep)
54. **Sixth Sense** — if first hand of round is a single 6, destroy it and create a Spectral (needs room) ($6, Uncommon, !!, On Played) · dep: consumables
55. **Constellation** — ×0.1 Mult per Planet card used (start: ×1) ($6, Uncommon, Xm, Indep) · dep: consumables
56. **Hiker** — every played card permanently gains +5 Chips when scored ($5, Uncommon, +c, On Scored)
57. **Faceless Joker** — earn $5 if ≥3 face cards discarded at once ($4, Common, +$, On Discard)
58. **Green Joker** — +1 Mult per hand played, −1 per discard (start: +0) ($4, Common, +m, Mixed)
59. **Superposition** — create a Tarot if hand contains an Ace and a Straight (needs room) ($4, Common, !!, Indep) · dep: consumables
60. **To Do List** — earn $4 if poker hand is [hand]; changes each round ($4, Common, +$, On Played)
61. **Cavendish** — ×3 Mult; 1 in 1000 destroyed at end of round ($4, Common, Xm, Indep)
62. **Card Sharp** — ×3 Mult if this poker hand was already played this round ($6, Uncommon, Xm, Indep)
63. **Red Card** — +3 Mult when any Booster Pack is skipped (start: +0) ($5, Common, +m, Indep) · dep: packs
64. **Madness** — on small/big blind select, ×0.5 Mult and destroy a random joker (start: ×1) ($7, Uncommon, Xm, Indep)
65. **Square Joker** — +4 Chips if played hand has exactly 4 cards (start: 0) ($4, Common, +c, Mixed)
66. **Séance** — if poker hand is a Straight Flush, create a random Spectral (needs room) ($6, Uncommon, !!, Indep) · dep: consumables
67. **Riff-Raff** — on blind select, create 2 Common Jokers (needs room) ($6, Common, !!, On Blind Select)
68. **Vampire** — ×0.1 Mult per scoring Enhanced card played, removes its enhancement (start: ×1) ($7, Uncommon, Xm, Mixed) · dep: enhancements
69. **Shortcut** — Straights can be made with gaps of 1 rank (e.g. 10 8 6 5 3) ($7, Uncommon, !!, rule)
70. **Hologram** — ×0.25 Mult each time a playing card is added to your deck (start: ×1) ($7, Uncommon, Xm, Indep)
71. **Vagabond** — create a Tarot if hand is played with ≤$4 ($8, Rare, !!, Indep) · dep: consumables
72. **Baron** — each King held in hand gives ×1.5 Mult ($8, Rare, Xm, On Held)
73. **Cloud 9** — earn $1 per 9 in full deck at end of round ($7, Uncommon, +$, N/A)
74. **Rocket** — earn $1 at end of round; payout +$2 when Boss defeated ($6, Uncommon, +$, N/A) · dep: bosses
75. **Obelisk** — ×0.2 Mult per consecutive hand without playing your most-played hand (start: ×1) ($8, Rare, Xm, Mixed)
76. **Midas Mask** — all played face cards become Gold cards when scored ($7, Uncommon, !!, On Played) · dep: enhancements
77. **Luchador** — sell to disable the current Boss Blind ($5, Uncommon, !!, N/A) · dep: bosses
78. **Photograph** — first played face card gives ×2 Mult when scored ($5, Common, Xm, On Scored)
79. **Gift Card** — +$1 sell value to every joker and consumable at end of round ($6, Uncommon, +$, N/A)
80. **Turtle Bean** — +5 hand size, reduces by 1 each round ($6, Uncommon, !!, N/A)
81. **Erosion** — +4 Mult per card below the deck's starting size in your full deck ($6, Uncommon, +m, Indep)
82. **Reserved Parking** — each held face card 1 in 2 to give $1 ($6, Common, +$, On Held)
83. **Mail-In Rebate** — earn $5 per discarded [rank]; changes each round ($4, Common, +$, On Discard)
84. **To the Moon** — +$1 interest per $5 held at end of round ($5, Uncommon, +$, N/A)
85. **Hallucination** — 1 in 2 to create a Tarot when any Booster Pack is opened (needs room) ($4, Common, !!, N/A) · dep: packs+consumables
86. **Fortune Teller** — +1 Mult per Tarot card used this run (start: +0) ($6, Common, +m, Indep) · dep: consumables
87. **Juggler** — +1 hand size ($4, Common, !!, N/A)
88. **Drunkard** — +1 discard each round ($4, Common, !!, N/A)
89. **Stone Joker** — +25 Chips per Stone card in full deck ($6, Uncommon, +c, Indep) · dep: enhancements
90. **Golden Joker** — earn $4 at end of round ($6, Common, +$, N/A)
91. **Lucky Cat** — ×0.25 Mult each time a Lucky card triggers (start: ×1) ($6, Uncommon, Xm, Mixed) · dep: enhancements
92. **Baseball Card** — Uncommon jokers each give ×1.5 Mult ($8, Rare, Xm, On Other Jokers)
93. **Bull** — +2 Chips per $1 you have ($6, Uncommon, +c, Indep)
94. **Diet Cola** — sell to create a free Double Tag ($6, Uncommon, !!, N/A) · dep: tags
95. **Trading Card** — if first discard of round is 1 card, destroy it and earn $3 ($6, Uncommon, +$, On Discard)
96. **Flash Card** — +2 Mult per shop reroll (start: +0) ($5, Uncommon, +m, Indep) · dep: shop
97. **Popcorn** — +20 Mult, −4 per round played ($5, Common, +m, Indep)
98. **Spare Trousers** — +2 Mult if played hand contains Two Pair (start: +0) ($6, Uncommon, +m, Mixed)
99. **Ancient Joker** — each played [suit] gives ×1.5 Mult; suit changes each round ($8, Rare, Xm, On Scored)
100. **Ramen** — ×2 Mult, −×0.01 per card discarded ($6, Uncommon, Xm, Mixed)
101. **Walkie Talkie** — each played 10 or 4 gives +10 Chips and +4 Mult when scored ($4, Common, ++, On Scored)
102. **Seltzer** — retrigger all played cards for the next 10 hands ($6, Uncommon, ..., On Scored)
103. **Castle** — +3 Chips per discarded [suit] card; suit changes each round (start: +0) ($6, Uncommon, +c, Mixed)
104. **Smiley Face** — played face cards give +5 Mult when scored ($4, Common, +m, On Scored)
105. **Campfire** — ×0.25 Mult per card sold, resets when Boss defeated (start: ×1) ($9, Rare, Xm, Indep) · dep: shop+bosses
106. **Golden Ticket** — played Gold cards earn $4 when scored ($5, Common, +$, On Scored) · unlock: play a 5-card hand of only Gold cards · dep: enhancements
107. **Mr. Bones** — prevents death if chips ≥25% of required; self-destructs ($5, Uncommon, !!, N/A) · unlock: lose 5 runs
108. **Acrobat** — ×3 Mult on final hand of round ($6, Uncommon, Xm, Indep) · unlock: play 200 hands
109. **Sock and Buskin** — retrigger all played face cards ($6, Uncommon, ..., On Scored) · unlock: play 300 face cards
110. **Swashbuckler** — adds sell value of all other owned jokers to Mult ($4, Common, +m, Indep) · unlock: sell 20 jokers
111. **Troubadour** — +2 hand size, −1 hand each round ($6, Uncommon, !!, N/A) · unlock: win 5 consecutive rounds playing only 1 hand
112. **Certificate** — on round begin, add a random card with a random seal to hand ($6, Uncommon, !!, N/A) · unlock: gold card w/ gold seal · dep: seals
113. **Smeared Joker** — ♥/♦ count as the same suit, ♠/♣ count as the same suit ($7, Uncommon, !!, rule) · unlock: 3+ Wild cards
114. **Throwback** — ×0.25 Mult per blind skipped this run (start: ×1) ($6, Uncommon, Xm, Indep)
115. **Hanging Chad** — retrigger the first played scoring card 2 additional times ($4, Common, ..., On Scored) · unlock: beat a Boss with a High Card
116. **Rough Gem** — played ♦ cards earn $1 when scored ($7, Uncommon, +$, On Scored) · unlock: 30 ♦ cards in deck
117. **Bloodstone** — 1 in 2 for played ♥ cards to give ×1.5 Mult ($7, Uncommon, Xm, On Scored) · unlock: 30 ♥ cards
118. **Arrowhead** — played ♠ cards give +50 Chips when scored ($7, Uncommon, +c, On Scored) · unlock: 30 ♠ cards
119. **Onyx Agate** — played ♣ cards give +7 Mult when scored ($7, Uncommon, +m, On Scored) · unlock: 30 ♣ cards
120. **Glass Joker** — ×0.75 Mult per Glass card destroyed (start: ×1) ($6, Uncommon, Xm, Indep) · unlock: 5 Glass cards · dep: enhancements
121. **Showman** — Joker/Tarot/Planet/Spectral may appear multiple times ($5, Uncommon, !!, N/A) · unlock: ante 4
122. **Flower Pot** — ×3 Mult if scoring hand contains a ♦, ♣, ♥, and ♠ card ($6, Uncommon, Xm, Indep) · unlock: ante 8
123. **Blueprint** — copies ability of the joker to the right ($10, Rare, !!, copy) · unlock: win a run
124. **Wee Joker** — +8 Chips per played 2 scored (start: +0) ($8, Rare, +c, Mixed) · unlock: win in ≤18 rounds
125. **Merry Andy** — +3 discards each round, −1 hand size ($7, Uncommon, !!, N/A) · unlock: win in ≤12 rounds
126. **Oops! All 6s** — doubles all listed probabilities ($4, Uncommon, !!, rule) · unlock: 10,000 chips in one hand
127. **The Idol** — each played [rank] of [suit] gives ×2 Mult; changes each round ($6, Uncommon, Xm, On Scored) · unlock: 1,000,000 chips in one hand
128. **Seeing Double** — ×2 Mult if scoring hand has a ♣ card and a scoring card of any other suit ($6, Uncommon, Xm, Indep) · unlock: hand with four 7♣
129. **Matador** — earn $8 if played hand triggers the Boss Blind ability ($7, Uncommon, +$, Indep) · unlock: defeat a boss in 1 hand, no discards · dep: bosses
130. **Hit the Road** — ×0.5 Mult per Jack discarded this round (start: ×1) ($8, Rare, Xm, Mixed) · unlock: discard 5 Jacks at once
131. **The Duo** — ×2 Mult if played hand contains a Pair ($8, Rare, Xm, Indep) · unlock: win without playing a Pair
132. **The Trio** — ×3 Mult if contains Three of a Kind ($8, Rare, Xm, Indep) · unlock: win without Three of a Kind
133. **The Family** — ×4 Mult if contains Four of a Kind ($8, Rare, Xm, Indep) · unlock: win without Four of a Kind
134. **The Order** — ×3 Mult if contains a Straight ($8, Rare, Xm, Indep) · unlock: win without a Straight
135. **The Tribe** — ×2 Mult if contains a Flush ($8, Rare, Xm, Indep) · unlock: win without a Flush
136. **Stuntman** — +250 Chips, −2 hand size ($7, Rare, +c, Indep) · unlock: 100,000,000 chips in one hand
137. **Invisible Joker** — after 2 rounds, sell to duplicate a random joker ($8, Rare, !!, N/A) · unlock: win without ever having >4 jokers
138. **Brainstorm** — copies the ability of the leftmost joker ($10, Rare, !!, copy) · unlock: discard a Royal Flush
139. **Satellite** — earn $1 at end of round per unique Planet used this run ($6, Uncommon, +$, N/A) · unlock: $400 · dep: consumables
140. **Shoot the Moon** — each Queen held in hand gives +13 Mult ($5, Common, +m, On Held) · unlock: play every ♥ in deck in one round
141. **Driver's License** — ×3 Mult if ≥16 Enhanced cards in full deck ($7, Rare, Xm, Indep) · unlock: enhance 16 cards · dep: enhancements
142. **Cartomancer** — create a Tarot on blind select (needs room) ($6, Uncommon, !!, On Blind Select) · unlock: discover every Tarot · dep: consumables
143. **Astronomer** — all Planet cards and Celestial Packs in shop are free ($8, Uncommon, !!, N/A) · unlock: discover every Planet · dep: shop+packs
144. **Burnt Joker** — upgrade the level of the first discarded poker hand each round ($8, Rare, !!, On Discard) · unlock: sell 50 cards
145. **Bootstraps** — +2 Mult per $5 you have ($7, Uncommon, +m, Indep) · unlock: 2 Polychrome jokers at once
146. **Canio** — gains ×1 Mult when a face card is destroyed (start: ×1) (Legendary, Xm, Mixed) · The Soul
147. **Triboulet** — played Kings and Queens each give ×2 Mult when scored (Legendary, Xm, On Scored) · The Soul
148. **Yorick** — gains ×1 Mult every 23 cards discarded (start: ×1) (Legendary, Xm, Mixed) · The Soul
149. **Chicot** — disables the effect of every Boss Blind (Legendary, !!, On Blind Select) · The Soul · dep: bosses
150. **Perkeo** — creates a Negative copy of 1 random consumable at end of shop (Legendary, !!, N/A) · The Soul · dep: consumables

---

## System dependencies (drives implementation order)
- **Base game only** (cards/hands/deck/hands/discards/joker count/money read): the large majority — implementable on the Tier-0 engine + the joker engine.
- **shop**: Flash Card, Campfire, Astronomer (+ acquisition of all jokers).
- **consumables (tarot/planet/spectral)**: 8 Ball, Sixth Sense, Constellation, Superposition, Séance, Vagabond, Hallucination, Fortune Teller, Satellite, Cartomancer, Perkeo.
- **enhancements/editions/seals**: Steel Joker, Vampire, Midas Mask, Stone Joker, Lucky Cat, Golden Ticket, Glass Joker, Driver's License, Certificate (seals), Bootstraps (polychrome unlock only).
- **booster packs**: Red Card, Hallucination, Astronomer.
- **tags**: Diet Cola.
- **boss-blind effects**: Rocket, Luchador, Matador, Campfire, Chicot.

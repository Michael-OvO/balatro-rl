"""Card-aware actor-critic (Flax Linen).

A CARD-AWARE net (vs. a flat softmax that produces each play-logit from one pooled state
vector with no link to *which* cards a play selects):

  per-card SAB encoder  -> FiLM-conditioned card stream H[B,8,d]
  candidate-scoring head: static [218,5] gather over H -> masked DeepSets pool ->
                          per-subset play/discard/use-target scorers (a play-logit is now a
                          function of its selected cards)
  shop head:              per-slot MLPs over the joker/consumable/shop embeds + ctx
  acquisition heads (E5): pack OPEN / PICK / SKIP, BUY_VOUCHER, and USE_TARGET (the targeting
                          two-step, scored for free off the SAME candidate pool as play/discard)

Contract: `apply(params, obs, mask) -> (logits[B,706], value_logits[B,255])`. All 706 logits
are computed then `jnp.where(mask, logits, finfo.min)`. Everything is fixed-shape (static gather
over a constant index, masked pooling, broadcast_to) so the net jits exactly twice in training
(act@num_envs, update@mb_size). The logit assembly order is load-bearing — it matches
actions.py index-for-index: [play, disc, shop, use, use_target, open, pick, skip, voucher].
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal

from ..envs.actions import _PAIRS, _SUBSETS
from ..envs.obs import CONSUM_VOCAB, PACK_KIND_VOCAB, PACK_SIZE_VOCAB, VOUCHER_VOCAB
from .value_head import NBINS

JOKER_VOCAB = 200   # >= max JokerType id (123) + buffer; id 0 = empty/pad slot. Shared by jokers+shop+pack-items.

# --- static module constants (built once at import; baked into the graph, never params) ---
# _SUBSETS = the 218 fixed hand subsets (sizes {1:8,2:28,3:56,4:70,5:56}, maxlen 5);
# _PAIRS = the 20 ordered reorder pairs (i,j), i!=j, over the 5 joker slots.
_IDX = np.zeros((len(_SUBSETS), 5), np.int32)
_CNT = np.zeros((len(_SUBSETS), 5), np.float32)
for _i, _c in enumerate(_SUBSETS):
    for _j, _s in enumerate(_c):
        _IDX[_i, _j] = _s        # real member -> hand slot index
        _CNT[_i, _j] = 1.0       # 1 = real member; pad slots stay index 0 / count 0
SUBSET_IDX = jnp.asarray(_IDX)                                  # [218,5] int32, max 7 (in-range)
SUBSET_CNT = jnp.asarray(_CNT)                                  # [218,5] f32, 1=real / 0=pad
PAIR_I = jnp.asarray([p[0] for p in _PAIRS], np.int32)         # [20]
PAIR_J = jnp.asarray([p[1] for p in _PAIRS], np.int32)         # [20]


def _masked_pool(tok, valid):
    """Masked sum+mean over a token set: [B,N,d],[B,N] -> [B,2d]."""
    v = valid[..., None]                                       # [B,N,1]
    summed = (tok * v).sum(1)                                  # [B,d]
    mean = summed / jnp.clip(v.sum(1), 1.0)                    # [B,d]
    return jnp.concatenate([summed, mean], axis=-1)            # [B,2d]


class ActorCritic(nn.Module):
    action_dim: int
    d_model: int = 128
    n_bins: int = NBINS
    num_heads: int = 4
    n_layers: int = 2
    pool: str = "deepsets"

    @nn.compact
    def __call__(self, obs, action_mask):
        d = self.d_model

        # ---- per-card tokens: [B,8,37] one-hots -> [B,8,d], + learned pos + card segment ----
        # seg tags: 0 card, 1 joker, 2 shop, 3 consumable, 4 pack, 5 voucher.
        seg = self.param("seg", nn.initializers.normal(0.02), (6, d))
        pos = self.param("pos", nn.initializers.normal(0.02), (8, d))   # hand-slot positions
        x = nn.Dense(d)(obs["hand"]) + pos[None] + seg[0]               # [B,8,d]

        # ---- L pre-LN masked self-attention over ONLY the 8 card tokens ----
        attn_mask = (obs["hand_mask"][:, None, None, :] > 0)           # [B,1,1,8] bool
        for _ in range(self.n_layers):
            y = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=d)(
                y, y, mask=attn_mask)
            y = nn.LayerNorm()(x)
            x = x + nn.Dense(d)(nn.gelu(nn.Dense(4 * d)(y)))           # FFN
        attended = x                                                   # [B,8,d]

        # ---- shared embeds (jokers + shop-jokers + pack-item-jokers; consumables everywhere) ----
        embed = nn.Embed(JOKER_VOCAB, d)
        cembed = nn.Embed(CONSUM_VOCAB, d)
        pkembed = nn.Embed(PACK_KIND_VOCAB, d)
        psembed = nn.Embed(PACK_SIZE_VOCAB, d)
        vembed = nn.Embed(VOUCHER_VOCAB, d)

        je = embed(obs["joker_types"]) + nn.Dense(d)(obs["joker_counter"][..., None]) + seg[1]  # [B,6,d]
        # shop slot = joker embed (if a joker offer) + consumable embed (if a consumable offer) + cost
        se = (embed(obs["shop_types"]) + cembed(obs["shop_consum"])
              + nn.Dense(d)(obs["shop_cost"][..., None]) + seg[2])      # [B,2,d]
        ce = cembed(obs["consum_types"]) + seg[3]                      # [B,MAX_CONSUM,d] owned consumables
        # pack OFFERS (shop) and revealed pack ITEMS (OPEN_PACK)
        pe = (pkembed(obs["pack_kind"]) + psembed(obs["pack_size"])
              + nn.Dense(d)(obs["pack_cost"][..., None]) + seg[4])      # [B,MAX_PACK,d]
        pie = embed(obs["pack_item_joker"]) + cembed(obs["pack_item_consum"]) + seg[4]  # [B,MAX_PACK_ITEMS,d]
        ve = vembed(obs["voucher_offer"]) + seg[5]                     # [B,1,d]
        pending_ctx = cembed(obs["pending_consum"])[:, 0, :]          # [B,d] which Tarot is armed

        joker_ctx = _masked_pool(je, obs["joker_mask"])               # [B,2d]
        shop_ctx = _masked_pool(se, obs["shop_mask"])                 # [B,2d]
        consum_ctx = _masked_pool(ce, obs["consum_mask"])             # [B,2d]
        pack_ctx = _masked_pool(pe, obs["pack_offer_mask"])           # [B,2d]
        item_ctx = _masked_pool(pie, obs["pack_open_mask"])           # [B,2d]
        voucher_ctx = ve[:, 0, :] * obs["voucher_offer_mask"]        # [B,d]
        # boss one-hot + owned-voucher multihot ride in the global block.
        g = jnp.concatenate([obs["global"], obs["boss_onehot"], obs["levels"],
                             obs["deck_rank_hist"], obs["deck_suit_hist"],
                             obs["vouchers_owned"]], axis=-1)
        g = nn.gelu(nn.Dense(d)(g))                                   # [B,d]
        ctx = nn.gelu(nn.Dense(d)(jnp.concatenate(
            [joker_ctx, shop_ctx, consum_ctx, pack_ctx, item_ctx,
             voucher_ctx, pending_ctx, g], axis=-1)))                 # [B,d]

        # ---- FiLM-condition the card stream on ctx (keeps the 8-slot gather indices stable) ----
        gamma, beta = jnp.split(nn.Dense(2 * d)(ctx), 2, axis=-1)
        H = attended * (1.0 + gamma[:, None, :]) + beta[:, None, :]   # [B,8,d]

        # ---- candidate pool: static gather -> masked DeepSets pool -> per-subset scorers ----
        gathered = H[:, SUBSET_IDX, :]                               # [B,218,5,d] one XLA gather
        slot_valid = obs["hand_mask"][:, SUBSET_IDX]                 # [B,218,5] absent-slot subsets -> 0
        eff = (SUBSET_CNT[None] * slot_valid)[..., None]            # [B,218,5,1] real & present members
        phi = nn.Dense(d)(nn.gelu(nn.Dense(d)(gathered)))           # shared per-card MLP_phi [B,218,5,d]
        s_sum = (phi * eff).sum(2)                                   # [B,218,d]
        n = eff.sum(2)                                               # [B,218,1] cardinality 1..5
        s_mean = s_sum / jnp.clip(n, 1.0, None)                      # [B,218,d]
        pooled = jnp.concatenate([s_sum, s_mean, n], axis=-1)       # [B,218,2d+1] sum+mean+count
        set_emb = nn.gelu(nn.Dense(d)(pooled))                      # shared MLP_set [B,218,d]
        ctx_b = jnp.broadcast_to(ctx[:, None, :], set_emb.shape)    # broadcast_to (no materialized repeat)
        cand = jnp.concatenate([set_emb, ctx_b], axis=-1)           # [B,218,2d]
        # SEPARATE final scorers, SHARED phi/set: discarding a pair is good, playing it is bad,
        # and applying a Tarot to it is a third, distinct judgement — all card-aware off one pool.
        play_logits = nn.Dense(1, kernel_init=orthogonal(0.01))(nn.gelu(nn.Dense(d)(cand)))[..., 0]  # [B,218]
        disc_logits = nn.Dense(1, kernel_init=orthogonal(0.01))(nn.gelu(nn.Dense(d)(cand)))[..., 0]  # [B,218]
        use_target_logits = nn.Dense(1, kernel_init=orthogonal(0.01))(
            nn.gelu(nn.Dense(d)(cand)))[..., 0]                                                       # [B,218]

        # ---- SHOP head (29): order matches actions.py (buy2, sell5, reroll1, reorder20, leave1) ----
        ctx_b2 = jnp.broadcast_to(ctx[:, None, :], se.shape)        # [B,2,d]
        ctx_b6 = jnp.broadcast_to(ctx[:, None, :], je.shape)        # [B,6,d]
        buy = nn.Dense(1)(nn.gelu(nn.Dense(d)(jnp.concatenate([se, ctx_b2], -1))))[..., 0]   # [B,2]
        sell = nn.Dense(1)(nn.gelu(nn.Dense(d)(jnp.concatenate([je, ctx_b6], -1))))[..., 0]  # [B,6]
        reroll = nn.Dense(1)(nn.gelu(nn.Dense(d)(ctx)))                                       # [B,1]
        ji = je[:, PAIR_I, :]                                       # [B,30,d]
        jj = je[:, PAIR_J, :]                                       # [B,30,d]
        ctx_b30 = jnp.broadcast_to(ctx[:, None, :], ji.shape)      # [B,30,d]
        reorder = nn.Dense(1)(nn.gelu(nn.Dense(d)(jnp.concatenate([ji, jj, ctx_b30], -1))))[..., 0]  # [B,30]
        leave = nn.Dense(1)(nn.gelu(nn.Dense(d)(ctx)))                                        # [B,1]
        shop_logits = jnp.concatenate([buy, sell, reroll, reorder, leave], axis=-1)          # [B,40]

        # ---- USE head (MAX_CONSUM): score each owned-consumable slot from its embed + ctx ----
        ctx_bc = jnp.broadcast_to(ctx[:, None, :], ce.shape)        # [B,MAX_CONSUM,d]
        use = nn.Dense(1)(nn.gelu(nn.Dense(d)(jnp.concatenate([ce, ctx_bc], -1))))[..., 0]    # [B,MAX_CONSUM]

        # ---- E5 acquisition heads: OPEN (pack offers), PICK (revealed items), SKIP, BUY_VOUCHER ----
        ctx_bp = jnp.broadcast_to(ctx[:, None, :], pe.shape)        # [B,MAX_PACK,d]
        open_logits = nn.Dense(1)(nn.gelu(nn.Dense(d)(jnp.concatenate([pe, ctx_bp], -1))))[..., 0]  # [B,MAX_PACK]
        ctx_bpi = jnp.broadcast_to(ctx[:, None, :], pie.shape)     # [B,MAX_PACK_ITEMS,d]
        pick_logits = nn.Dense(1)(nn.gelu(nn.Dense(d)(jnp.concatenate([pie, ctx_bpi], -1))))[..., 0]  # [B,5]
        skip_logits = nn.Dense(1)(nn.gelu(nn.Dense(d)(ctx)))                                  # [B,1]
        ctx_bv = jnp.broadcast_to(ctx[:, None, :], ve.shape)       # [B,1,d]
        voucher_logits = nn.Dense(1)(nn.gelu(nn.Dense(d)(jnp.concatenate([ve, ctx_bv], -1))))[..., 0]  # [B,1]

        # ---- assembly + mask + value (load-bearing order: matches actions.py index-for-index) ----
        logits = jnp.concatenate([play_logits, disc_logits, shop_logits, use,
                                  use_target_logits, open_logits, pick_logits,
                                  skip_logits, voucher_logits], axis=-1)                       # [B,694]
        logits = jnp.where(action_mask, logits, jnp.finfo(logits.dtype).min)
        value_logits = nn.Dense(self.n_bins, kernel_init=orthogonal(1.0))(
            nn.gelu(nn.Dense(d)(ctx)))                                                        # distributional head on ctx
        return logits, value_logits

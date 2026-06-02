"""Deep-Sets actor-critic (Flax Linen). Per-group embed + segment tags ->
masked mean+sum pool -> MLP. Flat masked-categorical policy head + symlog
two-hot value head. (SAB attention encoder is the Plan-6 swap of the encoder.)
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal

from .value_head import NBINS

JOKER_VOCAB = 200   # >= max JokerType id (123) + buffer; id 0 = empty/pad slot


class DeepSetsEncoder(nn.Module):
    d_model: int = 128

    @nn.compact
    def __call__(self, obs):
        d = self.d_model
        seg = self.param("seg", nn.initializers.normal(0.02), (3, d))
        card_tok = nn.Dense(d)(obs["hand"]) + seg[0]                              # [B,8,d]
        je = nn.Embed(JOKER_VOCAB, d)(obs["joker_types"])                         # [B,5,d]
        jok_tok = je + nn.Dense(d)(obs["joker_counter"][..., None]) + seg[1]
        se = nn.Embed(JOKER_VOCAB, d)(obs["shop_types"])
        shop_tok = se + nn.Dense(d)(obs["shop_cost"][..., None]) + seg[2]

        tokens = nn.relu(jnp.concatenate([card_tok, jok_tok, shop_tok], axis=1))  # [B,15,d]
        valid = jnp.concatenate([obs["hand_mask"], obs["joker_mask"], obs["shop_mask"]],
                                axis=1)[..., None]                                # [B,15,1]
        summed = (tokens * valid).sum(1)                                          # [B,d]
        mean = summed / jnp.clip(valid.sum(1), 1.0)
        state_emb = nn.relu(nn.Dense(d)(jnp.concatenate([mean, summed], axis=-1)))

        g = jnp.concatenate([obs["global"], obs["levels"],
                             obs["deck_rank_hist"], obs["deck_suit_hist"]], axis=-1)
        g = nn.relu(nn.Dense(d)(g))
        return nn.relu(nn.Dense(d)(jnp.concatenate([state_emb, g], axis=-1)))      # [B,d]


class ActorCritic(nn.Module):
    action_dim: int
    d_model: int = 128
    n_bins: int = NBINS

    @nn.compact
    def __call__(self, obs, action_mask):
        h = DeepSetsEncoder(self.d_model)(obs)
        a = nn.relu(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)))(h))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(a)
        logits = jnp.where(action_mask, logits, jnp.finfo(logits.dtype).min)
        c = nn.relu(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)))(h))
        value_logits = nn.Dense(self.n_bins, kernel_init=orthogonal(1.0))(c)
        return logits, value_logits

"""JAX-native vectorized environment: a drop-in replacement for SyncVectorEnv.

``JaxVectorEnv`` holds N ``CoreState`` instances on-device and steps them
in parallel via ``batched_step = vmap(step_with_action)``.  The public
interface matches ``SyncVectorEnv`` exactly so ``train.py`` can swap the
factory with no rollout-loop changes (Task 1.9).

Key design points
-----------------
* ``reset() -> (obs_dict, masks)``   — same shapes/keys as SyncVectorEnv.
* ``step(actions int32[N]) -> (obs, rewards, dones, infos, masks)``
  - ``obs``     : dict of jnp arrays, each leading dim N (same keys as SyncVec).
  - ``rewards`` : float32[N] numpy array.
  - ``dones``   : bool[N] numpy array.
  - ``infos``   : list of N dicts with keys train.py reads:
                  "cleared" (bool), "ante" (int), "score" (int),
                  "round_score" (int).
  - ``masks``   : bool[N, 708] numpy array.
* ``set_req_scale(scale)``  — rebuilds the required table and patches all
  in-state ``required_table`` fields; future blind advances use the new scale.
* ``set_boss_rate(rate)``   — no-op (core has no boss system); warns once.
"""
from __future__ import annotations

import warnings
import numpy as np
import jax
import jax.numpy as jnp

from balatro_rl.engine_jax.curriculum import build_required_table
from balatro_rl.engine_jax.step import batched_reset, batched_step
from balatro_rl.engine_jax.obs import encode_core, legal_mask_core
from balatro_rl.envs.actions import MAX_JOKERS

# vmap wrappers for obs + mask encoding over a batched CoreState.
_vmapped_encode    = jax.jit(jax.vmap(encode_core))
_vmapped_legal     = jax.jit(jax.vmap(legal_mask_core))
_jit_batched_step  = jax.jit(batched_step)
_jit_batched_reset = jax.jit(batched_reset)


class JaxVectorEnv:
    """N-env fully-on-device vectorized Balatro environment.

    Args:
        num_envs:     Number of parallel environments.
        reward_name:  Reward shaping variant.  Only ``"shaped"`` is supported
                      by the JAX core; other names raise ValueError.
        base_seed:    Integer seed for the initial PRNG keys.
        req_scale:    Curriculum scale for required blind scores (1.0 = real game).
        enable_bosses: Must be False — the JAX core has no boss system.

    Raises:
        ValueError: if ``enable_bosses=True`` (not implemented in the core).
        ValueError: if ``reward_name`` is not ``"shaped"``.
    """

    def __init__(
        self,
        num_envs: int,
        reward_name: str = "shaped",
        base_seed: int = 0,
        req_scale: float = 1.0,
        enable_bosses: bool = False,
    ):
        if enable_bosses:
            raise ValueError(
                "JaxVectorEnv: enable_bosses=True is not supported — the JAX "
                "core engine has no boss blind system.  Use SyncVectorEnv for "
                "boss-blind training."
            )
        if reward_name != "shaped":
            raise ValueError(
                f"JaxVectorEnv: reward_name={reward_name!r} is not supported. "
                "Only 'shaped' is implemented in the JAX core."
            )

        self.num_envs = num_envs
        self._base_seed = base_seed
        self._req_scale = req_scale
        self._boss_rate_warned = False

        # Build the required-score lookup table and seed N PRNG keys.
        self.required_table = jnp.asarray(
            build_required_table(req_scale), dtype=jnp.int32
        )  # [9, 3]

        base_key = jax.random.PRNGKey(base_seed)
        keys = jax.random.split(base_key, num_envs)  # [N, 2] uint32

        # Initialise N environments on-device (empty joker loadout by default).
        _zero_jk = jnp.zeros((num_envs, MAX_JOKERS), dtype=jnp.int32)
        self.state = _jit_batched_reset(keys, self.required_table, _zero_jk)

        # Cache obs/mask for the reset state (avoids a redundant encode on reset()).
        self._obs  = None
        self._mask = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all N environments and return initial (obs, masks).

        Returns:
            obs:   dict mapping str -> jnp.ndarray with leading dim N.
            masks: bool ndarray [N, 708].
        """
        # Re-seed and reset every env.
        base_key = jax.random.PRNGKey(self._base_seed)
        keys = jax.random.split(base_key, self.num_envs)
        _zero_jk = jnp.zeros((self.num_envs, MAX_JOKERS), dtype=jnp.int32)
        self.state = _jit_batched_reset(keys, self.required_table, _zero_jk)

        self._obs  = _vmapped_encode(self.state)
        self._mask = np.asarray(_vmapped_legal(self.state), dtype=bool)
        return self._obs, self._mask

    def step(self, actions):
        """Step all N environments with the provided actions.

        Args:
            actions: int32 array-like of shape [N]; one flat action id per env.

        Returns:
            obs:     dict str -> jnp.ndarray [N, ...] (post-step / post-reset obs).
            rewards: float32 ndarray [N].
            dones:   bool ndarray [N].
            infos:   list of N dicts, each with keys:
                       "cleared"     (bool)  — True iff this step cleared a blind.
                       "ante"        (int)   — ante AFTER the step (pre-reset for done).
                       "score"       (int)   — hand score for this PLAY (0 for DISCARD).
                       "round_score" (int)   — cumulative round score (pre-reset if done).
            masks:   bool ndarray [N, 708] (from the post-step / fresh-reset state).
        """
        actions = jnp.asarray(actions, dtype=jnp.int32)

        # Capture pre-step state fields needed for infos (terminal values before reset).
        # step_with_action returns (final_state, reward, done, signals) where:
        #   - final_state is the POST-reset state for done envs (fresh episode).
        #   - reward / done / signals reflect the TERMINAL transition.
        # We need the terminal ante and round_score for logging; since step_with_action
        # auto-resets done envs, we read those from signals (cleared/won) and from a
        # separate pre-step snapshot of ante/round_score for done envs.
        # Strategy: capture ante and round_score from the CURRENT state before stepping,
        # then use signals for cleared/won; for ante we use the pre-step value for done
        # envs (to expose the ante at which the episode ended, matching SyncVectorEnv's
        # info["ante"] = int(nxt.ante) which is the terminal state's ante before reset).
        pre_ante        = self.state.ante          # [N] int32, before step
        pre_round_score = self.state.round_score   # [N] int32, before step

        # --- on-device batched step -------------------------------------------
        self.state, rewards_jnp, dones_jnp, signals = _jit_batched_step(
            self.state, actions
        )

        # --- encode obs + mask from the new (post-reset-if-done) state --------
        self._obs  = _vmapped_encode(self.state)
        self._mask = np.asarray(_vmapped_legal(self.state), dtype=bool)

        # --- materialise infos ------------------------------------------------
        # Convert to numpy once for the info-building loop.
        cleared_np     = np.asarray(signals.cleared, dtype=bool)     # [N]
        dones_np       = np.asarray(dones_jnp, dtype=bool)           # [N]
        rewards_np     = np.asarray(rewards_jnp, dtype=np.float32)   # [N]
        # For done envs: use the pre-step ante/round_score (terminal transition values).
        # For live envs: use the post-step state's ante/round_score (which is the same
        # as what SyncVectorEnv exposes: info["ante"] = int(nxt.ante)).
        # However, step_with_action auto-resets done envs to ante=1, so to expose the
        # terminal ante we must use pre_step for done lanes.
        post_ante        = np.asarray(self.state.ante, dtype=np.int32)          # [N], post-reset
        post_round_score = np.asarray(self.state.round_score, dtype=np.int32)   # [N], post-reset
        pre_ante_np        = np.asarray(pre_ante, dtype=np.int32)               # [N], pre-step
        pre_round_score_np = np.asarray(pre_round_score, dtype=np.int32)        # [N], pre-step
        score_np           = np.asarray(signals.score, dtype=np.int32)          # [N]

        infos = []
        for i in range(self.num_envs):
            d = dones_np[i]
            # ante: use pre-step for done envs (terminal ante), post-step for live
            ante_i        = int(pre_ante_np[i])        if d else int(post_ante[i])
            round_score_i = int(pre_round_score_np[i]) if d else int(post_round_score[i])
            infos.append({
                "cleared":     bool(cleared_np[i]),
                "ante":        ante_i,
                "score":       int(score_np[i]),
                "round_score": round_score_i,
            })

        return self._obs, rewards_np, dones_np, infos, self._mask

    def set_req_scale(self, scale: float):
        """Update the curriculum scale; future blind advances use the new required scores.

        Rebuilds the required-score table and patches ``required_table`` in every
        in-state CoreState so the NEXT blind advance (after the current blind ends)
        uses the new scale.  In-progress blinds keep their current ``required``
        field (matching SyncVectorEnv semantics: "new (incl. auto-reset) episodes
        use it").

        Note: because the JAX engine stores ``required_table`` inside the state
        pytree (used only on blind advance), patching it is sufficient — the
        current blind's ``state.required`` is not changed.
        """
        self._req_scale = scale
        self.required_table = jnp.asarray(
            build_required_table(scale), dtype=jnp.int32
        )
        # Patch the in-state required_table for all N envs so future blind
        # advances (in step -> advance_state) use the new scale.
        self.state = self.state._replace(
            required_table=jnp.broadcast_to(
                self.required_table[None],                          # [1, 9, 3]
                (self.num_envs,) + self.required_table.shape,      # [N, 9, 3]
            )
        )

    def set_boss_rate(self, rate: float):
        """No-op: the JAX core has no boss blind system.

        Stored for API parity with SyncVectorEnv; a one-time warning is emitted
        if rate != 0.0 (i.e. the caller expects bosses to actually activate).
        """
        if rate != 0.0 and not self._boss_rate_warned:
            warnings.warn(
                f"JaxVectorEnv.set_boss_rate({rate!r}): boss blinds are not "
                "implemented in the JAX core engine — this call is a no-op.",
                stacklevel=2,
            )
            self._boss_rate_warned = True

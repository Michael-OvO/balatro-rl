"""PPO training loop: numpy SyncVectorEnv stepped in a Python for-loop; only the
per-step policy `act` and the whole PPO `update` are jit-compiled. Fixed shapes /
dtypes / pytree keys keep both jit-stable (no per-step recompiles).
"""
from __future__ import annotations

import collections
import dataclasses
import typing
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from ..envs.actions import NUM_ACTIONS
from ..envs.vec_env import SyncVectorEnv
from .eval import evaluate
from .metrics_logger import NullLogger
from .networks import ActorCritic
from .ppo import gae, log_prob, ppo_loss, sample_action
from .spec import dummy_obs
from .value_head import value_decode


@dataclasses.dataclass
class TrainConfig:
    num_envs: int = 64
    num_steps: int = 128
    num_updates: int = 10
    d_model: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: typing.Union[float, typing.Callable[[int], float]] = 0.01
    reward_name: str = "shaped"
    seed: int = 0
    enable_bosses: bool = False     # boss blinds in the training env
    eval_interval: int = 0          # run greedy eval every N updates (0 = off)
    eval_seeds: tuple = (0, 1, 2, 3)
    # Curriculum: shrink the blind target so the agent experiences clearing, then ramp to 1.0.
    # `curr_floor` < 1.0 enables the closed-loop ramp (start at curr_floor, raise on clear-rate);
    # a callable `req_scale_schedule` overrides with an open-loop schedule. Default = real game.
    req_scale_schedule: typing.Union[float, typing.Callable[[int], float]] = 1.0
    curr_floor: float = 1.0
    ramp_clear_rate: float = 0.7
    ramp_step: float = 0.05
    ramp_window: int = 20
    # E5 boss curriculum: when True, the per-episode boss probability tracks cur_scale, so bosses
    # fade in as the score target ramps (the plateau came from full-strength bosses under a still-
    # low target). No-op unless enable_bosses; eval always uses the full deploy target (boss_rate=1).
    boss_curriculum: bool = True
    # Early stopping: stop once the eval metric stops improving — catches the plateau/overfit where
    # more updates no longer help (and may regress). Counts CONSECUTIVE evals without a > min_delta
    # improvement over the best-so-far; stops at `patience`. 0 = off (train the full num_updates).
    # Only meaningful with eval_interval > 0. Best-checkpoint tracking is the caller's job (on_update).
    early_stop_patience: int = 0
    early_stop_metric: str = "eval/mean_blinds_cleared"
    early_stop_min_delta: float = 0.0


@dataclasses.dataclass
class TrainResult:
    params: object
    losses: list           # [(total, pg, vl, ent), ...] one per update
    mean_returns: list      # mean per-step reward each update (finite scalar)
    eval_history: list = dataclasses.field(default_factory=list)   # one eval-metrics dict per eval


def _to_jax(obs: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in obs.items()}


def _ent_coef_at(ent_coef, update_idx: int) -> float:
    """Entropy coefficient for this update: a fixed float, or a schedule
    callable(update_idx)->float evaluated on the host. The value is passed into the
    jitted `update` as a 0-d traced scalar, so changing it never recompiles."""
    return float(ent_coef(update_idx)) if callable(ent_coef) else float(ent_coef)


def _ramp_scale(cur_scale: float, clear_rate: float, can_bump: bool, cfg) -> float:
    """Closed-loop curriculum: when a bump is allowed (window full AND the post-bump cooldown has
    elapsed) AND the agent clears reliably at the CURRENT scale (clear_rate > ramp_clear_rate),
    raise the blind target toward the real game (1.0) by ramp_step; otherwise hold. Already at 1.0
    stays at 1.0. The cooldown is what gates `can_bump` — without it the sliding window still holds
    clear-rates from the easier prior scale right after a bump, so the gate would re-fire and race
    0.2->1.0 in ~ramp_step^-1 updates without ever measuring the new scale."""
    if cur_scale < 1.0 and can_bump and clear_rate > cfg.ramp_clear_rate:
        return min(1.0, round(cur_scale + cfg.ramp_step, 4))
    return cur_scale


def train(cfg: TrainConfig, logger=None, init_params=None, on_update=None) -> TrainResult:
    """Train a card-aware PPO agent. `on_update(update_idx, params)`, if given, is called at the
    end of every update with the live params — the entrypoint uses it to checkpoint periodically
    so a long (multi-hour) run can't lose everything to a late crash. train() stays agnostic to
    serialization/filesystem; the callback decides cadence and where to write."""
    if logger is None:
        logger = NullLogger()
    key = jax.random.PRNGKey(cfg.seed)
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=cfg.d_model)
    key, init_key = jax.random.split(key)
    # Warm start from a prior checkpoint (continue training) when init_params is given;
    # else fresh init. The provided params must match this net's shape (d_model/action dim).
    params = (init_params if init_params is not None
              else net.init(init_key, _to_jax(dummy_obs(1)), jnp.ones((1, NUM_ACTIONS), bool)))
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr, eps=1e-5))
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    @jax.jit
    def act(params, obs, mask, key):
        logits, value_logits = net.apply(params, obs, mask)
        key, sub = jax.random.split(key)
        action = sample_action(logits, sub)
        lp = log_prob(logits, action)
        value = value_decode(value_logits)
        return action, lp, value, key

    @jax.jit
    def update(ts, batch, last_value, key, ent_coef):
        adv, targets = gae(batch["rewards"], batch["values"], batch["dones"],
                            last_value, cfg.gamma, cfg.gae_lambda)
        B = cfg.num_steps * cfg.num_envs
        flat = {
            "obs": {k: v.reshape((B,) + v.shape[2:]) for k, v in batch["obs"].items()},
            "masks": batch["masks"].reshape((B, NUM_ACTIONS)),
            "actions": batch["actions"].reshape((B,)),
            "old_logp": batch["logps"].reshape((B,)),
            "adv": adv.reshape((B,)),
            "targets": targets.reshape((B,)),
        }

        def epoch(state, _):
            ts, key = state
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, B)
            shuf = {kk: (jax.tree_util.tree_map(lambda x: x[perm], vv) if kk == "obs" else vv[perm])
                    for kk, vv in flat.items()}
            mb_size = B // cfg.num_minibatches

            def minibatch(ts, i):
                sl = jax.lax.dynamic_slice_in_dim
                mb = {kk: (jax.tree_util.tree_map(lambda x: sl(x, i * mb_size, mb_size), vv)
                           if kk == "obs" else sl(vv, i * mb_size, mb_size))
                      for kk, vv in shuf.items()}
                (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    ts.params, net.apply, mb, cfg.clip, cfg.vf_coef, ent_coef)
                return ts.apply_gradients(grads=grads), (loss,) + aux

            ts, losses = jax.lax.scan(minibatch, ts, jnp.arange(cfg.num_minibatches))
            return (ts, key), losses

        (ts, key), losses = jax.lax.scan(epoch, (ts, key), None, cfg.update_epochs)
        # mean of the last epoch's minibatch losses
        last = jax.tree_util.tree_map(lambda x: x[-1].mean(), losses)
        return ts, last, key

    # Curriculum: open-loop schedule (callable) overrides; else closed-loop ramp from
    # curr_floor when curr_floor < 1.0; else the real game (scale 1.0).
    _open_loop = callable(cfg.req_scale_schedule)
    cur_scale = float(cfg.req_scale_schedule(0)) if _open_loop else float(cfg.curr_floor)
    # Boss probability tracks the score curriculum (E5); 1.0 when the boss curriculum is off.
    def _boss_rate_for(scale: float) -> float:
        return float(scale) if cfg.boss_curriculum else 1.0
    venv = SyncVectorEnv(cfg.num_envs, cfg.reward_name, base_seed=cfg.seed + 1000,
                         req_scale=cur_scale, enable_bosses=cfg.enable_bosses,
                         boss_rate=_boss_rate_for(cur_scale))
    next_obs, next_mask = venv.reset()
    T, N = cfg.num_steps, cfg.num_envs
    assert (T * N) % cfg.num_minibatches == 0, (
        f"rollout batch {T * N} (num_steps*num_envs) must be divisible by "
        f"num_minibatches {cfg.num_minibatches}; otherwise minibatching silently drops rows"
    )
    if (T * N) // cfg.num_minibatches > 4096:
        warnings.warn(f"minibatch {(T * N) // cfg.num_minibatches} is large; the card-aware net's "
                      "[mb,218,5,d] candidate gather may OOM — raise num_minibatches.", stacklevel=2)
    losses, mean_returns = [], []
    eval_history = []
    # Curriculum ramp signal: fraction of EPISODES that clear >=1 blind (a probability in
    # [0,1], gated against ramp_clear_rate) -- not blinds-per-episode (which exceeds 1 and
    # made the 0.7 gate meaningless). ep_cleared persists across updates (episodes span
    # rollouts). The windows SLIDE (maxlen) and are NOT reset on a bump (bug fix), so the
    # ramp tracks a moving average instead of waiting a full fresh window after each +scale.
    # A COOLDOWN (updates_since_bump) gates consecutive bumps: after a +scale the window still
    # holds clear-rates measured at the EASIER prior scale, so without a cooldown the gate would
    # re-fire every update and race 0.2->1.0 before the new (harder) scale is ever evaluated.
    # Requiring ramp_window updates between bumps lets the window refill at the new scale first.
    # Init at ramp_window so the FIRST bump waits only on the window filling, not the cooldown.
    ep_cleared = np.zeros(cfg.num_envs, bool)
    cleared_w = collections.deque(maxlen=cfg.ramp_window)
    episodes_w = collections.deque(maxlen=cfg.ramp_window)
    updates_since_bump = cfg.ramp_window
    best_eval = float("-inf")        # early stop: best eval metric + consecutive non-improving evals
    evals_no_improve = 0

    for _ in range(cfg.num_updates):
        ec = _ent_coef_at(cfg.ent_coef, len(losses))   # 0-based index of THIS update
        if _open_loop:
            cur_scale = float(cfg.req_scale_schedule(len(losses)))
            venv.set_req_scale(cur_scale)
            venv.set_boss_rate(_boss_rate_for(cur_scale))
        update_cleared = update_episodes = 0   # episodes that cleared >=1 blind / total episodes
        update_max_ante, update_max_hand_score, update_max_round = 1, 0, 0
        buf = {
            "obs": {k: np.zeros((T, N) + v.shape[1:], v.dtype) for k, v in next_obs.items()},
            "masks": np.zeros((T, N, NUM_ACTIONS), bool),
            "actions": np.zeros((T, N), np.int32),
            "logps": np.zeros((T, N), np.float32),
            "values": np.zeros((T, N), np.float32),
            "rewards": np.zeros((T, N), np.float32),
            "dones": np.zeros((T, N), np.float32),
        }
        for t in range(T):
            for k, v in next_obs.items():
                buf["obs"][k][t] = v
            buf["masks"][t] = next_mask
            action, lp, value, key = act(ts.params, _to_jax(next_obs), jnp.asarray(next_mask), key)
            a = np.asarray(action)
            buf["actions"][t] = a
            buf["logps"][t] = np.asarray(lp)
            buf["values"][t] = np.asarray(value)
            next_obs, rewards, dones, infos, next_mask = venv.step(a)
            buf["rewards"][t] = rewards
            buf["dones"][t] = dones.astype(np.float32)
            for i, info in enumerate(infos):   # aggregate clears + depth/score reached this rollout
                if info.get("cleared"):
                    ep_cleared[i] = True       # this env cleared a blind at some point this episode
                if info.get("ante", 1) > update_max_ante:
                    update_max_ante = info["ante"]
                if (info.get("score") or 0) > update_max_hand_score:
                    update_max_hand_score = info["score"]
                if info.get("round_score", 0) > update_max_round:
                    update_max_round = info["round_score"]
            for i in range(N):                 # on episode end, bank cleared? then reset the flag
                if dones[i]:
                    update_episodes += 1
                    update_cleared += int(ep_cleared[i])
                    ep_cleared[i] = False

        _, _, last_value, key = act(ts.params, _to_jax(next_obs), jnp.asarray(next_mask), key)
        batch = {
            "obs": {k: jnp.asarray(v) for k, v in buf["obs"].items()},
            "masks": jnp.asarray(buf["masks"]),
            "actions": jnp.asarray(buf["actions"]),
            "logps": jnp.asarray(buf["logps"]),
            "values": jnp.asarray(buf["values"]),
            "rewards": jnp.asarray(buf["rewards"]),
            "dones": jnp.asarray(buf["dones"]),
        }
        ts, last, key = update(ts, batch, jnp.asarray(last_value), key, jnp.float32(ec))
        total, pg, vl, ent = (float(last[0]), float(last[1]), float(last[2]), float(last[3]))
        losses.append((total, pg, vl, ent))
        mean_returns.append(float(buf["rewards"].mean()))
        update_idx = len(losses) - 1
        cleared_w.append(update_cleared)
        episodes_w.append(update_episodes)
        clear_rate = sum(cleared_w) / max(1, sum(episodes_w))   # frac of episodes clearing >=1 blind
        logger.log({"loss/total": total, "loss/policy": pg, "loss/value": vl,
                    "loss/entropy": ent, "train/mean_reward": mean_returns[-1],
                    "train/ent_coef": ec, "train/req_scale": cur_scale,
                    "train/boss_rate": _boss_rate_for(cur_scale),
                    "train/clear_rate": clear_rate, "train/max_ante": update_max_ante,
                    "train/max_hand_score": update_max_hand_score,
                    "train/max_round_score": update_max_round}, step=update_idx)
        if not _open_loop:
            window_full = len(episodes_w) == cfg.ramp_window
            can_bump = window_full and updates_since_bump >= cfg.ramp_window
            new_scale = _ramp_scale(cur_scale, clear_rate, can_bump, cfg)
            if new_scale != cur_scale:
                cur_scale = new_scale
                updates_since_bump = 0          # cooldown: re-measure clear-rate at the new scale
                venv.set_req_scale(cur_scale)   # windows SLIDE (no reset) -> moving-average ramp
                venv.set_boss_rate(_boss_rate_for(cur_scale))   # bosses fade in with the score bar
            else:
                updates_since_bump += 1
        if cfg.eval_interval and (update_idx % cfg.eval_interval == 0):
            # Eval on the DEPLOY target: the real game (full req_scale, bosses as trained),
            # not the training distribution it never deploys to.
            metrics = evaluate(net, ts.params, cfg.eval_seeds, cfg.reward_name,
                               enable_bosses=cfg.enable_bosses, req_scale=1.0)
            eval_history.append(metrics)
            logger.log(metrics, step=update_idx)
            # Early-stop accounting: track the best eval metric; count a plateau ONLY once the
            # curriculum has reached full difficulty (cur_scale >= 1.0) — during the ramp the
            # deploy-target eval is legitimately still climbing, so counting then would stop early.
            if cfg.early_stop_patience and cfg.early_stop_metric in metrics:
                cur = float(metrics[cfg.early_stop_metric])
                if cur > best_eval + cfg.early_stop_min_delta:
                    best_eval, evals_no_improve = cur, 0
                elif cur_scale >= 1.0:
                    evals_no_improve += 1
        if on_update is not None:        # periodic checkpoint hook (entrypoint decides cadence)
            on_update(update_idx, ts.params)
        if cfg.early_stop_patience and evals_no_improve >= cfg.early_stop_patience:
            logger.log({"train/early_stopped_at": float(update_idx),
                        "train/best_eval": best_eval}, step=update_idx)
            break

    logger.finish()
    return TrainResult(params=ts.params, losses=losses, mean_returns=mean_returns,
                       eval_history=eval_history)


if __name__ == "__main__":
    res = train(TrainConfig(num_updates=5, num_envs=16, num_steps=64))
    for i, (tot, pg, vl, ent) in enumerate(res.losses):
        print(f"update {i}: loss={tot:.3f} pg={pg:.3f} vl={vl:.3f} ent={ent:.3f} "
              f"mean_reward={res.mean_returns[i]:.3f}")

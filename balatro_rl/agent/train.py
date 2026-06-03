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


def _ramp_scale(cur_scale: float, clear_rate: float, window_full: bool, cfg) -> float:
    """Closed-loop curriculum: once the clear-rate window is full AND the agent clears
    reliably (clear_rate > ramp_clear_rate), raise the blind target toward the real game
    (1.0) by ramp_step; otherwise hold. Already at 1.0 stays at 1.0."""
    if cur_scale < 1.0 and window_full and clear_rate > cfg.ramp_clear_rate:
        return min(1.0, round(cur_scale + cfg.ramp_step, 4))
    return cur_scale


def train(cfg: TrainConfig, logger=None) -> TrainResult:
    if logger is None:
        logger = NullLogger()
    key = jax.random.PRNGKey(cfg.seed)
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=cfg.d_model)
    key, init_key = jax.random.split(key)
    params = net.init(init_key, _to_jax(dummy_obs(1)), jnp.ones((1, NUM_ACTIONS), bool))
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
    venv = SyncVectorEnv(cfg.num_envs, cfg.reward_name, base_seed=cfg.seed + 1000,
                         req_scale=cur_scale)
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
    clears_w = collections.deque(maxlen=cfg.ramp_window)
    dones_w = collections.deque(maxlen=cfg.ramp_window)

    for _ in range(cfg.num_updates):
        ec = _ent_coef_at(cfg.ent_coef, len(losses))   # 0-based index of THIS update
        if _open_loop:
            cur_scale = float(cfg.req_scale_schedule(len(losses)))
            venv.set_req_scale(cur_scale)
        update_clears = update_dones = 0
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
            update_dones += int(dones.sum())
            for info in infos:   # aggregate clears + depth/score reached this rollout
                if info.get("cleared"):
                    update_clears += 1
                if info.get("ante", 1) > update_max_ante:
                    update_max_ante = info["ante"]
                if (info.get("score") or 0) > update_max_hand_score:
                    update_max_hand_score = info["score"]
                if info.get("round_score", 0) > update_max_round:
                    update_max_round = info["round_score"]

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
        clears_w.append(update_clears)
        dones_w.append(update_dones)
        clear_rate = sum(clears_w) / max(1, sum(dones_w))   # avg blinds cleared per episode
        logger.log({"loss/total": total, "loss/policy": pg, "loss/value": vl,
                    "loss/entropy": ent, "train/mean_reward": mean_returns[-1],
                    "train/ent_coef": ec, "train/req_scale": cur_scale,
                    "train/clear_rate": clear_rate, "train/max_ante": update_max_ante,
                    "train/max_hand_score": update_max_hand_score,
                    "train/max_round_score": update_max_round}, step=update_idx)
        if not _open_loop:
            new_scale = _ramp_scale(cur_scale, clear_rate, len(clears_w) == cfg.ramp_window, cfg)
            if new_scale != cur_scale:
                cur_scale = new_scale
                venv.set_req_scale(cur_scale)
                clears_w.clear()
                dones_w.clear()
        if cfg.eval_interval and (update_idx % cfg.eval_interval == 0):
            metrics = evaluate(net, ts.params, cfg.eval_seeds, cfg.reward_name)
            eval_history.append(metrics)
            logger.log(metrics, step=update_idx)

    logger.finish()
    return TrainResult(params=ts.params, losses=losses, mean_returns=mean_returns,
                       eval_history=eval_history)


if __name__ == "__main__":
    res = train(TrainConfig(num_updates=5, num_envs=16, num_steps=64))
    for i, (tot, pg, vl, ent) in enumerate(res.losses):
        print(f"update {i}: loss={tot:.3f} pg={pg:.3f} vl={vl:.3f} ent={ent:.3f} "
              f"mean_reward={res.mean_returns[i]:.3f}")

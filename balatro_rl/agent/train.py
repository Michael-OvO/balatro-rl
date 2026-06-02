"""PPO training loop: numpy SyncVectorEnv stepped in a Python for-loop; only the
per-step policy `act` and the whole PPO `update` are jit-compiled. Fixed shapes /
dtypes / pytree keys keep both jit-stable (no per-step recompiles).
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from ..envs.actions import NUM_ACTIONS
from ..envs.vec_env import SyncVectorEnv
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
    ent_coef: float = 0.01
    reward_name: str = "shaped"
    seed: int = 0


@dataclasses.dataclass
class TrainResult:
    params: object
    losses: list           # [(total, pg, vl, ent), ...] one per update
    mean_returns: list      # mean per-step reward each update (finite scalar)


def _to_jax(obs: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in obs.items()}


def train(cfg: TrainConfig) -> TrainResult:
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
    def update(ts, batch, last_value, key):
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
                    ts.params, net.apply, mb, cfg.clip, cfg.vf_coef, cfg.ent_coef)
                return ts.apply_gradients(grads=grads), (loss,) + aux

            ts, losses = jax.lax.scan(minibatch, ts, jnp.arange(cfg.num_minibatches))
            return (ts, key), losses

        (ts, key), losses = jax.lax.scan(epoch, (ts, key), None, cfg.update_epochs)
        # mean of the last epoch's minibatch losses
        last = jax.tree_util.tree_map(lambda x: x[-1].mean(), losses)
        return ts, last, key

    venv = SyncVectorEnv(cfg.num_envs, cfg.reward_name, base_seed=cfg.seed + 1000)
    next_obs, next_mask = venv.reset()
    T, N = cfg.num_steps, cfg.num_envs
    assert (T * N) % cfg.num_minibatches == 0, (
        f"rollout batch {T * N} (num_steps*num_envs) must be divisible by "
        f"num_minibatches {cfg.num_minibatches}; otherwise minibatching silently drops rows"
    )
    losses, mean_returns = [], []

    for _ in range(cfg.num_updates):
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
            next_obs, rewards, dones, _, next_mask = venv.step(a)
            buf["rewards"][t] = rewards
            buf["dones"][t] = dones.astype(np.float32)

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
        ts, last, key = update(ts, batch, jnp.asarray(last_value), key)
        total, pg, vl, ent = (float(last[0]), float(last[1]), float(last[2]), float(last[3]))
        losses.append((total, pg, vl, ent))
        mean_returns.append(float(buf["rewards"].mean()))

    return TrainResult(params=ts.params, losses=losses, mean_returns=mean_returns)


if __name__ == "__main__":
    res = train(TrainConfig(num_updates=5, num_envs=16, num_steps=64))
    for i, (tot, pg, vl, ent) in enumerate(res.losses):
        print(f"update {i}: loss={tot:.3f} pg={pg:.3f} vl={vl:.3f} ent={ent:.3f} "
              f"mean_reward={res.mean_returns[i]:.3f}")

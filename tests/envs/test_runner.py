import json
from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.agents import RandomAgent, GreedyAgent
from balatro_rl.envs.runner import run_episode, Trajectory, replay


def test_run_episode_records_and_is_deterministic():
    env = BalatroEnv(reward_name="max_depth")
    t1 = run_episode(env, RandomAgent(seed=5), seed=11)
    t2 = run_episode(BalatroEnv(reward_name="max_depth"), RandomAgent(seed=5), seed=11)
    assert isinstance(t1, Trajectory)
    assert t1.seed == 11 and len(t1.actions) > 0
    assert t1.actions == t2.actions               # same agent seed + env seed
    assert t1.total_reward == t2.total_reward


def test_trajectory_json_roundtrip(tmp_path):
    env = BalatroEnv(reward_name="max_depth")
    t = run_episode(env, RandomAgent(seed=1), seed=2)
    p = tmp_path / "traj.json"
    t.save(p)
    loaded = Trajectory.load(p)
    assert loaded.seed == t.seed and loaded.actions == t.actions


def test_replay_reconstructs_final_state():
    env = BalatroEnv(reward_name="max_depth")
    t = run_episode(env, GreedyAgent(), seed=7)
    final = replay(t)            # re-run engine from seed + recorded actions
    assert final.done
    assert final.ante == t.final_ante and final.won == t.won

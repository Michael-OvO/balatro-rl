from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.agents import RandomAgent, GreedyAgent
from balatro_rl.envs.runner import run_episode, replay


def _mean_ante(agent_factory, n=20):
    total = 0
    for seed in range(n):
        t = run_episode(BalatroEnv("max_depth"), agent_factory(seed), seed=seed)
        assert replay(t).done                       # every episode terminates + replays
        total += t.final_ante
    return total / n


def test_all_episodes_terminate_and_replay():
    # Both agents finish every episode and the recorded trajectory replays cleanly.
    _mean_ante(lambda s: RandomAgent(seed=s), n=20)
    _mean_ante(lambda s: GreedyAgent(), n=20)


def test_greedy_at_least_as_deep_as_random():
    rand = _mean_ante(lambda s: RandomAgent(seed=s), n=20)
    greedy = _mean_ante(lambda s: GreedyAgent(), n=20)
    # Greedy (always plays the best hand) should reach at least as deep on average.
    assert greedy >= rand

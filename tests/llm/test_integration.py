import re

from balatro_rl.envs.balatro_env import BalatroEnv
from balatro_rl.envs.runner import Trajectory, run_episode, replay
from balatro_rl.llm.agent import LLMAgent


class ScriptedStubPolicy:
    """A no-LLM policy for tests: reads the rendered menu in the last user message and
    always returns a legal choice -- prefer a single-card play, else discard, else the
    first listed discrete option. Exercises the full serialize->menu->parse->step path."""

    def generate(self, messages: list[dict]) -> str:
        text = messages[-1]["content"]
        if 'play' in text and '"action"' in text and "play cards" in text.lower():
            return '{"action": "play", "cards": [0]}'
        if "discard cards" in text.lower():
            return '{"action": "discard", "cards": [0]}'
        m = re.search(r"\[(\d+)\]", text)         # first listed discrete option index
        idx = int(m.group(1)) if m else 0
        return f'{{"choice": {idx}}}'


def test_llm_agent_plays_a_full_game_and_produces_a_valid_trajectory():
    env = BalatroEnv(reward_name="shaped")
    agent = LLMAgent(policy=ScriptedStubPolicy())
    traj = run_episode(env, agent, seed=0)
    assert isinstance(traj, Trajectory)
    assert len(traj.actions) > 0
    # Engine determinism: the recorded (seed, actions) reconstructs to a terminal state.
    final = replay(traj)
    assert final.done or traj.truncated

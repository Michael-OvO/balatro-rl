from balatro_rl.engine import engine
from balatro_rl.envs.actions import decode, legal_mask
from balatro_rl.llm.agent import LLMAgent


class _AlwaysPlayFirstCard:
    """A stand-in policy: always returns a legal play of the first hand card."""
    def __init__(self):
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return '{"action": "play", "cards": [0]}'


def test_act_returns_a_legal_action_id():
    policy = _AlwaysPlayFirstCard()
    agent = LLMAgent(policy=policy)
    state = engine.reset(0)
    aid = agent.act(state, legal_mask(state))
    assert legal_mask(state)[aid]
    verb, _ = decode(aid)
    assert verb.name == "PLAY"


def test_act_retries_then_falls_back_on_bad_replies():
    class _Garbage:
        def generate(self, messages):
            return "no json here"
    agent = LLMAgent(policy=_Garbage(), max_retries=2)
    state = engine.reset(0)
    aid = agent.act(state, legal_mask(state))   # must still return SOMETHING legal
    assert legal_mask(state)[aid]


def test_act_passes_serialized_state_to_the_policy():
    policy = _AlwaysPlayFirstCard()
    agent = LLMAgent(policy=policy)
    state = engine.reset(0)
    agent.act(state, legal_mask(state))
    last_user_msg = policy.calls[-1][-1]["content"]
    assert "Ante 1" in last_user_msg and "Legal actions" in last_user_msg


def test_fallback_does_not_store_garbage_reply_in_context():
    class _Garbage:
        def generate(self, messages):
            return "no json here"
    agent = LLMAgent(policy=_Garbage(), max_retries=1)
    state = engine.reset(0)
    agent.act(state, legal_mask(state))
    last_reply = agent._ctx._turns[-1][1]   # (observation, assistant_reply)
    assert "no json here" not in last_reply
    assert "defaulted" in last_reply

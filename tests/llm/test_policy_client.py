from balatro_rl.llm.policy_client import Policy


class _Echo:
    def generate(self, messages):
        return messages[-1]["content"]


def test_policy_protocol_is_satisfied_by_duck_typing():
    p: Policy = _Echo()
    out = p.generate([{"role": "user", "content": "hi"}])
    assert out == "hi"


def test_policy_is_runtime_checkable():
    assert isinstance(_Echo(), Policy)

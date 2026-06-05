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


from balatro_rl.llm.policy_client import FrozenEndpointPolicy


class _FakeChat:
    def __init__(self, text):
        self._text = text
        self.seen = None

    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})

    def create(self, **kwargs):
        self.seen = kwargs
        return type("R", (), {"choices": [self._Msg(self._text)]})


class _FakeClient:
    def __init__(self, text):
        self.chat = type("C", (), {"completions": _FakeChat(text)})()


def test_frozen_policy_returns_reply_text_and_passes_messages():
    client = _FakeClient('{"choice": 0}')
    policy = FrozenEndpointPolicy(model="test-model", client=client, temperature=0.7)
    msgs = [{"role": "user", "content": "go"}]
    out = policy.generate(msgs)
    assert out == '{"choice": 0}'
    assert client.chat.completions.seen["model"] == "test-model"
    assert client.chat.completions.seen["messages"] == msgs
    assert client.chat.completions.seen["temperature"] == 0.7

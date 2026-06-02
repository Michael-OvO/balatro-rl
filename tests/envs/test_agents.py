import numpy as np
from balatro_rl.engine.engine import reset, legal_actions, Verb
from balatro_rl.envs.actions import legal_mask, decode
from balatro_rl.envs.agents import RandomAgent, GreedyAgent


def test_random_agent_picks_legal():
    s = reset(seed=1)
    mask = legal_mask(s)
    agent = RandomAgent(seed=7)
    for _ in range(50):
        a = agent.act(s, mask)
        assert mask[a]


def test_greedy_agent_plays_highest_scoring_hand():
    # Greedy should pick a PLAY action that scores at least as high as any other PLAY.
    from balatro_rl.engine.scoring import score_play
    s = reset(seed=2)
    agent = GreedyAgent()
    a = agent.act(s, legal_mask(s))
    verb, arg = decode(a)
    assert verb in (Verb.PLAY, Verb.DISCARD)
    if verb == Verb.PLAY:
        chosen = score_play([s.hand[i] for i in arg]).score
        best = max(score_play([s.hand[i] for i in c]).score
                   for v, c in legal_actions(s) if v == Verb.PLAY)
        assert chosen == best


def test_greedy_agent_in_shop_is_legal():
    import dataclasses
    from balatro_rl.engine.cards import Card
    from balatro_rl.engine.engine import step
    s = reset(seed=1)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3),
            Card(2, 0), Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, required=10, money=100)
    s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))     # -> SHOP
    a = GreedyAgent().act(s, legal_mask(s))
    assert legal_mask(s)[a]

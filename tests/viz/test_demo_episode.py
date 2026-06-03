"""The scripted demo episode showcases the enriched viewer content (mods, boss, jokers with
descriptions, score breakdown, a Planet level-up) without needing a trained agent."""
from balatro_rl.viz.demo_episode import build_demo_episode


def test_demo_has_three_steps_with_full_schema():
    steps = build_demo_episode()
    assert len(steps) == 3
    for s in steps:
        for k in ("boss", "consumables", "score_trace", "jokers", "hand"):
            assert k in s, k


def test_demo_step0_is_a_boss_play_with_rich_trace():
    s0 = build_demo_episode()[0]
    assert s0["verb"] == "PLAY"
    assert s0["boss"]["name"] == "The Club" and s0["boss"]["desc"]
    # jokers carry effect descriptions
    assert all(j["desc"] for j in s0["jokers"])
    # the breakdown starts at the hand base and ends at the final score
    labels = [e["label"] for e in s0["score_trace"]]
    assert labels[0].startswith("PAIR base")
    assert any("Glass" in l for l in labels) and any(l == "JOKER" for l in labels)
    # the debuffed Club King is inert -> it never appears as a +chips contribution
    assert s0["score"] == int(s0["score_trace"][-1]["chips"] * s0["score_trace"][-1]["mult"])


def test_demo_planet_levels_the_pair():
    steps = build_demo_episode()
    assert steps[1]["verb"] == "USE"
    assert steps[1]["consumables"] and steps[1]["consumables"][0]["name"] == "Mercury"
    assert "(lvl 2)" in steps[2]["score_trace"][0]["label"]

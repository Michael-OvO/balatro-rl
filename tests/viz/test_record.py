from balatro_rl.viz.record import record_demo
from balatro_rl.viz.replay_data import load_episode, replay_states


def test_record_demo_trains_and_writes_episode(tmp_path):
    out = tmp_path / "episode.json"
    steps = record_demo(out_path=str(out), train_updates=2, seed=0,
                        num_envs=4, num_steps=16, d_model=32)
    assert len(steps) > 0
    loaded = load_episode(out)
    assert [s["action_id"] for s in loaded] == [s["action_id"] for s in steps]
    # recorded actions replay back to a terminal state
    assert replay_states(0, [s["action_id"] for s in loaded])[-1].done

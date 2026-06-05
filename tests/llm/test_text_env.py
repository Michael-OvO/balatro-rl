from balatro_rl.llm.text_env import BalatroTextEnv
from tests.llm.test_integration import ScriptedStubPolicy


def test_reset_returns_text_obs_and_info():
    env = BalatroTextEnv(reward_name="shaped")
    obs, info = env.reset(seed=0, req_scale=1.0)
    assert isinstance(obs, str) and "Ante 1" in obs and "Legal actions" in obs
    assert info["is_action_valid"] is True
    assert info["ante"] == 1 and info["won"] is False


def test_invalid_action_does_not_step_and_flags_invalid():
    env = BalatroTextEnv(reward_name="shaped")
    before, _ = env.reset(seed=0)
    obs, reward, done, info = env.step("I will not emit JSON")
    assert info["is_action_valid"] is False
    assert reward == 0.0
    assert obs == before                       # same observation re-presented (no engine step)


def test_full_game_runs_through_text_interface_with_low_req_scale():
    # Stub plays single cards; at a tiny req_scale it clears trivial blinds and advances antes,
    # exercising the full multi-turn text loop (serialize -> menu -> parse -> step) end to end.
    env = BalatroTextEnv(reward_name="shaped")
    policy = ScriptedStubPolicy()
    obs, _ = env.reset(seed=0, req_scale=0.001)
    done, steps, final_ante, cleared_any = False, 0, 1, False
    while not done and steps < 2000:
        action = policy.generate([{"role": "user", "content": obs}])
        obs, _reward, done, info = env.step(action)
        final_ante = info["ante"]
        cleared_any = cleared_any or info["cleared"]
        steps += 1
    assert done and steps > 0
    assert final_ante >= 2                      # cleared ante 1 -> passed through shops
    assert cleared_any                          # info["cleared"] latches once a blind is cleared


def test_cleared_signal_is_false_before_any_blind_cleared():
    env = BalatroTextEnv(reward_name="shaped")
    _, info = env.reset(seed=0, req_scale=1.0)
    assert info["cleared"] is False             # fresh game, nothing cleared yet


def test_curriculum_req_scale_is_applied_at_reset():
    env = BalatroTextEnv(reward_name="shaped")
    env.reset(seed=3, req_scale=0.2)
    assert env._env._req_scale == 0.2           # the knob reached the underlying BalatroEnv

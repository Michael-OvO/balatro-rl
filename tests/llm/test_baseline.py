from balatro_rl.llm.baseline import run_baseline, BaselineReport
from tests.llm.test_integration import ScriptedStubPolicy


def test_run_baseline_aggregates_win_rate_and_ante_depth():
    report = run_baseline(ScriptedStubPolicy(), seeds=[0, 1, 2], reward_name="shaped")
    assert isinstance(report, BaselineReport)
    assert report.games == 3
    assert 0.0 <= report.win_rate <= 1.0
    assert report.mean_final_ante >= 1.0
    assert len(report.trajectories) == 3

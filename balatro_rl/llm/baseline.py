"""Frozen-baseline eval: run the LLM agent over a set of seeds on the existing env +
runner, and report Ante-8 win rate and ante-depth -- the M1 go/no-go gate. No training.

CLI:  python -m balatro_rl.llm.baseline --model <name> --base-url <url> --seeds 0-31
"""
from __future__ import annotations

import argparse
import dataclasses

from ..envs.balatro_env import BalatroEnv
from ..envs.runner import Trajectory, run_episode
from .agent import LLMAgent
from .policy_client import FrozenEndpointPolicy


@dataclasses.dataclass
class BaselineReport:
    games: int
    win_rate: float
    mean_final_ante: float
    trajectories: list[Trajectory]


def run_baseline(policy, seeds, reward_name: str = "shaped",
                 window_turns: int = 12) -> BaselineReport:
    trajectories: list[Trajectory] = []
    for seed in seeds:
        env = BalatroEnv(reward_name=reward_name)
        agent = LLMAgent(policy=policy, window_turns=window_turns)
        trajectories.append(run_episode(env, agent, seed=seed))
    wins = sum(1 for t in trajectories if t.won)
    n = len(trajectories)
    return BaselineReport(
        games=n,
        win_rate=wins / n if n else 0.0,
        mean_final_ante=sum(t.final_ante for t in trajectories) / n if n else 0.0,
        trajectories=trajectories,
    )


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen LLM baseline for Balatro (M1 gate).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--seeds", default="0-31")
    ap.add_argument("--reward-name", default="shaped")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    policy = FrozenEndpointPolicy(model=args.model, base_url=args.base_url,
                                  api_key=args.api_key, temperature=args.temperature)
    report = run_baseline(policy, seeds=_parse_seeds(args.seeds), reward_name=args.reward_name)
    print(f"games={report.games} win_rate={report.win_rate:.3f} "
          f"mean_final_ante={report.mean_final_ante:.2f}")


if __name__ == "__main__":
    main()

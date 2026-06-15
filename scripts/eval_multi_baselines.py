from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.agents.multi_ugv_baselines import GreedyMultiUGVAgent, RandomMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml


class MultiPolicy(Protocol):
    def predict(self, env: FireRescueMultiUGVEnv) -> int:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-UGV baseline policies.")
    parser.add_argument("--config", default="configs/env_multi_ugv.yaml")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/eval/metrics_csv")
    return parser.parse_args()


def run_episode(
    env: FireRescueMultiUGVEnv,
    agent: MultiPolicy,
    *,
    agent_name: str,
    episode: int,
    seed: int,
) -> dict[str, float | int | str]:
    _, info = env.reset(seed=seed)
    total_reward = 0.0
    invalid_uav = 0
    invalid_ugv = 0
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        action = agent.predict(env)
        _, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        invalid_uav += int(bool(final_info.get("invalid_uav", False)))
        invalid_ugv += int(final_info.get("invalid_ugv_count", 0))

    reason = str(final_info.get("terminated_reason", "timeout" if truncated else ""))
    uav_len = _path_length(final_info.get("uav_path", []))
    ugv_paths = final_info.get("ugv_paths", [])
    ugv_len = sum(_path_length(path) for path in ugv_paths)
    return {
        "agent": agent_name,
        "episode": episode,
        "episode_seed": seed,
        "map_seed": int(final_info["map_seed"]),
        "success": int(bool(final_info["success"])),
        "reward": round(total_reward, 6),
        "steps": int(final_info["step_count"]),
        "delivered_count": int(final_info["delivered_count"]),
        "discovered_count": int(final_info["discovered_count"]),
        "burned_count": int(final_info["burned_count"]),
        "total_survivors": int(final_info["total_survivors"]),
        "risk_exposure": round(float(final_info["risk_exposure"]), 6),
        "timeout": int(reason == "timeout" or truncated),
        "fire_collision": int(reason == "ugv_fire_collision"),
        "survivor_burned": int(reason == "survivor_burned"),
        "invalid_uav": invalid_uav,
        "invalid_ugv": invalid_ugv,
        "uav_path_length": uav_len,
        "ugv_path_length": ugv_len,
        "terminated_reason": reason,
    }


def _path_length(path) -> int:
    if len(path) < 2:
        return 0
    return sum(1 for a, b in zip(path[:-1], path[1:]) if tuple(a) != tuple(b))


def summarize(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = [
        "success",
        "reward",
        "steps",
        "delivered_count",
        "discovered_count",
        "burned_count",
        "risk_exposure",
        "timeout",
        "fire_collision",
        "survivor_burned",
        "invalid_uav",
        "invalid_ugv",
        "uav_path_length",
        "ugv_path_length",
    ]
    return df.groupby("agent", as_index=False)[metrics].mean()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    env = FireRescueMultiUGVEnv.from_config(config, render_mode="rgb_array")
    policies: list[tuple[str, MultiPolicy]] = [
        ("random", RandomMultiUGVAgent(seed=args.seed + 100)),
        ("greedy", GreedyMultiUGVAgent()),
        ("coverage_astar", CoverageAStarMultiUGVAgent()),
        ("astar", AStarMultiUGVAgent()),
    ]

    rows: list[dict[str, float | int | str]] = []
    for agent_name, agent in policies:
        for episode in range(args.episodes):
            rows.append(
                run_episode(
                    env,
                    agent,
                    agent_name=agent_name,
                    episode=episode,
                    seed=args.seed + episode,
                )
            )

    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    detail_path = out_dir / "multi_baselines_detail.csv"
    summary_path = out_dir / "multi_baselines_summary.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.to_csv(summary_path, index=False)
    print("Saved detail CSV:", detail_path.relative_to(ROOT))
    print("Saved summary CSV:", summary_path.relative_to(ROOT))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

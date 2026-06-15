from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A* on FireRescueMultiUGVEnv.")
    parser.add_argument("--config", default="configs/env_multi_ugv.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/eval/metrics_csv/multi_astar_check.csv")
    parser.add_argument("--agent", choices=["astar", "coverage"], default="astar")
    return parser.parse_args()


def run_episode(env: FireRescueMultiUGVEnv, agent: AStarMultiUGVAgent, seed: int) -> dict[str, float | int | str]:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        action = agent.predict(env)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
    return {
        "episode_seed": seed,
        "map_seed": int(final_info["map_seed"]),
        "success": int(bool(final_info["success"])),
        "steps": int(final_info["step_count"]),
        "reward": round(total_reward, 6),
        "delivered_count": int(final_info["delivered_count"]),
        "discovered_count": int(final_info["discovered_count"]),
        "burned_count": int(final_info["burned_count"]),
        "total_survivors": int(final_info["total_survivors"]),
        "risk_exposure": round(float(final_info["risk_exposure"]), 6),
        "terminated_reason": str(final_info.get("terminated_reason", "")),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    env = FireRescueMultiUGVEnv.from_config(config, render_mode="rgb_array")
    agent = CoverageAStarMultiUGVAgent() if args.agent == "coverage" else AStarMultiUGVAgent()
    rows = [run_episode(env, agent, args.seed + idx) for idx in range(args.episodes)]

    out_path = ROOT / args.out
    ensure_dirs(out_path.parent)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(int(row["success"]) for row in rows)
    avg_steps = sum(int(row["steps"]) for row in rows) / len(rows)
    avg_reward = sum(float(row["reward"]) for row in rows) / len(rows)
    avg_delivered = sum(int(row["delivered_count"]) for row in rows) / len(rows)
    print(f"{args.agent} episodes: {len(rows)}")
    print(f"Success rate: {successes}/{len(rows)} = {successes / len(rows):.2%}")
    print(f"Average delivered: {avg_delivered:.2f}/{rows[0]['total_survivors']}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.2f}")
    print("Saved CSV:", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fire_rescue_rl.agents.astar_team import AStarTeamAgent
from fire_rescue_rl.agents.greedy_team import GreedyTeamAgent, RandomTeamAgent
from fire_rescue_rl.envs.fire_rescue_env import FireRescueMAEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.metrics import MetricsAccumulator, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-rl", action="store_true")
    return parser.parse_args()


def rollout(env: FireRescueMAEnv, agent, model_agent: bool, seed: int):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    infos = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if model_agent:
            action = int(agent.predict(obs, deterministic=True)[0])
        else:
            action = int(agent.predict(env))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        infos.append(info)
    return {"reward": total_reward, "steps": env.step_count, "infos": infos, "truncated": truncated}


def main() -> None:
    args = parse_args()
    config = load_yaml(f"configs/env_{args.difficulty}.yaml")
    agents = [
        ("random", RandomTeamAgent(args.seed), False),
        ("greedy", GreedyTeamAgent(), False),
        ("astar", AStarTeamAgent(), False),
    ]
    if args.include_rl:
        from stable_baselines3 import DQN

        best = ROOT / "outputs" / "models" / "dqn" / f"dqn_{args.difficulty}_seed{args.seed}_best.zip"
        last = ROOT / "outputs" / "models" / "dqn" / f"dqn_{args.difficulty}_seed{args.seed}.zip"
        model_path = best if best.exists() else last
        if model_path.exists():
            agents.append(("dqn", DQN.load(model_path), True))

    out_dir = ROOT / "outputs" / "eval" / "metrics_csv"
    ensure_dirs(out_dir)
    all_frames = []
    for agent_name, agent, model_agent in agents:
        acc = MetricsAccumulator(agent_name, args.difficulty)
        for episode in range(args.episodes):
            env = FireRescueMAEnv.from_config(config)
            result = rollout(env, agent, model_agent, args.seed + episode)
            acc.add_episode(episode, result)
        frame = acc.to_frame()
        frame.to_csv(out_dir / f"{agent_name}_{args.difficulty}_seed{args.seed}.csv", index=False)
        all_frames.append(frame)
    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(out_dir / f"all_{args.difficulty}_seed{args.seed}.csv", index=False)
    summary = summarize_metrics(full)
    summary.to_csv(out_dir / f"summary_{args.difficulty}_seed{args.seed}.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


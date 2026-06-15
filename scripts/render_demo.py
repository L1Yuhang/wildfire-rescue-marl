from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_rescue_rl.agents.astar_team import AStarTeamAgent
from fire_rescue_rl.agents.greedy_team import GreedyTeamAgent, RandomTeamAgent
from fire_rescue_rl.envs.fire_rescue_env import FireRescueMAEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.video import save_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="astar", choices=["random", "greedy", "astar", "dqn"])
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--format", default="gif", choices=["gif", "mp4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def load_agent(name: str, difficulty: str, seed: int):
    if name == "random":
        return RandomTeamAgent(seed)
    if name == "greedy":
        return GreedyTeamAgent()
    if name == "astar":
        return AStarTeamAgent()
    from stable_baselines3 import DQN

    best = ROOT / "outputs" / "models" / "dqn" / f"dqn_{difficulty}_seed{seed}_best.zip"
    last = ROOT / "outputs" / "models" / "dqn" / f"dqn_{difficulty}_seed{seed}.zip"
    return DQN.load(best if best.exists() else last)


def main() -> None:
    args = parse_args()
    config = load_yaml(f"configs/env_{args.difficulty}.yaml")
    env = FireRescueMAEnv.from_config(config, render_mode="rgb_array")
    agent = load_agent(args.agent, args.difficulty, args.seed)
    obs, info = env.reset(seed=args.seed)
    frames = [env.render()]
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if args.agent == "dqn":
            action = int(agent.predict(obs, deterministic=True)[0])
        else:
            action = int(agent.predict(env))
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
    out_dir = ROOT / "outputs" / "videos"
    ensure_dirs(out_dir)
    name = args.tag or f"{args.agent}_{args.difficulty}_demo"
    output = out_dir / f"{name}.{args.format}"
    save_frames(frames, output, fps=args.fps)
    print(f"Saved demo: {output}")
    print(f"steps={env.step_count}, reward={env.episode_reward:.2f}, success={info.get('success')}, stage={env.stage_name()}")


if __name__ == "__main__":
    main()

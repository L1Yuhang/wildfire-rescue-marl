from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.agents.multi_ugv_baselines import GreedyMultiUGVAgent, RandomMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.envs.generated_maps import save_generated_map_preview
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.video import save_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-UGV policy demo.")
    parser.add_argument("--config", default="configs/env_multi_easy.yaml")
    parser.add_argument(
        "--agent",
        choices=[
            "random",
            "greedy",
            "astar",
            "coverage",
            "dqn",
            "factorized",
            "factorized_hybrid",
            "factorized_reverse_hybrid",
        ],
        default="astar",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def experiment_name(env_config: dict, config_path: str) -> str:
    if env_config.get("experiment_name"):
        return str(env_config["experiment_name"])
    return Path(config_path).stem.replace("env_", "")


class DQNPolicy:
    def __init__(self, model) -> None:
        self.model = model

    def predict_obs(self, obs) -> int:
        return int(self.model.predict(obs, deterministic=True)[0])


def load_agent(agent_name: str, name: str, seed: int, model_path: str):
    if agent_name == "random":
        return RandomMultiUGVAgent(seed)
    if agent_name == "greedy":
        return GreedyMultiUGVAgent()
    if agent_name == "astar":
        return AStarMultiUGVAgent()
    if agent_name == "coverage":
        return CoverageAStarMultiUGVAgent()
    if agent_name in {"factorized", "factorized_hybrid", "factorized_reverse_hybrid"}:
        from fire_rescue_rl.agents.factorized_q_policy import FactorizedDQNPolicy

        if not model_path:
            raise ValueError("--model-path is required for --agent factorized")
        return FactorizedDQNPolicy.load(model_path)
    from stable_baselines3 import DQN

    if model_path:
        path = Path(model_path)
    else:
        bc = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_{name}_seed{seed}_bc.zip"
        best = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_{name}_seed{seed}_best.zip"
        last = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_{name}_seed{seed}.zip"
        if bc.exists():
            path = bc
        elif best.exists():
            path = best
        else:
            path = last
    if not path.exists():
        raise FileNotFoundError(f"DQN model not found: {path}")
    return DQNPolicy(DQN.load(path))


def main() -> None:
    args = parse_args()
    env_config = load_yaml(args.config)
    name = experiment_name(env_config, args.config)
    env = FireRescueMultiUGVEnv.from_config(env_config, render_mode="rgb_array")
    agent = load_agent(args.agent, name, args.seed, args.model_path)
    obs, info = env.reset(seed=args.seed)

    ensure_dirs(ROOT / "outputs" / "figures" / "multi_ugv", ROOT / "outputs" / "videos")
    if env.rescue_map is None:
        raise RuntimeError("env.reset() did not create a rescue map")
    preview_path = ROOT / "outputs" / "figures" / "multi_ugv" / f"{name}_map_seed{args.seed}.png"
    save_generated_map_preview(env.rescue_map, preview_path, title=f"{name}: Generated Rescue Map")

    frames = [env.render()]
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        if args.agent == "factorized_hybrid":
            action = agent.predict_uav_with_astar_ugv(env)
        elif args.agent == "factorized_reverse_hybrid":
            action = agent.predict_astar_uav_with_ugv(env)
        elif args.agent in {"dqn", "factorized"}:
            if hasattr(agent, "predict_env"):
                action = agent.predict_env(env)
            else:
                action = agent.predict_obs(obs)
        else:
            action = int(agent.predict(env))
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        if env.step_count % max(1, args.frame_skip) == 0 or terminated or truncated:
            frames.append(env.render())

    tag = args.tag or f"multi_{args.agent}_{name}_seed{args.seed}"
    output = ROOT / "outputs" / "videos" / f"{tag}.{args.format}"
    save_frames(frames, output, fps=args.fps)
    print("Saved map preview:", preview_path.relative_to(ROOT))
    print("Saved demo:", output.relative_to(ROOT))
    print(
        "Final:",
        {
            "success": final_info["success"],
            "steps": final_info["step_count"],
            "delivered": f"{final_info['delivered_count']}/{final_info['total_survivors']}",
            "reward": round(total_reward, 3),
            "risk": round(float(final_info["risk_exposure"]), 3),
        },
    )


if __name__ == "__main__":
    main()

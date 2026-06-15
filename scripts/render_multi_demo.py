from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.envs.generated_maps import save_generated_map_preview
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.video import save_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-UGV generated map and A* demo.")
    parser.add_argument("--config", default="configs/env_multi_ugv.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--out-prefix", default="astar_multi_ugv_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    env = FireRescueMultiUGVEnv.from_config(config, render_mode="rgb_array")
    agent = AStarMultiUGVAgent()
    obs, info = env.reset(seed=args.seed)

    ensure_dirs(ROOT / "outputs" / "figures", ROOT / "outputs" / "videos")
    preview_path = ROOT / "outputs" / "figures" / f"multi_generated_map_seed{args.seed}.png"
    if env.rescue_map is None:
        raise RuntimeError("env.reset() did not create a rescue map")
    save_generated_map_preview(
        env.rescue_map,
        preview_path,
        title="Generated Fire Rescue Map: 1 UAV + 2 UGV + 3 Survivors",
    )

    frames = [env.render()]
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        action = agent.predict(env)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        if env.step_count % max(1, args.frame_skip) == 0 or terminated or truncated:
            frames.append(env.render())

    video_path = ROOT / "outputs" / "videos" / f"{args.out_prefix}.{args.format}"
    save_frames(frames, video_path, fps=args.fps)
    print("Saved map preview:", preview_path.relative_to(ROOT))
    print("Saved A* demo:", video_path.relative_to(ROOT))
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

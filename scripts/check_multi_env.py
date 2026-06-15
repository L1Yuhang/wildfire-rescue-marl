from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v2 as imageio
import numpy as np

from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FireRescueMultiUGVEnv.")
    parser.add_argument("--config", default="configs/env_multi_ugv.yaml")
    parser.add_argument("--frame-name", default="multi_env_check_frame.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    env = FireRescueMultiUGVEnv.from_config(config, render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print("Environment: FireRescueMultiUGVEnv")
    print("Map seed:", info["map_seed"])
    print("Observation shape:", obs.shape)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Survivors:", [item["position"] for item in info["survivors"]])
    print("Fire sources:", sorted(info["fire_cells"]))

    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True)
        print("Stable-Baselines3 check_env: OK")
    except ImportError:
        print("Stable-Baselines3 not installed; skipped check_env.")

    total_reward = 0.0
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=1)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(0, env.action_space.n)))
        total_reward += reward
        if terminated or truncated:
            break

    ensure_dirs(ROOT / "outputs" / "figures")
    frame_path = ROOT / "outputs" / "figures" / args.frame_name
    imageio.imwrite(frame_path, env.render())
    print("Random rollout reward:", round(total_reward, 3))
    print("Saved frame:", frame_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

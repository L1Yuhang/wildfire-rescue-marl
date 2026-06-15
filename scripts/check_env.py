from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import imageio.v2 as imageio
import numpy as np

from fire_rescue_rl.envs.fire_rescue_env import FireRescueMAEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml


def main() -> None:
    config = load_yaml("configs/env_easy.yaml")
    env = FireRescueMAEnv.from_config(config, render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print("Environment:", env.difficulty)
    print("Observation shape:", obs.shape)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True)
        print("Stable-Baselines3 check_env: OK")
    except ImportError:
        print("Stable-Baselines3 not installed; skipped check_env.")

    total_reward = 0.0
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=1)
    for _ in range(15):
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(0, env.action_space.n)))
        total_reward += reward
        if terminated or truncated:
            break
    frame = env.render()
    ensure_dirs(ROOT / "outputs" / "figures")
    imageio.imwrite(ROOT / "outputs" / "figures" / "env_check_frame.png", frame)
    print("Random rollout reward:", round(total_reward, 3))
    print("Saved frame: outputs/figures/env_check_frame.png")


if __name__ == "__main__":
    main()


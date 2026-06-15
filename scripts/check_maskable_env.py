from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks

from fire_rescue_rl.envs.ppo_wrappers import make_ppo_env
from fire_rescue_rl.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/env_multi_hard_random_guided.yaml")
    for mode in ["uav", "ugv", "full"]:
        env = make_ppo_env(cfg, mode)
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        counts: list[int] = []
        for _ in range(30):
            mask = get_action_masks(env)
            counts.append(int(mask.sum()))
            if mode == "uav":
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                parts = []
                n_parts = 2 if mode == "ugv" else 3
                for idx in range(n_parts):
                    submask = mask[idx * 5 : (idx + 1) * 5]
                    parts.append(int(rng.choice(np.flatnonzero(submask))))
                action = np.asarray(parts, dtype=np.int64)
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        print(mode, "mask_shape", get_action_masks(env).shape, "min_valid", min(counts), "max_valid", max(counts))


if __name__ == "__main__":
    main()

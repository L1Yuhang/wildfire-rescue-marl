from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from fire_rescue_rl.envs.generated_maps import save_generated_map_preview
from fire_rescue_rl.envs.ppo_wrappers import make_ppo_env
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.video import save_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MaskablePPO rescue demo.")
    parser.add_argument("--config", default="configs/env_multi_hard_random_guided.yaml")
    parser.add_argument("--mode", choices=["uav", "ugv", "full"], default="uav")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-kind", choices=["bc", "best", "final"], default="best")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def experiment_name(env_config: dict, config_path: str) -> str:
    if env_config.get("experiment_name"):
        return str(env_config["experiment_name"])
    return Path(config_path).stem.replace("env_", "")


def resolve_model_path(name: str, mode: str, seed: int, model_path: str, model_kind: str) -> Path:
    if model_path:
        return Path(model_path)
    model_dir = ROOT / "outputs" / "models" / "maskable_ppo_multi"
    candidates = {
        "bc": model_dir / f"maskppo_{name}_{mode}_seed{seed}_bc.zip",
        "best": model_dir / f"maskppo_{name}_{mode}_seed{seed}_best.zip",
        "final": model_dir / f"maskppo_{name}_{mode}_seed{seed}.zip",
    }
    path = candidates[model_kind]
    if not path.exists() and model_kind == "best":
        path = candidates["final"] if candidates["final"].exists() else candidates["bc"]
    if not path.exists():
        raise FileNotFoundError(f"MaskablePPO model not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    env_config = load_yaml(args.config)
    name = experiment_name(env_config, args.config)
    env = make_ppo_env(env_config, args.mode, render_mode="rgb_array", safe_projection=True)
    model_path = resolve_model_path(name, args.mode, 0, args.model_path, args.model_kind)
    model = MaskablePPO.load(model_path)
    obs, info = env.reset(seed=args.seed)

    ensure_dirs(ROOT / "outputs" / "figures" / "multi_ugv", ROOT / "outputs" / "videos")
    base_env = env.unwrapped
    if base_env.rescue_map is None:
        raise RuntimeError("env.reset() did not create a rescue map")
    preview_path = ROOT / "outputs" / "figures" / "multi_ugv" / f"maskppo_{name}_{args.mode}_map_seed{args.seed}.png"
    save_generated_map_preview(base_env.rescue_map, preview_path, title=f"MaskablePPO {args.mode}: Generated Rescue Map")

    frames = [env.render()]
    total_reward = 0.0
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        masks = get_action_masks(env)
        action, _state = model.predict(obs, deterministic=True, action_masks=masks)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        if base_env.step_count % max(1, args.frame_skip) == 0 or terminated or truncated:
            frames.append(env.render())

    tag = args.tag or f"maskppo_{name}_{args.mode}_seed{args.seed}"
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

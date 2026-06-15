from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.agents.multi_ugv_baselines import GreedyMultiUGVAgent, RandomMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.envs.ppo_wrappers import make_ppo_env
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO variants and baselines.")
    parser.add_argument("--config", default="configs/env_multi_hard_random_guided.yaml")
    parser.add_argument("--mode", choices=["uav", "ugv", "full"], default="uav")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-kind", choices=["bc", "best", "final"], default="best")
    parser.add_argument("--include-maskppo", action="store_true")
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def experiment_name(env_config: dict, config_path: str) -> str:
    if env_config.get("experiment_name"):
        return str(env_config["experiment_name"])
    return Path(config_path).stem.replace("env_", "")


class MaskablePolicy:
    def __init__(self, model: MaskablePPO, mode: str, env_config: dict) -> None:
        self.model = model
        self.mode = mode
        self.env_config = env_config

    def run_episode(self, seed: int) -> dict:
        env = make_ppo_env(self.env_config, self.mode, render_mode=None, safe_projection=True)
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        invalid_uav = 0
        invalid_ugv = 0
        terminated = False
        truncated = False
        final_info = info
        while not (terminated or truncated):
            masks = get_action_masks(env)
            action, _state = self.model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward)
            invalid_uav += int(bool(final_info.get("invalid_uav", False)))
            invalid_ugv += int(final_info.get("invalid_ugv_count", 0))
        return episode_metrics(final_info, total_reward, invalid_uav, invalid_ugv)


def run_rule_episode(env_config: dict, agent, seed: int) -> dict:
    env = FireRescueMultiUGVEnv.from_config(env_config)
    _obs, info = env.reset(seed=seed)
    total_reward = 0.0
    invalid_uav = 0
    invalid_ugv = 0
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        action = int(agent.predict(env))
        _obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        invalid_uav += int(bool(final_info.get("invalid_uav", False)))
        invalid_ugv += int(final_info.get("invalid_ugv_count", 0))
    return episode_metrics(final_info, total_reward, invalid_uav, invalid_ugv)


def episode_metrics(final_info: dict, total_reward: float, invalid_uav: int, invalid_ugv: int) -> dict:
    reason = str(final_info.get("terminated_reason", ""))
    return {
        "map_seed": int(final_info["map_seed"]),
        "success": int(bool(final_info["success"])),
        "reward": round(float(total_reward), 6),
        "steps": int(final_info["step_count"]),
        "delivered_count": int(final_info["delivered_count"]),
        "discovered_count": int(final_info["discovered_count"]),
        "burned_count": int(final_info["burned_count"]),
        "total_survivors": int(final_info["total_survivors"]),
        "risk_exposure": round(float(final_info["risk_exposure"]), 6),
        "timeout": int(reason == "timeout" or final_info["step_count"] >= 520),
        "fire_collision": int(reason == "ugv_fire_collision"),
        "survivor_burned": int(reason == "survivor_burned"),
        "invalid_uav": int(invalid_uav),
        "invalid_ugv": int(invalid_ugv),
        "uav_path_length": path_length(final_info.get("uav_path", [])),
        "ugv_path_length": sum(path_length(path) for path in final_info.get("ugv_paths", [])),
        "terminated_reason": reason,
    }


def path_length(path) -> int:
    if len(path) < 2:
        return 0
    return sum(1 for a, b in zip(path[:-1], path[1:]) if tuple(a) != tuple(b))


def summarize(rows: list[dict]) -> pd.DataFrame:
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
    policies: list[tuple[str, object]] = [
        ("random", RandomMultiUGVAgent(seed=args.seed + 500)),
        ("greedy", GreedyMultiUGVAgent()),
        ("coverage_astar", CoverageAStarMultiUGVAgent()),
        ("astar", AStarMultiUGVAgent()),
    ]
    mask_policy: MaskablePolicy | None = None
    if args.include_maskppo:
        model_path = resolve_model_path(name, args.mode, 0, args.model_path, args.model_kind)
        mask_policy = MaskablePolicy(MaskablePPO.load(model_path), args.mode, env_config)

    rows: list[dict] = []
    for agent_name, agent in policies:
        for episode in range(args.episodes):
            metrics = run_rule_episode(env_config, agent, args.seed + episode)
            metrics.update({"agent": agent_name, "episode": episode, "episode_seed": args.seed + episode})
            rows.append(metrics)
    if mask_policy is not None:
        label = f"maskppo_{args.mode}"
        for episode in range(args.episodes):
            metrics = mask_policy.run_episode(args.seed + episode)
            metrics.update({"agent": label, "episode": episode, "episode_seed": args.seed + episode})
            rows.append(metrics)

    out_dir = ROOT / "outputs" / "eval" / "metrics_csv"
    ensure_dirs(out_dir)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    detail_path = out_dir / f"maskppo_all_{name}_{args.mode}_seed{args.seed}{tag}.csv"
    summary_path = out_dir / f"maskppo_summary_{name}_{args.mode}_seed{args.seed}{tag}.csv"
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

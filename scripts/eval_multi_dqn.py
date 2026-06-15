from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.agents.multi_ugv_baselines import GreedyMultiUGVAgent, RandomMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml


class MultiPolicy(Protocol):
    def predict(self, env: FireRescueMultiUGVEnv) -> int:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN and baselines on FireRescueMultiUGVEnv.")
    parser.add_argument("--config", default="configs/env_multi_easy.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--include-rl", action="store_true")
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--policy-type",
        choices=["sb3", "factorized", "factorized_hybrid", "factorized_reverse_hybrid"],
        default="sb3",
    )
    parser.add_argument("--model-kind", choices=["bc", "er", "best", "final"], default="bc")
    parser.add_argument("--rl-label", default="")
    parser.add_argument("--include-coverage", action="store_true")
    parser.add_argument("--output-tag", default="")
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


def run_episode(env: FireRescueMultiUGVEnv, agent, *, is_dqn: bool, episode: int, seed: int, agent_name: str):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    invalid_uav = 0
    invalid_ugv = 0
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        if is_dqn and hasattr(agent, "predict_env"):
            action = int(agent.predict_env(env))
        else:
            action = int(agent.predict_obs(obs) if is_dqn else agent.predict(env))
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        invalid_uav += int(bool(final_info.get("invalid_uav", False)))
        invalid_ugv += int(final_info.get("invalid_ugv_count", 0))
    reason = str(final_info.get("terminated_reason", "timeout" if truncated else ""))
    return {
        "agent": agent_name,
        "episode": episode,
        "episode_seed": seed,
        "map_seed": int(final_info["map_seed"]),
        "success": int(bool(final_info["success"])),
        "reward": round(total_reward, 6),
        "steps": int(final_info["step_count"]),
        "delivered_count": int(final_info["delivered_count"]),
        "discovered_count": int(final_info["discovered_count"]),
        "burned_count": int(final_info["burned_count"]),
        "total_survivors": int(final_info["total_survivors"]),
        "risk_exposure": round(float(final_info["risk_exposure"]), 6),
        "timeout": int(reason == "timeout" or truncated),
        "fire_collision": int(reason == "ugv_fire_collision"),
        "survivor_burned": int(reason == "survivor_burned"),
        "invalid_uav": invalid_uav,
        "invalid_ugv": invalid_ugv,
        "uav_path_length": _path_length(final_info.get("uav_path", [])),
        "ugv_path_length": sum(_path_length(path) for path in final_info.get("ugv_paths", [])),
        "terminated_reason": reason,
    }


def _path_length(path) -> int:
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


def load_dqn_policy(name: str, seed: int, model_path: str, model_kind: str):
    from stable_baselines3 import DQN

    if model_path:
        path = Path(model_path)
    else:
        bc = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_{name}_seed{seed}_bc.zip"
        er_best = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_er_{name}_seed{seed}_best.zip"
        er_final = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_er_{name}_seed{seed}.zip"
        best = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_{name}_seed{seed}_best.zip"
        last = ROOT / "outputs" / "models" / "dqn_multi" / f"dqn_{name}_seed{seed}.zip"
        if model_kind == "er":
            path = er_best if er_best.exists() else er_final
        elif model_kind == "best":
            path = best
        elif model_kind == "final":
            path = last
        elif bc.exists():
            path = bc
        elif best.exists():
            path = best
        else:
            path = last
    if not path.exists():
        raise FileNotFoundError(f"DQN model not found: {path}")
    return DQNPolicy(DQN.load(path))


def load_factorized_policy(model_path: str):
    from fire_rescue_rl.agents.factorized_q_policy import FactorizedDQNPolicy

    if not model_path:
        raise ValueError("--model-path is required when --policy-type factorized")
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Factorized model not found: {path}")
    return FactorizedDQNPolicy.load(path)


class HybridFactorizedPolicy:
    def __init__(self, base_policy) -> None:
        self.base_policy = base_policy

    def predict_env(self, env) -> int:
        return int(self.base_policy.predict_uav_with_astar_ugv(env))


class ReverseHybridFactorizedPolicy:
    def __init__(self, base_policy) -> None:
        self.base_policy = base_policy

    def predict_env(self, env) -> int:
        return int(self.base_policy.predict_astar_uav_with_ugv(env))


def main() -> None:
    args = parse_args()
    env_config = load_yaml(args.config)
    name = experiment_name(env_config, args.config)
    policies: list[tuple[str, object, bool]] = [
        ("random", RandomMultiUGVAgent(seed=args.seed + 500), False),
        ("greedy", GreedyMultiUGVAgent(), False),
        ("astar", AStarMultiUGVAgent(), False),
    ]
    if args.include_coverage:
        policies.insert(2, ("coverage_astar", CoverageAStarMultiUGVAgent(), False))
    if args.include_rl:
        label = args.rl_label or f"dqn_{args.model_kind}"
        if args.policy_type in {"factorized", "factorized_hybrid", "factorized_reverse_hybrid"}:
            policy = load_factorized_policy(args.model_path)
            if args.policy_type == "factorized_hybrid":
                policy = HybridFactorizedPolicy(policy)
            elif args.policy_type == "factorized_reverse_hybrid":
                policy = ReverseHybridFactorizedPolicy(policy)
            policies.append((label, policy, True))
        else:
            policies.append((label, load_dqn_policy(name, args.seed, args.model_path, args.model_kind), True))

    rows = []
    for agent_name, agent, is_dqn in policies:
        for episode in range(args.episodes):
            env = FireRescueMultiUGVEnv.from_config(env_config)
            rows.append(
                run_episode(
                    env,
                    agent,
                    is_dqn=is_dqn,
                    episode=episode,
                    seed=args.seed + episode,
                    agent_name=agent_name,
                )
            )

    out_dir = ROOT / "outputs" / "eval" / "metrics_csv"
    ensure_dirs(out_dir)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    detail_path = out_dir / f"multi_all_{name}_seed{args.seed}{tag}.csv"
    summary_path = out_dir / f"multi_summary_{name}_seed{args.seed}{tag}.csv"
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

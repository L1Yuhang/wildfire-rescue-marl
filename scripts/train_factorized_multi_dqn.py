from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch as th
import torch.nn.functional as F

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.agents.factorized_q_policy import FactorizedDQNPolicy, FactorizedQNetwork
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train factorized multi-agent Q heads with BC + DAgger.")
    parser.add_argument("--config", default="configs/env_multi_hard_random_explore.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert", choices=["astar", "coverage"], default="coverage")
    parser.add_argument("--control-mode", choices=["all", "ugv"], default="all")
    parser.add_argument("--initial-episodes", type=int, default=300)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--dagger-iters", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=int, default=60)
    parser.add_argument("--dagger-train-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    return parser.parse_args()


def experiment_name(env_config: dict, config_path: str) -> str:
    if env_config.get("experiment_name"):
        return str(env_config["experiment_name"])
    return Path(config_path).stem.replace("env_", "")


def build_expert(name: str):
    if name == "coverage":
        return CoverageAStarMultiUGVAgent()
    return AStarMultiUGVAgent()


def collect_dataset(
    env_config: dict,
    *,
    episodes: int,
    seed: int,
    expert_name: str,
    control_mode: str,
    rollin_policy: FactorizedDQNPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    expert = build_expert(expert_name)
    observations: list[np.ndarray] = []
    action_parts: list[list[int]] = []
    for episode in range(episodes):
        env = FireRescueMultiUGVEnv.from_config(env_config)
        obs, _info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            expert_action = int(expert.predict(env))
            observations.append(obs)
            decoded_expert = env.decode_action(expert_action)
            if control_mode == "ugv":
                action_parts.append(decoded_expert[1:])
            else:
                action_parts.append(decoded_expert)

            if rollin_policy is None:
                action = expert_action
            elif control_mode == "ugv":
                action = int(rollin_policy.predict_astar_uav_with_ugv(env))
            else:
                action = int(rollin_policy.predict_obs(obs))
            obs, _reward, terminated, truncated, _info = env.step(action)
    return np.asarray(observations, dtype=np.float32), np.asarray(action_parts, dtype=np.int64)


def train_bc(
    model: FactorizedQNetwork,
    observations: np.ndarray,
    action_parts: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
    act_tensor = th.as_tensor(action_parts, dtype=th.long, device=device)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    joint_accs: list[float] = []
    head_accs: list[float] = []
    model.train()
    batch = min(batch_size, len(observations))
    for _ in range(steps):
        indices = rng.integers(0, len(observations), size=batch)
        logits = model(obs_tensor[indices])
        loss = sum(F.cross_entropy(head, act_tensor[indices, idx]) for idx, head in enumerate(logits))
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        with th.no_grad():
            preds = th.stack([head.argmax(dim=1) for head in logits], dim=1)
            target = act_tensor[indices]
            head_accs.append(float((preds == target).float().mean().detach().cpu()))
            joint_accs.append(float((preds == target).all(dim=1).float().mean().detach().cpu()))
            losses.append(float(loss.detach().cpu()))

    return {
        "loss": losses[-1] if losses else 0.0,
        "head_acc": float(np.mean(head_accs[-100:])) if head_accs else 0.0,
        "joint_acc": float(np.mean(joint_accs[-100:])) if joint_accs else 0.0,
    }


def save_checkpoint(
    policy: FactorizedDQNPolicy,
    path: Path,
    *,
    obs_dim: int,
    hidden_sizes: tuple[int, ...],
) -> None:
    policy.save(path, obs_dim=obs_dim, hidden_sizes=hidden_sizes)


def save_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env_config = load_yaml(args.config)
    name = experiment_name(env_config, args.config)
    env = FireRescueMultiUGVEnv.from_config(env_config)
    obs_dim = int(env.observation_space.shape[0])
    num_agents = int(env.num_ugv) if args.control_mode == "ugv" else 1 + int(env.num_ugv)
    hidden_sizes = (int(args.hidden_size), int(args.hidden_size))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = FactorizedQNetwork(obs_dim=obs_dim, num_agents=num_agents, hidden_sizes=hidden_sizes).to(device)
    policy = FactorizedDQNPolicy(model, device=device)

    log_dir = ROOT / "outputs" / "logs" / "factorized_multi" / f"{name}_seed{args.seed}_{args.expert}_{args.control_mode}"
    model_dir = ROOT / "outputs" / "models" / "factorized_multi"
    ensure_dirs(log_dir, model_dir)
    metrics_path = log_dir / "training_metrics.csv"
    rows: list[dict[str, float | int | str]] = []

    observations, actions = collect_dataset(
        env_config,
        episodes=args.initial_episodes,
        seed=args.seed,
        expert_name=args.expert,
        control_mode=args.control_mode,
    )
    stats = train_bc(
        model,
        observations,
        actions,
        steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    rows.append({"stage": "initial_bc", "samples": len(actions), **stats})
    stage_path = model_dir / f"factorized_{name}_seed{args.seed}_{args.expert}_{args.control_mode}_initial.pt"
    save_checkpoint(policy, stage_path, obs_dim=obs_dim, hidden_sizes=hidden_sizes)
    save_metrics(metrics_path, rows)
    print(
        f"initial_bc samples={len(actions)} loss={stats['loss']:.4f} "
        f"head_acc={stats['head_acc']:.3f} joint_acc={stats['joint_acc']:.3f}",
        flush=True,
    )

    for iteration in range(args.dagger_iters):
        new_obs, new_actions = collect_dataset(
            env_config,
            episodes=args.dagger_episodes,
            seed=args.seed + 10000 + iteration * 1000,
            expert_name=args.expert,
            control_mode=args.control_mode,
            rollin_policy=policy,
        )
        observations = np.concatenate([observations, new_obs], axis=0)
        actions = np.concatenate([actions, new_actions], axis=0)
        stats = train_bc(
            model,
            observations,
            actions,
            steps=args.dagger_train_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + iteration + 1,
        )
        rows.append({"stage": f"dagger_{iteration + 1}", "samples": len(actions), **stats})
        stage_path = model_dir / f"factorized_{name}_seed{args.seed}_{args.expert}_{args.control_mode}_dagger{iteration + 1}.pt"
        save_checkpoint(policy, stage_path, obs_dim=obs_dim, hidden_sizes=hidden_sizes)
        save_metrics(metrics_path, rows)
        print(
            f"dagger_{iteration + 1} samples={len(actions)} loss={stats['loss']:.4f} "
            f"head_acc={stats['head_acc']:.3f} joint_acc={stats['joint_acc']:.3f}",
            flush=True,
        )

    model_path = model_dir / f"factorized_{name}_seed{args.seed}_{args.expert}_{args.control_mode}.pt"
    policy.save(model_path, obs_dim=obs_dim, hidden_sizes=hidden_sizes)

    save_metrics(metrics_path, rows)
    print("Saved model:", model_path.relative_to(ROOT))
    print("Saved metrics:", metrics_path.relative_to(ROOT))


if __name__ == "__main__":
    main()

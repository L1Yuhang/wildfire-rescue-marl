from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from fire_rescue_rl.agents.astar_multi_ugv import CoverageAStarMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.envs.ppo_wrappers import make_ppo_env
from fire_rescue_rl.utils.callbacks import RewardCsvCallback
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO variants on multi-UGV rescue.")
    parser.add_argument("--config", default="configs/env_multi_hard_random_guided.yaml")
    parser.add_argument("--algo-config", default="configs/ppo_multi.yaml")
    parser.add_argument("--mode", choices=["uav", "ugv", "full"], default="uav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--bc-episodes", type=int, default=120)
    parser.add_argument("--bc-steps", type=int, default=2500)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--no-safe-projection", action="store_true")
    return parser.parse_args()


def experiment_name(env_config: dict, config_path: str) -> str:
    if env_config.get("experiment_name"):
        return str(env_config["experiment_name"])
    return Path(config_path).stem.replace("env_", "")


def extract_mode_action(env: FireRescueMultiUGVEnv, joint_action: int, mode: str):
    decoded = env.decode_action(joint_action)
    if mode == "uav":
        return int(decoded[0])
    if mode == "ugv":
        return np.asarray(decoded[1:], dtype=np.int64)
    if mode == "full":
        return np.asarray(decoded, dtype=np.int64)
    raise ValueError(mode)


def collect_expert_dataset(env_config: dict, mode: str, episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    expert = CoverageAStarMultiUGVAgent()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray | int] = []
    for episode in range(episodes):
        env = FireRescueMultiUGVEnv.from_config(env_config)
        obs, _info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            joint_action = int(expert.predict(env))
            observations.append(obs)
            actions.append(extract_mode_action(env, joint_action, mode))
            obs, _reward, terminated, truncated, _info = env.step(joint_action)
    obs_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    return obs_array, action_array


def behavior_clone_ppo(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> None:
    if steps <= 0 or len(actions) == 0:
        print("Skipped PPO behavior cloning.")
        return
    device = model.device
    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
    action_tensor = th.as_tensor(actions, dtype=th.long, device=device)
    rng = np.random.default_rng(seed)
    batch = min(batch_size, len(actions))
    losses: list[float] = []
    accs: list[float] = []
    model.policy.set_training_mode(True)
    for _ in range(steps):
        indices = rng.integers(0, len(actions), size=batch)
        distribution = model.policy.get_distribution(obs_tensor[indices])
        if action_tensor.ndim == 1:
            batch_actions = action_tensor[indices]
            loss = -distribution.log_prob(batch_actions).mean()
            pred_actions = distribution.distribution.logits.argmax(dim=1)
            acc = (pred_actions == batch_actions).float().mean()
        else:
            batch_actions = action_tensor[indices]
            loss = -distribution.log_prob(batch_actions).mean()
            pred_parts = [
                categorical.logits.argmax(dim=1)
                for categorical in distribution.distribution
            ]
            pred_actions = th.stack(pred_parts, dim=1)
            acc = (pred_actions == batch_actions).all(dim=1).float().mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
        model.policy.optimizer.step()
        losses.append(float(loss.detach().cpu()))
        accs.append(float(acc.detach().cpu()))
    print(
        f"PPO BC initialized on {len(actions)} samples; final loss={losses[-1]:.4f}, "
        f"joint/action acc={np.mean(accs[-100:]):.3f}."
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env_config = load_yaml(args.config)
    algo_config = load_yaml(args.algo_config)
    name = experiment_name(env_config, args.config)
    safe_projection = not args.no_safe_projection

    model_dir = ROOT / "outputs" / "models" / "ppo_multi"
    log_dir = ROOT / "outputs" / "logs" / "ppo_multi" / f"{name}_{args.mode}_seed{args.seed}"
    ensure_dirs(model_dir, log_dir)

    env = Monitor(
        make_ppo_env(env_config, args.mode, safe_projection=safe_projection),
        filename=str(log_dir / "monitor.csv"),
    )
    eval_env = Monitor(make_ppo_env(env_config, args.mode, safe_projection=safe_projection))
    total_timesteps = args.timesteps or int(algo_config["total_timesteps"].get(args.mode, 30000))
    model_kwargs = {key: value for key, value in algo_config.items() if key != "total_timesteps"}
    model = PPO(env=env, seed=args.seed, verbose=1, tensorboard_log=str(log_dir), **model_kwargs)

    observations, actions = collect_expert_dataset(env_config, args.mode, args.bc_episodes, args.seed)
    behavior_clone_ppo(
        model,
        observations,
        actions,
        steps=args.bc_steps,
        batch_size=args.bc_batch_size,
        seed=args.seed,
    )
    if args.bc_steps > 0:
        bc_path = model_dir / f"ppo_{name}_{args.mode}_seed{args.seed}_bc.zip"
        model.save(bc_path)
        print(f"Saved PPO BC model: {bc_path.relative_to(ROOT)}")

    callbacks = [
        RewardCsvCallback(log_dir / "step_rewards.csv"),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=8,
            deterministic=True,
            render=False,
        ),
    ]
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    final_path = model_dir / f"ppo_{name}_{args.mode}_seed{args.seed}.zip"
    model.save(final_path)
    best_path = model_dir / "best_model.zip"
    if best_path.exists():
        named_best = model_dir / f"ppo_{name}_{args.mode}_seed{args.seed}_best.zip"
        best_path.replace(named_best)
        print(f"Saved PPO best model: {named_best.relative_to(ROOT)}")
    print(f"Saved PPO final model: {final_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

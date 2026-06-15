from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.utils.callbacks import RewardCsvCallback
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on FireRescueMultiUGVEnv.")
    parser.add_argument("--config", default="configs/env_multi_easy.yaml")
    parser.add_argument("--algo-config", default="configs/dqn_multi.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--expert-episodes", type=int, default=80)
    parser.add_argument("--bc-episodes", type=int, default=120)
    parser.add_argument("--bc-steps", type=int, default=1500)
    parser.add_argument("--eval-freq", type=int, default=2500)
    parser.add_argument("--expert", choices=["astar", "coverage"], default="astar")
    return parser.parse_args()


def experiment_name(env_config: dict, config_path: str) -> str:
    if env_config.get("experiment_name"):
        return str(env_config["experiment_name"])
    return Path(config_path).stem.replace("env_", "")


def build_expert(name: str):
    if name == "coverage":
        return CoverageAStarMultiUGVAgent()
    return AStarMultiUGVAgent()


def collect_expert_dataset(env_config: dict, episodes: int, seed: int, expert_name: str) -> tuple[np.ndarray, np.ndarray]:
    expert = build_expert(expert_name)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    for episode in range(episodes):
        env = FireRescueMultiUGVEnv.from_config(env_config)
        obs, _info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(expert.predict(env))
            observations.append(obs)
            actions.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def behavior_clone(model: DQN, observations: np.ndarray, actions: np.ndarray, steps: int, seed: int) -> None:
    if steps <= 0 or len(actions) == 0:
        print("Skipped behavior cloning.")
        return
    device = model.device
    obs_tensor = th.as_tensor(observations, device=device)
    action_tensor = th.as_tensor(actions, device=device)
    batch_size = min(256, len(actions))
    rng = np.random.default_rng(seed)
    model.policy.set_training_mode(True)
    losses: list[float] = []
    for _ in range(steps):
        indices = rng.integers(0, len(actions), size=batch_size)
        q_values = model.q_net(obs_tensor[indices])
        loss = F.cross_entropy(q_values, action_tensor[indices])
        model.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=10.0)
        model.policy.optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    print(f"Behavior cloned on {len(actions)} expert samples; final loss={losses[-1]:.4f}.")


def warm_start(model: DQN, env_config: dict, episodes: int, seed: int, expert_name: str) -> None:
    expert = build_expert(expert_name)
    added = 0
    for episode in range(episodes):
        env = FireRescueMultiUGVEnv.from_config(env_config)
        obs, _info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(expert.predict(env))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            model.replay_buffer.add(
                np.asarray([obs], dtype=np.float32),
                np.asarray([next_obs], dtype=np.float32),
                np.asarray([action]),
                np.asarray([reward], dtype=np.float32),
                np.asarray([done], dtype=bool),
                [info],
            )
            obs = next_obs
            added += 1
    print(f"Warm-started replay buffer with {added} A* transitions.")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env_config = load_yaml(args.config)
    algo_config = load_yaml(args.algo_config)
    name = experiment_name(env_config, args.config)
    expert_suffix = "" if args.expert == "astar" else f"_{args.expert}"
    model_dir = ROOT / "outputs" / "models" / "dqn_multi"
    log_dir = ROOT / "outputs" / "logs" / "dqn_multi" / f"{name}_seed{args.seed}{expert_suffix}"
    ensure_dirs(model_dir, log_dir)

    env = Monitor(FireRescueMultiUGVEnv.from_config(env_config), filename=str(log_dir / "monitor.csv"))
    eval_env = Monitor(FireRescueMultiUGVEnv.from_config(env_config))
    total_timesteps = args.timesteps or int(algo_config["total_timesteps"].get(name, 60000))
    model_kwargs = {key: value for key, value in algo_config.items() if key != "total_timesteps"}
    model = DQN(env=env, seed=args.seed, verbose=1, tensorboard_log=str(log_dir), **model_kwargs)

    observations, actions = collect_expert_dataset(env_config, args.bc_episodes, args.seed, args.expert)
    behavior_clone(model, observations, actions, args.bc_steps, args.seed)
    if args.bc_steps > 0:
        bc_path = model_dir / f"dqn_{name}_seed{args.seed}{expert_suffix}_bc.zip"
        model.save(bc_path)
        print(f"Saved behavior-cloned model: {bc_path.relative_to(ROOT)}")
    if args.expert_episodes > 0:
        warm_start(model, env_config, args.expert_episodes, args.seed, args.expert)

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
    final_path = model_dir / f"dqn_{name}_seed{args.seed}{expert_suffix}.zip"
    model.save(final_path)
    best_path = model_dir / "best_model.zip"
    if best_path.exists():
        named_best = model_dir / f"dqn_{name}_seed{args.seed}{expert_suffix}_best.zip"
        best_path.replace(named_best)
        print(f"Saved best model: {named_best.relative_to(ROOT)}")
    print(f"Saved final model: {final_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

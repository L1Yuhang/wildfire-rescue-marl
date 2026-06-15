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

from fire_rescue_rl.agents.astar_team import AStarTeamAgent
from fire_rescue_rl.envs.fire_rescue_env import FireRescueMAEnv
from fire_rescue_rl.utils.callbacks import RewardCsvCallback
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_easy.yaml")
    parser.add_argument("--algo-config", default="configs/dqn.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--expert-episodes", type=int, default=30)
    parser.add_argument("--bc-steps", type=int, default=0)
    parser.add_argument("--bc-episodes", type=int, default=120)
    return parser.parse_args()


def warm_start(model: DQN, env_config: dict, episodes: int, seed: int) -> None:
    expert = AStarTeamAgent()
    added = 0
    for episode in range(episodes):
        env = FireRescueMAEnv.from_config(env_config)
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = expert.predict(env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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


def behavior_clone(model: DQN, env_config: dict, episodes: int, steps: int, seed: int) -> None:
    expert = AStarTeamAgent()
    observations = []
    actions = []
    for episode in range(episodes):
        env = FireRescueMAEnv.from_config(env_config)
        obs, _info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = int(expert.predict(env))
            observations.append(obs)
            actions.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
    if not observations:
        print("Skipped behavior cloning: no expert samples collected.")
        return

    device = model.device
    obs_tensor = th.as_tensor(np.asarray(observations, dtype=np.float32), device=device)
    action_tensor = th.as_tensor(np.asarray(actions, dtype=np.int64), device=device)
    dataset_size = len(actions)
    batch_size = min(256, dataset_size)
    model.policy.set_training_mode(True)
    rng = np.random.default_rng(seed)
    losses = []
    for _step in range(steps):
        indices = rng.integers(0, dataset_size, size=batch_size)
        q_values = model.q_net(obs_tensor[indices])
        loss = F.cross_entropy(q_values, action_tensor[indices])
        model.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=10.0)
        model.policy.optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    print(
        f"Behavior-cloned DQN policy on {dataset_size} A* samples "
        f"for {steps} gradient steps; final loss={losses[-1]:.4f}."
    )


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env_config = load_yaml(args.config)
    algo_config = load_yaml(args.algo_config)
    difficulty = env_config.get("difficulty", "easy")
    model_dir = ROOT / "outputs" / "models" / "dqn"
    log_dir = ROOT / "outputs" / "logs" / "dqn" / f"{difficulty}_seed{args.seed}"
    ensure_dirs(model_dir, log_dir)

    env = Monitor(FireRescueMAEnv.from_config(env_config), filename=str(log_dir / "monitor.csv"))
    eval_env = Monitor(FireRescueMAEnv.from_config(env_config))
    total_timesteps = args.timesteps or int(algo_config["total_timesteps"].get(difficulty, 70000))
    model_kwargs = {k: v for k, v in algo_config.items() if k != "total_timesteps"}
    model = DQN(env=env, seed=args.seed, verbose=1, tensorboard_log=str(log_dir), **model_kwargs)
    if args.bc_steps > 0:
        behavior_clone(model, env_config, args.bc_episodes, args.bc_steps, args.seed)
    if args.expert_episodes > 0:
        warm_start(model, env_config, args.expert_episodes, args.seed)
    callbacks = [
        RewardCsvCallback(log_dir / "step_rewards.csv"),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir / "eval"),
            eval_freq=2500,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
    ]
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    final_path = model_dir / f"dqn_{difficulty}_seed{args.seed}.zip"
    model.save(final_path)
    best_path = model_dir / "best_model.zip"
    if best_path.exists():
        named_best = model_dir / f"dqn_{difficulty}_seed{args.seed}_best.zip"
        best_path.replace(named_best)
        print(f"Saved best model: {named_best}")
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()

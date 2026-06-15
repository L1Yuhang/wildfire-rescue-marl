from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from fire_rescue_rl.agents.astar_multi_ugv import CoverageAStarMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import FireRescueMultiUGVEnv
from fire_rescue_rl.envs.ppo_wrappers import make_ppo_env
from fire_rescue_rl.utils.callbacks import RewardCsvCallback
from fire_rescue_rl.utils.config import ensure_dirs, load_yaml
from fire_rescue_rl.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO variants on multi-UGV rescue.")
    parser.add_argument("--config", default="configs/env_multi_hard_random_guided.yaml")
    parser.add_argument("--algo-config", default="configs/ppo_multi.yaml")
    parser.add_argument("--mode", choices=["uav", "ugv", "full"], default="uav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--bc-episodes", type=int, default=180)
    parser.add_argument("--bc-steps", type=int, default=2500)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--no-masked-bc", action="store_true")
    parser.add_argument("--init-model-path", default="")
    parser.add_argument("--output-suffix", default="")
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


def collect_expert_dataset(
    env_config: dict,
    mode: str,
    episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert = CoverageAStarMultiUGVAgent()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray | int] = []
    action_masks: list[np.ndarray] = []
    for episode in range(episodes):
        env = make_ppo_env(env_config, mode, safe_projection=True)
        obs, _info = env.reset(seed=seed + episode)
        base_env: FireRescueMultiUGVEnv = env.unwrapped
        terminated = False
        truncated = False
        while not (terminated or truncated):
            joint_action = int(expert.predict(base_env))
            observations.append(obs)
            actions.append(extract_mode_action(base_env, joint_action, mode))
            action_masks.append(np.asarray(get_action_masks(env), dtype=bool))
            obs, _reward, terminated, truncated, _info = env.step(actions[-1])
        env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(action_masks, dtype=bool),
    )


def behavior_clone(
    model: MaskablePPO,
    observations: np.ndarray,
    actions: np.ndarray,
    action_masks: np.ndarray | None,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> None:
    if steps <= 0 or len(actions) == 0:
        print("Skipped MaskablePPO behavior cloning.")
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
        batch_masks = action_masks[indices] if action_masks is not None else None
        distribution = model.policy.get_distribution(obs_tensor[indices], action_masks=batch_masks)
        batch_actions = action_tensor[indices]
        loss = -distribution.log_prob(batch_actions).mean()
        if batch_actions.ndim == 1:
            pred_actions = distribution.distribution.logits.argmax(dim=1)
            acc = (pred_actions == batch_actions).float().mean()
        else:
            categoricals = getattr(distribution, "distribution", None)
            if categoricals is None:
                categoricals = distribution.distributions
            pred_parts = [categorical.logits.argmax(dim=1) for categorical in categoricals]
            pred_actions = th.stack(pred_parts, dim=1)
            acc = (pred_actions == batch_actions).all(dim=1).float().mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
        model.policy.optimizer.step()
        losses.append(float(loss.detach().cpu()))
        accs.append(float(acc.detach().cpu()))
    print(
        f"MaskablePPO BC initialized on {len(actions)} samples; final loss={losses[-1]:.4f}, "
        f"joint/action acc={np.mean(accs[-100:]):.3f}."
    )


def apply_loaded_model_hparams(model: MaskablePPO, model_kwargs: dict) -> None:
    """Apply selected PPO hyperparameters after loading a warm-start policy."""
    if "learning_rate" in model_kwargs:
        learning_rate = float(model_kwargs["learning_rate"])
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = learning_rate
    if "clip_range" in model_kwargs:
        model.clip_range = get_schedule_fn(float(model_kwargs["clip_range"]))
    for name in ["n_epochs", "batch_size", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm"]:
        if name in model_kwargs:
            setattr(model, name, model_kwargs[name])


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    env_config = load_yaml(args.config)
    algo_config = load_yaml(args.algo_config)
    name = experiment_name(env_config, args.config)
    suffix = args.output_suffix.strip()
    file_suffix = f"_{suffix}" if suffix and not suffix.startswith("_") else suffix

    model_dir = ROOT / "outputs" / "models" / "maskable_ppo_multi"
    log_dir = ROOT / "outputs" / "logs" / "maskable_ppo_multi" / f"{name}_{args.mode}_seed{args.seed}{file_suffix}"
    ensure_dirs(model_dir, log_dir)

    env = Monitor(make_ppo_env(env_config, args.mode, safe_projection=True), filename=str(log_dir / "monitor.csv"))
    eval_env = Monitor(make_ppo_env(env_config, args.mode, safe_projection=True))
    total_timesteps = (
        int(args.timesteps)
        if args.timesteps is not None
        else int(algo_config["total_timesteps"].get(args.mode, 30000))
    )
    model_kwargs = {key: value for key, value in algo_config.items() if key != "total_timesteps"}
    if args.init_model_path:
        model = MaskablePPO.load(args.init_model_path, env=env)
        apply_loaded_model_hparams(model, model_kwargs)
        model.verbose = 1
        model.tensorboard_log = str(log_dir)
        print(f"Loaded initial MaskablePPO model: {Path(args.init_model_path)}")
        print(
            "Applied loaded-model PPO hparams:",
            {
                "learning_rate": model.learning_rate,
                "clip_range": model_kwargs.get("clip_range"),
                "n_epochs": model.n_epochs,
                "ent_coef": model.ent_coef,
            },
        )
    else:
        model = MaskablePPO(env=env, seed=args.seed, verbose=1, tensorboard_log=str(log_dir), **model_kwargs)

    if args.bc_steps > 0 and args.bc_episodes > 0:
        observations, actions, action_masks = collect_expert_dataset(
            env_config,
            args.mode,
            args.bc_episodes,
            args.seed,
        )
        behavior_clone(
            model,
            observations,
            actions,
            None if args.no_masked_bc else action_masks,
            steps=args.bc_steps,
            batch_size=args.bc_batch_size,
            seed=args.seed,
        )
        bc_path = model_dir / f"maskppo_{name}_{args.mode}_seed{args.seed}{file_suffix}_bc.zip"
        model.save(bc_path)
        print(f"Saved MaskablePPO BC model: {bc_path.relative_to(ROOT)}")
    else:
        print("Skipped MaskablePPO behavior cloning.")

    if total_timesteps > 0:
        callbacks = [
            RewardCsvCallback(log_dir / "step_rewards.csv"),
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(model_dir),
                log_path=str(log_dir / "eval"),
                eval_freq=args.eval_freq,
                n_eval_episodes=8,
                deterministic=True,
                render=False,
            ),
        ]
        model.learn(total_timesteps=total_timesteps, callback=callbacks, use_masking=True)
    else:
        print("Skipped MaskablePPO rollout fine-tuning because --timesteps <= 0.")
    final_path = model_dir / f"maskppo_{name}_{args.mode}_seed{args.seed}{file_suffix}.zip"
    model.save(final_path)
    best_path = model_dir / "best_model.zip"
    if best_path.exists():
        named_best = model_dir / f"maskppo_{name}_{args.mode}_seed{args.seed}{file_suffix}_best.zip"
        best_path.replace(named_best)
        print(f"Saved MaskablePPO best model: {named_best.relative_to(ROOT)}")
    print(f"Saved MaskablePPO final model: {final_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

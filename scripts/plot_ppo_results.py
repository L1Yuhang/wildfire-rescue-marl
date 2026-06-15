from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fire_rescue_rl.utils.config import ensure_dirs


SUMMARY_FILES = [
    "outputs/eval/metrics_csv/ppo_summary_multi_hard_random_guided_uav_seed1000_ppo_uav_bc_holdout50.csv",
    "outputs/eval/metrics_csv/ppo_summary_multi_hard_random_guided_uav_seed1000_ppo_uav_best_holdout50.csv",
    "outputs/eval/metrics_csv/ppo_summary_multi_hard_random_guided_ugv_seed1000_ppo_ugv_bc_holdout50.csv",
    "outputs/eval/metrics_csv/ppo_summary_multi_hard_random_guided_ugv_seed1000_ppo_ugv_best_holdout50.csv",
    "outputs/eval/metrics_csv/ppo_summary_multi_hard_random_guided_full_seed1000_ppo_full_bc_holdout50.csv",
    "outputs/eval/metrics_csv/ppo_summary_multi_hard_random_guided_full_seed1000_ppo_full_best_holdout50.csv",
    "outputs/eval/metrics_csv/multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
    "outputs/eval/metrics_csv/multi_summary_multi_hard_random_guided_seed1000_dedicated_dqn_ugv_astar_uav_holdout50.csv",
]

SELECT_ROWS = {
    ("ppo_uav", "PPO-UAV BC"): "ppo_summary_multi_hard_random_guided_uav_seed1000_ppo_uav_bc_holdout50.csv",
    ("ppo_uav", "PPO-UAV Best"): "ppo_summary_multi_hard_random_guided_uav_seed1000_ppo_uav_best_holdout50.csv",
    ("ppo_ugv", "PPO-UGV BC"): "ppo_summary_multi_hard_random_guided_ugv_seed1000_ppo_ugv_bc_holdout50.csv",
    ("ppo_ugv", "PPO-UGV Best"): "ppo_summary_multi_hard_random_guided_ugv_seed1000_ppo_ugv_best_holdout50.csv",
    ("ppo_full", "Full PPO BC"): "ppo_summary_multi_hard_random_guided_full_seed1000_ppo_full_bc_holdout50.csv",
    ("ppo_full", "Full PPO Best"): "ppo_summary_multi_hard_random_guided_full_seed1000_ppo_full_best_holdout50.csv",
    ("learned_uav_astar_ugv", "Learned UAV + A* UGV"): "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
    ("dedicated_dqn_ugv_astar_uav", "DQN UGV + A* UAV"): "multi_summary_multi_hard_random_guided_seed1000_dedicated_dqn_ugv_astar_uav_holdout50.csv",
    ("coverage_astar", "Coverage A*"): "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
    ("astar", "Oracle A*"): "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
    ("greedy", "Greedy"): "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
}

POLICY_ORDER = [
    "Greedy",
    "PPO-UAV BC",
    "PPO-UAV Best",
    "PPO-UGV BC",
    "PPO-UGV Best",
    "Full PPO BC",
    "Full PPO Best",
    "DQN UGV + A* UAV",
    "Learned UAV + A* UGV",
    "Coverage A*",
    "Oracle A*",
]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_selected() -> pd.DataFrame:
    rows = []
    metrics_dir = ROOT / "outputs/eval/metrics_csv"
    for (agent, label), filename in SELECT_ROWS.items():
        path = metrics_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        match = df[df["agent"] == agent]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["policy"] = label
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No PPO summary CSV files found.")
    result = pd.DataFrame(rows)
    result["_rank"] = result["policy"].map({name: idx for idx, name in enumerate(POLICY_ORDER)})
    return result.sort_values("_rank")


def save_bar(df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(12.0, 4.8), dpi=160)
    palette = [
        "#ef8a17",
        "#79addc",
        "#2f78bd",
        "#d8a7ca",
        "#b565a7",
        "#d9534f",
        "#8b1e3f",
        "#9467bd",
        "#188977",
        "#4c78a8",
        "#1f4e79",
    ]
    sns.barplot(data=df, x="policy", y=metric, hue="policy", palette=palette[: len(df)], legend=False)
    plt.title(title, fontsize=12, weight="bold")
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_training_curves(out_dir: Path) -> None:
    candidates = [
        ("PPO-UAV", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_uav_seed0/step_rewards.csv"),
        ("PPO-UGV", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_ugv_seed0/step_rewards.csv"),
        ("Full PPO", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_full_seed0/step_rewards.csv"),
    ]
    frames = []
    for label, path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "reward" not in df.columns:
            continue
        df["policy"] = label
        df["rolling_reward"] = df["reward"].rolling(100, min_periods=1).mean()
        frames.append(df)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    plt.figure(figsize=(9.0, 4.4), dpi=160)
    x_col = "step" if "step" in data.columns else "timesteps"
    sns.lineplot(data=data, x=x_col, y="rolling_reward", hue="policy", linewidth=1.5)
    plt.title("PPO Training Reward (Rolling Mean)", fontsize=12, weight="bold")
    plt.xlabel("Environment Step")
    plt.ylabel("Rolling Reward")
    plt.tight_layout()
    plt.savefig(out_dir / "ppo_training_reward_curves.png")
    plt.close()


def main() -> None:
    out_dir = ROOT / "outputs/figures/multi_ugv/ppo"
    ensure_dirs(out_dir)
    set_style()
    df = load_selected()
    out_csv = ROOT / "outputs/eval/metrics_csv/ppo_generalization_summary.csv"
    df.to_csv(out_csv, index=False)
    save_bar(df, "success", "Success Rate", "PPO Holdout Success vs Hybrid Baselines", out_dir / "ppo_success_rate.png")
    save_bar(df, "delivered_count", "Delivered Survivors", "PPO Holdout Delivery", out_dir / "ppo_delivered_count.png")
    save_bar(df, "discovered_count", "Discovered Survivors", "PPO Holdout Discovery", out_dir / "ppo_discovered_count.png")
    save_bar(df, "risk_exposure", "Risk Exposure", "PPO Holdout Risk", out_dir / "ppo_risk_exposure.png")
    save_training_curves(out_dir)
    print("Saved PPO figures to:", out_dir.relative_to(ROOT))
    print(df[["policy", "success", "reward", "steps", "delivered_count", "discovered_count", "risk_exposure"]].to_string(index=False))


if __name__ == "__main__":
    main()

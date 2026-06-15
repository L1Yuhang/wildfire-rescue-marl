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
    "outputs/eval/metrics_csv/multi_summary_multi_hard_random_explore_seed1000_coverage_bc_holdout50.csv",
    "outputs/eval/metrics_csv/multi_summary_multi_hard_random_explore_seed1000_factorized_dagger_masked_holdout50.csv",
    "outputs/eval/metrics_csv/multi_summary_multi_hard_random_guided_seed1000_factorized_guided_dagger_masked_holdout50.csv",
    "outputs/eval/metrics_csv/multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
    "outputs/eval/metrics_csv/multi_summary_multi_hard_random_guided_seed1000_dedicated_dqn_ugv_astar_uav_holdout50.csv",
]

POLICY_ORDER = [
    "random",
    "greedy",
    "dqn_coverage_bc",
    "factorized_dqn_dagger_masked",
    "factorized_guided_dagger_masked",
    "dedicated_dqn_ugv_astar_uav",
    "learned_uav_astar_ugv",
    "coverage_astar",
    "astar",
]

LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "dqn_coverage_bc": "Joint DQN-BC",
    "factorized_dqn_dagger_masked": "Factorized",
    "factorized_guided_dagger_masked": "Guided Factorized",
    "dedicated_dqn_ugv_astar_uav": "DQN UGV + A* UAV",
    "learned_uav_astar_ugv": "Learned UAV + A* UGV",
    "coverage_astar": "Coverage A*",
    "astar": "Oracle A*",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_summary() -> pd.DataFrame:
    frames = []
    for rel_path in SUMMARY_FILES:
        path = ROOT / rel_path
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No generalization summary CSV files found.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["agent"].isin(POLICY_ORDER)].copy()
    df["_rank"] = df["agent"].map({agent: idx for idx, agent in enumerate(POLICY_ORDER)})
    df = df.sort_values("_rank").drop_duplicates("agent", keep="last")
    df["policy"] = df["agent"].map(LABELS)
    return df.sort_values("_rank")


def save_bar(df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(10.8, 4.8), dpi=160)
    palette = [
        "#9ca3af",
        "#ef8a17",
        "#d9534f",
        "#c44e52",
        "#7b61b8",
        "#9467bd",
        "#188977",
        "#2f78bd",
        "#1f4e79",
    ]
    sns.barplot(data=df, x="policy", y=metric, hue="policy", palette=palette[: len(df)], legend=False)
    plt.title(title, fontsize=12, weight="bold")
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.xticks(rotation=22, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_training_plot(path: Path) -> None:
    metrics_path = ROOT / "outputs/logs/factorized_multi/multi_hard_random_guided_seed0_coverage/training_metrics.csv"
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    x = range(len(metrics))
    ax1.plot(x, metrics["loss"], marker="o", color="#c44e52", label="Loss")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_xticks(list(x), metrics["stage"])
    ax2 = ax1.twinx()
    ax2.plot(x, metrics["head_acc"], marker="s", color="#2f78bd", label="Head Accuracy")
    ax2.plot(x, metrics["joint_acc"], marker="^", color="#188977", label="Joint Accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right", frameon=True)
    ax1.set_title("Guided Factorized DQN-DAgger Training", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs/figures/multi_ugv/generalization"
    ensure_dirs(out_dir)
    set_style()
    df = load_summary()
    df.to_csv(ROOT / "outputs/eval/metrics_csv/multi_generalization_diagnosis_summary.csv", index=False)
    save_bar(df, "success", "Success Rate", "Random-Map Holdout Success", out_dir / "generalization_success_rate.png")
    save_bar(df, "delivered_count", "Delivered Survivors", "Random-Map Holdout Delivery", out_dir / "generalization_delivered_count.png")
    save_bar(df, "risk_exposure", "Risk Exposure", "Random-Map Holdout Risk", out_dir / "generalization_risk_exposure.png")
    save_bar(df, "invalid_ugv", "Invalid UGV Actions", "Ground-Robot Invalid Actions", out_dir / "generalization_invalid_ugv.png")
    save_training_plot(out_dir / "guided_factorized_training_metrics.png")
    print("Saved generalization figures to:", out_dir.relative_to(ROOT))
    print(df[["agent", "success", "reward", "steps", "delivered_count", "discovered_count", "risk_exposure"]].to_string(index=False))


if __name__ == "__main__":
    main()

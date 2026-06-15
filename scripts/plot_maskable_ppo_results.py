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


METRICS_DIR = ROOT / "outputs" / "eval" / "metrics_csv"
OUT_DIR = ROOT / "outputs" / "figures" / "multi_ugv" / "maskable_ppo"

ROW_SPECS = [
    (
        "Greedy",
        "Rule baseline",
        "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
        "greedy",
    ),
    (
        "PPO-UAV BC",
        "PPO",
        "ppo_summary_multi_hard_random_guided_uav_seed1000_ppo_uav_bc_holdout50.csv",
        "ppo_uav",
    ),
    (
        "PPO-UAV Best",
        "PPO",
        "ppo_summary_multi_hard_random_guided_uav_seed1000_ppo_uav_best_holdout50.csv",
        "ppo_uav",
    ),
    (
        "MaskablePPO-UAV BC",
        "Masked PPO",
        "maskppo_summary_multi_hard_random_guided_uav_seed1000_maskppo_uav_bc_holdout50.csv",
        "maskppo_uav",
    ),
    (
        "Masked BC seed3",
        "Masked BC",
        "maskppo_summary_multi_hard_random_guided_uav_seed1000_maskppo_uav_maskedbc_seed3_holdout50.csv",
        "maskppo_uav",
    ),
    (
        "Masked BC seed4",
        "Masked BC",
        "maskppo_summary_multi_hard_random_guided_uav_seed1000_maskppo_uav_maskedbc_seed4_holdout50.csv",
        "maskppo_uav",
    ),
    (
        "Masked BC + conservative PPO",
        "Final PPO",
        "maskppo_summary_multi_hard_random_guided_uav_seed1000_maskppo_uav_seed4_trueft_final_holdout50.csv",
        "maskppo_uav",
    ),
    (
        "PPO-UGV Best",
        "PPO",
        "ppo_summary_multi_hard_random_guided_ugv_seed1000_ppo_ugv_best_holdout50.csv",
        "ppo_ugv",
    ),
    (
        "MaskablePPO-UGV Best",
        "Masked PPO",
        "maskppo_summary_multi_hard_random_guided_ugv_seed1000_maskppo_ugv_best_holdout50.csv",
        "maskppo_ugv",
    ),
    (
        "Full MaskablePPO",
        "Masked PPO",
        "maskppo_summary_multi_hard_random_guided_full_seed1000_maskppo_full_best_holdout50.csv",
        "maskppo_full",
    ),
    (
        "DQN-UAV + A* UGV",
        "Hybrid DQN",
        "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
        "learned_uav_astar_ugv",
    ),
    (
        "Coverage A*",
        "Planning upper bound",
        "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
        "coverage_astar",
    ),
    (
        "Oracle A*",
        "Planning upper bound",
        "multi_summary_multi_hard_random_guided_seed1000_learned_uav_astar_ugv_holdout50.csv",
        "astar",
    ),
]

PALETTE = {
    "Rule baseline": "#8c8c8c",
    "PPO": "#d95f5f",
    "Masked PPO": "#5b8fd6",
    "Masked BC": "#36a3a1",
    "Final PPO": "#157f5b",
    "Hybrid DQN": "#8a66cc",
    "Planning upper bound": "#26384f",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#f7f8fa"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["axes.edgecolor"] = "#d1d5db"
    plt.rcParams["grid.color"] = "#e5e7eb"


def read_row(filename: str, agent: str) -> dict | None:
    path = METRICS_DIR / filename
    if not path.exists():
        print(f"Missing summary CSV: {path.relative_to(ROOT)}")
        return None
    df = pd.read_csv(path)
    match = df[df["agent"] == agent]
    if match.empty:
        print(f"Missing agent '{agent}' in {path.relative_to(ROOT)}")
        return None
    return match.iloc[0].to_dict()


def load_summary() -> pd.DataFrame:
    rows = []
    for rank, (label, family, filename, agent) in enumerate(ROW_SPECS):
        row = read_row(filename, agent)
        if row is None:
            continue
        row["policy"] = label
        row["family"] = family
        row["_rank"] = rank
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No MaskablePPO summary rows were found.")
    return pd.DataFrame(rows).sort_values("_rank").reset_index(drop=True)


def annotate_bars(ax, metric: str, fmt: str) -> None:
    for patch in ax.patches:
        value = patch.get_width()
        if pd.isna(value):
            continue
        x = value + max(0.01, abs(value) * 0.015)
        y = patch.get_y() + patch.get_height() / 2
        ax.text(x, y, fmt.format(value), va="center", ha="left", fontsize=9, color="#111827")


def save_horizontal_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    filename: str,
    *,
    fmt: str = "{:.2f}",
    xlim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 6.8), dpi=180)
    colors = [PALETTE[group] for group in df["family"]]
    sns.barplot(data=df, y="policy", x=metric, hue="policy", palette=colors, dodge=False, legend=False, ax=ax)
    ax.set_title(title, fontsize=15, weight="bold", color="#111827", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color="#374151")
    ax.set_ylabel("")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    annotate_bars(ax, metric, fmt)
    sns.despine(ax=ax, left=True, bottom=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)


def save_progression(df: pd.DataFrame) -> None:
    labels = [
        "PPO-UAV BC",
        "MaskablePPO-UAV BC",
        "Masked BC seed3",
        "Masked BC seed4",
        "Masked BC + conservative PPO",
    ]
    plot_df = df[df["policy"].isin(labels)].copy()
    plot_df["policy"] = pd.Categorical(plot_df["policy"], categories=labels, ordered=True)
    plot_df = plot_df.sort_values("policy")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    for ax, metric, ylabel, ylim in [
        (axes[0], "success", "Success rate", (0.0, 1.05)),
        (axes[1], "delivered_count", "Delivered survivors", (0.0, 3.15)),
    ]:
        sns.lineplot(
            data=plot_df,
            x="policy",
            y=metric,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color="#157f5b",
            ax=ax,
        )
        ax.set_title(ylabel, fontsize=12, weight="bold", color="#111827")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelrotation=28, labelsize=8)
        for x, value in enumerate(plot_df[metric].tolist()):
            ax.text(x, value + (0.03 if metric == "success" else 0.05), f"{value:.2f}", ha="center", fontsize=9)
    fig.suptitle("MaskablePPO-UAV Optimization Path", fontsize=15, weight="bold", color="#111827")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "maskppo_uav_optimization_path.png")
    plt.close(fig)


def save_training_curve() -> None:
    candidates = [
        ("MaskablePPO-UAV seed0", ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_uav_seed0/step_rewards.csv"),
        (
            "BC + conservative PPO",
            ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_uav_seed4_true_conservative_ft/step_rewards.csv",
        ),
        (
            "BC + default PPO",
            ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_uav_seed4_conservative_ft/step_rewards.csv",
        ),
    ]
    frames = []
    for label, path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "reward" not in frame.columns:
            continue
        frame["policy"] = label
        frame["rolling_reward"] = frame["reward"].rolling(300, min_periods=20).mean()
        frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    sns.lineplot(data=data, x="timesteps", y="rolling_reward", hue="policy", linewidth=1.8, ax=ax)
    ax.set_title("MaskablePPO Rollout Fine-Tuning Reward", fontsize=14, weight="bold", color="#111827")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rolling step reward")
    ax.legend(title="", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "maskppo_training_reward_curves.png")
    plt.close(fig)


def save_final_radar(df: pd.DataFrame) -> None:
    labels = ["Greedy", "PPO-UAV BC", "DQN-UAV + A* UGV", "Masked BC + conservative PPO", "Coverage A*"]
    plot_df = df[df["policy"].isin(labels)].copy()
    metrics = [
        ("success", "Success"),
        ("delivered_count", "Delivered"),
        ("discovered_count", "Discovered"),
        ("reward", "Reward"),
        ("risk_score", "Low risk"),
    ]
    plot_df["risk_score"] = 1.0 - (plot_df["risk_exposure"] / max(1.0, plot_df["risk_exposure"].max()))
    plot_df["reward"] = (plot_df["reward"] - plot_df["reward"].min()) / max(1e-9, plot_df["reward"].max() - plot_df["reward"].min())
    plot_df["delivered_count"] = plot_df["delivered_count"] / 3.0
    plot_df["discovered_count"] = plot_df["discovered_count"] / 3.0

    angles = [idx / float(len(metrics)) * 2.0 * 3.1415926 for idx in range(len(metrics))]
    angles += angles[:1]
    fig = plt.figure(figsize=(7.2, 6.2), dpi=180)
    ax = plt.subplot(111, polar=True)
    for _, row in plot_df.iterrows():
        values = [float(row[key]) for key, _label in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=row["policy"], color=PALETTE[row["family"]])
        ax.fill(angles, values, alpha=0.08, color=PALETTE[row["family"]])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _key, label in metrics], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticklabels([])
    ax.grid(color="#d1d5db")
    ax.set_title("Final Policy Profile", fontsize=14, weight="bold", color="#111827", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "maskppo_final_policy_radar.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs(OUT_DIR, METRICS_DIR)
    set_style()
    df = load_summary()
    out_csv = METRICS_DIR / "maskable_ppo_generalization_summary.csv"
    df.to_csv(out_csv, index=False)

    save_progression(df)
    save_horizontal_bar(
        df,
        "success",
        "Hard Random Holdout Success Rate",
        "Success rate",
        "maskppo_success_rate.png",
        xlim=(0.0, 1.12),
    )
    save_horizontal_bar(
        df,
        "delivered_count",
        "Average Delivered Survivors",
        "Delivered survivors / episode",
        "maskppo_delivered_count.png",
        xlim=(0.0, 3.35),
    )
    save_horizontal_bar(
        df,
        "risk_exposure",
        "Risk Exposure",
        "Accumulated smoke / near-fire exposure",
        "maskppo_risk_exposure.png",
        fmt="{:.1f}",
    )
    save_horizontal_bar(
        df,
        "steps",
        "Average Episode Steps",
        "Steps",
        "maskppo_average_steps.png",
        fmt="{:.0f}",
    )
    save_training_curve()
    save_final_radar(df)

    cols = ["policy", "family", "success", "reward", "steps", "delivered_count", "discovered_count", "risk_exposure"]
    print("Saved MaskablePPO figures to:", OUT_DIR.relative_to(ROOT))
    print("Saved summary CSV:", out_csv.relative_to(ROOT))
    print(df[cols].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

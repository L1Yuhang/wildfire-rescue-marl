from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fire_rescue_rl.utils.config import ensure_dirs


OUT_DIR = ROOT / "outputs" / "figures" / "report_training_curves"
INDIVIDUAL_DIR = OUT_DIR / "individual"
DATA_DIR = ROOT / "outputs" / "eval" / "metrics_csv"


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    group: str
    log_dir: Path
    note: str = ""


CURVES = [
    CurveSpec("dqn_easy_seed0", "DQN Easy seed0", "Single-UGV DQN", ROOT / "outputs/logs/dqn/easy_seed0"),
    CurveSpec("dqn_easy_seed1", "DQN Easy seed1", "Single-UGV DQN", ROOT / "outputs/logs/dqn/easy_seed1"),
    CurveSpec("dqn_easy_seed2", "DQN Easy seed2", "Single-UGV DQN", ROOT / "outputs/logs/dqn/easy_seed2"),
    CurveSpec("dqn_medium_seed0", "DQN Medium seed0", "Single-UGV DQN", ROOT / "outputs/logs/dqn/medium_seed0"),
    CurveSpec("dqn_hard_seed0", "DQN Hard seed0", "Single-UGV DQN", ROOT / "outputs/logs/dqn/hard_seed0"),
    CurveSpec("dqn_hard_seed1", "DQN Hard seed1", "Single-UGV DQN", ROOT / "outputs/logs/dqn/hard_seed1"),
    CurveSpec("multi_dqn_easy", "Multi-UGV DQN Easy", "Multi-UGV DQN", ROOT / "outputs/logs/dqn_multi/multi_easy_seed0"),
    CurveSpec("multi_dqn_medium", "Multi-UGV DQN Medium", "Multi-UGV DQN", ROOT / "outputs/logs/dqn_multi/multi_medium_seed0"),
    CurveSpec("multi_dqn_hard", "Multi-UGV DQN Hard", "Multi-UGV DQN", ROOT / "outputs/logs/dqn_multi/multi_hard_seed0"),
    CurveSpec("multi_dqn_random", "Multi-UGV DQN Random Hard", "Multi-UGV DQN", ROOT / "outputs/logs/dqn_multi/multi_hard_random_seed0"),
    CurveSpec(
        "multi_dqn_coverage",
        "Multi-UGV DQN Coverage Hard",
        "Multi-UGV DQN",
        ROOT / "outputs/logs/dqn_multi/multi_hard_random_explore_seed0_coverage",
    ),
    CurveSpec("dqn_er_easy", "Expert-Regularized DQN Easy", "Expert-Regularized DQN", ROOT / "outputs/logs/dqn_multi_er/multi_easy_seed0"),
    CurveSpec("dqn_er_medium", "Expert-Regularized DQN Medium", "Expert-Regularized DQN", ROOT / "outputs/logs/dqn_multi_er/multi_medium_seed0"),
    CurveSpec("dqn_er_hard", "Expert-Regularized DQN Hard", "Expert-Regularized DQN", ROOT / "outputs/logs/dqn_multi_er/multi_hard_seed0"),
    CurveSpec("ppo_uav_seed0", "PPO-UAV seed0", "PPO", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_uav_seed0"),
    CurveSpec("ppo_uav_seed1", "PPO-UAV seed1", "PPO", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_uav_seed1"),
    CurveSpec("ppo_uav_seed2", "PPO-UAV seed2", "PPO", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_uav_seed2"),
    CurveSpec("ppo_ugv_seed0", "PPO-UGV seed0", "PPO", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_ugv_seed0"),
    CurveSpec("ppo_full_seed0", "Full PPO seed0", "PPO", ROOT / "outputs/logs/ppo_multi/multi_hard_random_guided_full_seed0"),
    CurveSpec("maskppo_uav_seed0", "MaskablePPO-UAV seed0", "MaskablePPO", ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_uav_seed0"),
    CurveSpec("maskppo_ugv_seed0", "MaskablePPO-UGV seed0", "MaskablePPO", ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_ugv_seed0"),
    CurveSpec("maskppo_full_seed0", "Full MaskablePPO seed0", "MaskablePPO", ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_full_seed0"),
    CurveSpec(
        "maskppo_uav_default_ft",
        "MaskablePPO-UAV default fine-tune",
        "MaskablePPO Repair",
        ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_uav_seed4_conservative_ft",
        "Loaded model accidentally kept default PPO hyperparameters; included as negative comparison.",
    ),
    CurveSpec(
        "maskppo_uav_conservative_ft",
        "MaskablePPO-UAV conservative fine-tune",
        "MaskablePPO Repair",
        ROOT / "outputs/logs/maskable_ppo_multi/multi_hard_random_guided_uav_seed4_true_conservative_ft",
    ),
]

FACTOR_METRICS = [
    (
        "Factorized DQN explore",
        ROOT / "outputs/logs/factorized_multi/multi_hard_random_explore_seed0_coverage/training_metrics.csv",
    ),
    (
        "Factorized DQN guided UAV",
        ROOT / "outputs/logs/factorized_multi/multi_hard_random_guided_seed0_coverage/training_metrics.csv",
    ),
    (
        "Factorized DQN guided UGV",
        ROOT / "outputs/logs/factorized_multi/multi_hard_random_guided_seed0_coverage_ugv/training_metrics.csv",
    ),
]

GROUP_FILES = {
    "Single-UGV DQN": "01_single_ugv_dqn_returns.png",
    "Multi-UGV DQN": "02_multi_ugv_dqn_returns.png",
    "Expert-Regularized DQN": "03_expert_regularized_dqn_returns.png",
    "PPO": "04_ppo_returns.png",
    "MaskablePPO": "05_maskable_ppo_returns.png",
    "MaskablePPO Repair": "06_maskable_ppo_repair_returns.png",
}

OVERVIEW_KEYS = [
    "dqn_medium_seed0",
    "dqn_hard_seed1",
    "ppo_uav_seed0",
    "maskppo_uav_seed0",
    "maskppo_uav_conservative_ft",
]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#f7f8fa"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["axes.edgecolor"] = "#d1d5db"
    plt.rcParams["axes.labelcolor"] = "#374151"
    plt.rcParams["xtick.color"] = "#374151"
    plt.rcParams["ytick.color"] = "#374151"
    plt.rcParams["grid.color"] = "#e5e7eb"
    plt.rcParams["legend.frameon"] = False


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_monitor(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, comment="#")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty or "r" not in df.columns or "l" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["r", "l"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["episode"] = range(1, len(df) + 1)
    df["timesteps"] = pd.to_numeric(df["l"], errors="coerce").fillna(0).cumsum()
    df["episode_return"] = pd.to_numeric(df["r"], errors="coerce")
    df["episode_length"] = pd.to_numeric(df["l"], errors="coerce")
    return df[["episode", "timesteps", "episode_return", "episode_length"]].dropna()


def read_step_rewards(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    required = {"timesteps", "reward", "done"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    total = 0.0
    length = 0
    episode = 0
    for item in df.itertuples(index=False):
        total += float(item.reward)
        length += 1
        if int(item.done) == 1:
            episode += 1
            rows.append(
                {
                    "episode": episode,
                    "timesteps": int(item.timesteps),
                    "episode_return": total,
                    "episode_length": length,
                }
            )
            total = 0.0
            length = 0
    return pd.DataFrame(rows)


def read_curve(spec: CurveSpec) -> tuple[pd.DataFrame, str]:
    monitor = read_monitor(spec.log_dir / "monitor.csv")
    if len(monitor) >= 3:
        return monitor, "monitor.csv"
    reconstructed = read_step_rewards(spec.log_dir / "step_rewards.csv")
    if len(reconstructed) >= 3:
        return reconstructed, "step_rewards.csv"
    if len(monitor) > 0:
        return monitor, "monitor.csv (short)"
    return pd.DataFrame(), "missing"


def add_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timesteps").copy()
    window = max(3, min(20, len(df) // 8 if len(df) >= 32 else 5))
    df["smoothed_return"] = df["episode_return"].rolling(window, min_periods=1).mean()
    df["rolling_std"] = df["episode_return"].rolling(window, min_periods=2).std().fillna(0.0)
    df["window"] = window
    return df


def save_individual_curve(df: pd.DataFrame, spec: CurveSpec, source: str) -> None:
    if df.empty:
        return
    df = add_smoothing(df)
    fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=180)
    color = "#2563eb"
    ax.plot(df["timesteps"], df["episode_return"], color=color, alpha=0.18, linewidth=0.9, label="Episode return")
    ax.plot(df["timesteps"], df["smoothed_return"], color=color, linewidth=2.4, label=f"Rolling mean ({int(df['window'].iloc[0])} ep)")
    lower = df["smoothed_return"] - 0.45 * df["rolling_std"]
    upper = df["smoothed_return"] + 0.45 * df["rolling_std"]
    ax.fill_between(df["timesteps"].to_numpy(), lower.to_numpy(), upper.to_numpy(), color=color, alpha=0.10, linewidth=0)
    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.35)
    ax.set_title(spec.label, fontsize=15, weight="bold", color="#111827", pad=12)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode return")
    subtitle = f"group: {spec.group} | source: {source} | episodes: {len(df)}"
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#64748b", va="bottom")
    ax.legend(loc="best", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(INDIVIDUAL_DIR / f"{spec.key}_return_curve.png")
    plt.close(fig)


def save_group_curve(df: pd.DataFrame, group: str, filename: str) -> None:
    group_df = df[df["group"] == group].copy()
    if group_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11.8, 6.2), dpi=180)
    palette = sns.color_palette("tab10", n_colors=max(3, group_df["label"].nunique()))
    for (label, part), color in zip(group_df.groupby("label", sort=False), palette):
        part = add_smoothing(part)
        ax.plot(part["timesteps"], part["smoothed_return"], linewidth=2.0, label=label, color=color)
        ax.plot(part["timesteps"], part["episode_return"], linewidth=0.6, alpha=0.08, color=color)
    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.35)
    ax.set_title(f"{group}: Training Episode Return", fontsize=15, weight="bold", color="#111827", pad=12)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode return")
    ax.legend(loc="best", fontsize=8, ncol=1)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)


def save_overview(df: pd.DataFrame) -> None:
    overview = df[df["key"].isin(OVERVIEW_KEYS)].copy()
    if overview.empty:
        return
    fig, ax = plt.subplots(figsize=(12.2, 6.2), dpi=180)
    colors = {
        "dqn_medium_seed0": "#4c78a8",
        "dqn_hard_seed1": "#2f6f4e",
        "ppo_uav_seed0": "#d95f5f",
        "maskppo_uav_seed0": "#36a3a1",
        "maskppo_uav_conservative_ft": "#157f5b",
    }
    for key in OVERVIEW_KEYS:
        part = overview[overview["key"] == key]
        if part.empty:
            continue
        label = str(part["label"].iloc[0])
        part = add_smoothing(part)
        ax.plot(part["timesteps"], part["smoothed_return"], linewidth=2.3, label=label, color=colors.get(key))
    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.35)
    ax.set_title("Representative Training Return Curves", fontsize=15, weight="bold", color="#111827", pad=12)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode return")
    ax.legend(loc="best", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "00_representative_training_returns.png")
    plt.close(fig)


def save_factorized_diagnostics() -> None:
    frames = []
    for label, path in FACTOR_METRICS:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["label"] = label
        frame["stage_index"] = range(1, len(frame) + 1)
        frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), dpi=180)
    metrics = [("loss", "BC / DAgger Loss"), ("head_acc", "Per-head Accuracy"), ("joint_acc", "Joint Action Accuracy")]
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in data.columns:
            continue
        sns.lineplot(data=data, x="stage_index", y=metric, hue="label", marker="o", linewidth=2.0, ax=ax)
        ax.set_title(title, fontsize=12, weight="bold", color="#111827")
        ax.set_xlabel("Training stage")
        ax.set_ylabel(title)
        ax.set_xticks(sorted(data["stage_index"].unique()))
        ax.legend(title="", fontsize=7)
        sns.despine(ax=ax)
    fig.suptitle("Factorized DQN / DAgger Training Diagnostics", fontsize=15, weight="bold", color="#111827")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_factorized_dagger_training_diagnostics.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs(OUT_DIR, INDIVIDUAL_DIR, DATA_DIR)
    set_style()
    all_frames = []
    coverage_rows = []
    for spec in CURVES:
        curve, source = read_curve(spec)
        coverage_rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "group": spec.group,
                "log_dir": str(spec.log_dir.relative_to(ROOT)),
                "source": source,
                "episodes": int(len(curve)),
                "note": spec.note,
            }
        )
        if curve.empty:
            continue
        curve = curve.copy()
        curve["key"] = spec.key
        curve["label"] = spec.label
        curve["group"] = spec.group
        all_frames.append(curve)
        save_individual_curve(curve, spec, source)

    coverage_path = DATA_DIR / "training_curve_coverage_report.csv"
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coverage_rows)

    if not all_frames:
        raise FileNotFoundError("No training curves could be read from outputs/logs.")

    data = pd.concat(all_frames, ignore_index=True)
    data_path = DATA_DIR / "training_return_curves_long.csv"
    data.to_csv(data_path, index=False)

    for group, filename in GROUP_FILES.items():
        save_group_curve(data, group, filename)
    save_overview(data)
    save_factorized_diagnostics()

    print("Saved report training curves to:", OUT_DIR.relative_to(ROOT))
    print("Saved individual curves to:", INDIVIDUAL_DIR.relative_to(ROOT))
    print("Saved curve data:", data_path.relative_to(ROOT))
    print("Saved coverage report:", coverage_path.relative_to(ROOT))
    printable = pd.DataFrame(coverage_rows)
    print(printable[["group", "label", "source", "episodes"]].to_string(index=False))


if __name__ == "__main__":
    main()

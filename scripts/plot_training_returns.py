from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from fire_rescue_rl.utils.config import ensure_dirs


DATA_PATH = ROOT / "outputs" / "eval" / "metrics_csv" / "training_return_curves_long.csv"
OUT_DIR = ROOT / "outputs" / "figures" / "report_training_curves_polished"
INDIVIDUAL_DIR = OUT_DIR / "individual"

GROUP_ORDER = [
    ("Single-UGV DQN", "01_single_ugv_dqn_returns_polished.png"),
    ("Multi-UGV DQN", "02_multi_ugv_dqn_returns_polished.png"),
    ("Expert-Regularized DQN", "03_expert_regularized_dqn_returns_polished.png"),
    ("PPO", "04_ppo_returns_polished.png"),
    ("MaskablePPO", "05_maskable_ppo_returns_polished.png"),
    ("MaskablePPO Repair", "06_maskable_ppo_repair_returns_polished.png"),
]

OVERVIEW_KEYS = [
    "dqn_medium_seed0",
    "dqn_hard_seed1",
    "ppo_uav_seed0",
    "maskppo_uav_seed0",
    "maskppo_uav_conservative_ft",
]

LABEL_ALIASES = {
    "DQN Easy seed0": "DQN Easy s0",
    "DQN Easy seed1": "DQN Easy s1",
    "DQN Easy seed2": "DQN Easy s2",
    "DQN Medium seed0": "DQN Medium",
    "DQN Hard seed0": "DQN Hard s0",
    "DQN Hard seed1": "DQN Hard s1",
    "Multi-UGV DQN Easy": "Multi-UGV Easy",
    "Expert-Regularized DQN Easy": "ER-DQN Easy",
    "Expert-Regularized DQN Medium": "ER-DQN Medium",
    "Expert-Regularized DQN Hard": "ER-DQN Hard",
    "PPO-UAV seed0": "PPO-UAV s0",
    "PPO-UAV seed1": "PPO-UAV s1",
    "PPO-UAV seed2": "PPO-UAV s2",
    "PPO-UGV seed0": "PPO-UGV",
    "Full PPO seed0": "Full PPO",
    "MaskablePPO-UAV seed0": "MaskPPO-UAV",
    "MaskablePPO-UGV seed0": "MaskPPO-UGV",
    "Full MaskablePPO seed0": "Full MaskPPO",
    "MaskablePPO-UAV default fine-tune": "Default fine-tune",
    "MaskablePPO-UAV conservative fine-tune": "Conservative fine-tune",
}

GROUP_TITLES = {
    "Single-UGV DQN": "Single-UGV DQN Training Returns",
    "Multi-UGV DQN": "Multi-UGV DQN Training Returns",
    "Expert-Regularized DQN": "Expert-Regularized DQN Training Returns",
    "PPO": "PPO Variant Training Returns",
    "MaskablePPO": "MaskablePPO Variant Training Returns",
    "MaskablePPO Repair": "MaskablePPO Fine-Tuning Comparison",
}

COLOR_MAP = {
    "dqn_easy_seed0": "#9ca3af",
    "dqn_easy_seed1": "#60a5fa",
    "dqn_easy_seed2": "#2563eb",
    "dqn_medium_seed0": "#0f766e",
    "dqn_hard_seed0": "#7c3aed",
    "dqn_hard_seed1": "#166534",
    "multi_dqn_easy": "#2563eb",
    "dqn_er_easy": "#60a5fa",
    "dqn_er_medium": "#0f766e",
    "dqn_er_hard": "#7c3aed",
    "ppo_uav_seed0": "#2563eb",
    "ppo_uav_seed1": "#f59e0b",
    "ppo_uav_seed2": "#10b981",
    "ppo_ugv_seed0": "#dc2626",
    "ppo_full_seed0": "#7c3aed",
    "maskppo_uav_seed0": "#2563eb",
    "maskppo_ugv_seed0": "#f97316",
    "maskppo_full_seed0": "#16a34a",
    "maskppo_uav_default_ft": "#ef4444",
    "maskppo_uav_conservative_ft": "#157f5b",
}


def set_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#f5f7fb",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cfd6df",
            "axes.labelcolor": "#263241",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "grid.color": "#e6eaf0",
            "grid.linewidth": 0.85,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "savefig.facecolor": "#f5f7fb",
        }
    )


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run scripts/collect_training_returns.py first: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["short_label"] = df["label"].map(LABEL_ALIASES).fillna(df["label"])
    return df


def smooth_curve(part: pd.DataFrame) -> pd.DataFrame:
    part = part.sort_values("timesteps").copy()
    if len(part) < 3:
        part["smooth"] = part["episode_return"]
        return part
    span = max(5, min(40, int(len(part) / 6)))
    part["smooth"] = part["episode_return"].ewm(span=span, adjust=False).mean()
    return part


def format_k(value: float, _pos: int | None = None) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def prepare_axis(ax, *, show_zero: bool = True) -> None:
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cfd6df")
    ax.spines["bottom"].set_color("#cfd6df")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    if show_zero:
        ax.axhline(0, color="#94a3b8", linewidth=0.9, alpha=0.65, zorder=0)


def add_chart_title(ax, title: str, subtitle: str) -> None:
    ax.text(
        0.0,
        1.135,
        title,
        transform=ax.transAxes,
        fontsize=13.5,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.0,
        1.085,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.4,
        color="#64748b",
        ha="left",
        va="bottom",
    )


def endpoint_positions(items: list[tuple[str, float, float, str]], y_min: float, y_max: float) -> list[tuple[str, float, float, str]]:
    if not items:
        return []
    value_range = max(1.0, y_max - y_min)
    min_gap = value_range * 0.055
    ordered = sorted(items, key=lambda item: item[2])
    adjusted: list[list[object]] = []
    for key, x, y, color in ordered:
        y_adj = y
        if adjusted and y_adj - float(adjusted[-1][2]) < min_gap:
            y_adj = float(adjusted[-1][2]) + min_gap
        adjusted.append([key, x, y_adj, color])
    overflow = float(adjusted[-1][2]) - y_max
    if overflow > 0:
        for item in adjusted:
            item[2] = float(item[2]) - overflow
    underflow = y_min - float(adjusted[0][2])
    if underflow > 0:
        for item in adjusted:
            item[2] = float(item[2]) + underflow
    return [(str(key), float(x), float(y), str(color)) for key, x, y, color in adjusted]


def add_direct_labels(ax, label_items: list[tuple[str, float, float, str]], x_pad: float) -> None:
    y_min, y_max = ax.get_ylim()
    adjusted = endpoint_positions(label_items, y_min, y_max)
    for label, x, y, color in adjusted:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + x_pad, y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.2,
            color=color,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.65},
            clip_on=False,
        )


def save_group_plot(df: pd.DataFrame, group: str, filename: str) -> None:
    part_df = df[df["group"] == group].copy()
    if part_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=220)
    label_items: list[tuple[str, float, float, str]] = []
    x_max = float(part_df["timesteps"].max())
    x_min = float(part_df["timesteps"].min())
    for key, part in part_df.groupby("key", sort=False):
        part = smooth_curve(part)
        color = COLOR_MAP.get(key, "#334155")
        label = str(part["short_label"].iloc[0])
        ax.plot(part["timesteps"], part["smooth"], color=color, linewidth=2.35, alpha=0.96, solid_capstyle="round")
        if len(part) >= 20:
            raw = part.iloc[:: max(1, len(part) // 90)]
            ax.scatter(raw["timesteps"], raw["episode_return"], color=color, s=7, alpha=0.075, linewidth=0)
        last = part.iloc[-1]
        label_items.append((label, float(last["timesteps"]), float(last["smooth"]), color))

    prepare_axis(ax)
    ax.margins(y=0.12)
    add_chart_title(ax, GROUP_TITLES.get(group, group), "Smoothed episode returns from real training logs")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Episode return")
    x_pad = max(1.0, (x_max - x_min) * 0.035)
    ax.set_xlim(left=max(0.0, x_min - (x_max - x_min) * 0.02), right=x_max + x_pad * 5.6)
    add_direct_labels(ax, label_items, x_pad)
    fig.tight_layout(pad=1.1)
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_overview(df: pd.DataFrame) -> None:
    overview = df[df["key"].isin(OVERVIEW_KEYS)].copy()
    if overview.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=220)
    label_items: list[tuple[str, float, float, str]] = []
    x_max = float(overview["timesteps"].max())
    x_min = float(overview["timesteps"].min())
    for key in OVERVIEW_KEYS:
        part = overview[overview["key"] == key].copy()
        if part.empty:
            continue
        part = smooth_curve(part)
        color = COLOR_MAP.get(key, "#334155")
        label = str(part["short_label"].iloc[0])
        ax.plot(part["timesteps"], part["smooth"], color=color, linewidth=2.5, alpha=0.96, solid_capstyle="round")
        last = part.iloc[-1]
        label_items.append((label, float(last["timesteps"]), float(last["smooth"]), color))
    prepare_axis(ax)
    ax.margins(y=0.12)
    add_chart_title(
        ax,
        "Representative Training Return Curves",
        "A compact comparison of the main training stages used in the report",
    )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Episode return")
    x_pad = max(1.0, (x_max - x_min) * 0.035)
    ax.set_xlim(left=max(0.0, x_min - (x_max - x_min) * 0.02), right=x_max + x_pad * 6.4)
    add_direct_labels(ax, label_items, x_pad)
    fig.tight_layout(pad=1.1)
    fig.savefig(OUT_DIR / "00_representative_training_returns_polished.png", bbox_inches="tight")
    plt.close(fig)


def save_small_multiples(df: pd.DataFrame) -> None:
    groups = [group for group, _filename in GROUP_ORDER if not df[df["group"] == group].empty]
    if not groups:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.3), dpi=220)
    axes = axes.flatten()
    for ax, group in zip(axes, groups):
        part_df = df[df["group"] == group].copy()
        for key, part in part_df.groupby("key", sort=False):
            part = smooth_curve(part)
            color = COLOR_MAP.get(key, "#334155")
            ax.plot(part["timesteps"], part["smooth"], color=color, linewidth=1.65, alpha=0.96)
        prepare_axis(ax)
        ax.set_title(group, loc="left", fontsize=10.5, color="#111827", pad=7)
        ax.set_xlabel("steps")
        ax.set_ylabel("return")
    for ax in axes[len(groups) :]:
        ax.axis("off")
    fig.suptitle(
        "Training Return Overview by Experiment Line",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(OUT_DIR / "08_all_training_returns_small_multiples.png", bbox_inches="tight")
    plt.close(fig)


def save_individual(df: pd.DataFrame) -> None:
    for key, part in df.groupby("key", sort=False):
        part = smooth_curve(part)
        label = str(part["short_label"].iloc[0])
        group = str(part["group"].iloc[0])
        color = COLOR_MAP.get(key, "#2563eb")
        fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=220)
        if len(part) >= 12:
            raw = part.iloc[:: max(1, len(part) // 130)]
            ax.scatter(raw["timesteps"], raw["episode_return"], color=color, s=9, alpha=0.11, linewidth=0, label="episode")
        ax.plot(part["timesteps"], part["smooth"], color=color, linewidth=2.45, solid_capstyle="round", label="smoothed")
        prepare_axis(ax)
        final_return = float(part["smooth"].iloc[-1])
        ax.margins(y=0.12)
        add_chart_title(
            ax,
            label,
            f"{group} | episodes={len(part)} | final smoothed return={final_return:.1f}",
        )
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Episode return")
        ax.legend(loc="best", fontsize=7.6)
        fig.tight_layout(pad=1.0)
        fig.savefig(INDIVIDUAL_DIR / f"{key}_return_curve_polished.png", bbox_inches="tight")
        plt.close(fig)


def save_factorized_diagnostics() -> None:
    paths = [
        ("Explore", ROOT / "outputs/logs/factorized_multi/multi_hard_random_explore_seed0_coverage/training_metrics.csv", "#2563eb"),
        ("Guided UAV", ROOT / "outputs/logs/factorized_multi/multi_hard_random_guided_seed0_coverage/training_metrics.csv", "#157f5b"),
        ("Guided UGV", ROOT / "outputs/logs/factorized_multi/multi_hard_random_guided_seed0_coverage_ugv/training_metrics.csv", "#f97316"),
    ]
    frames = []
    for label, path, color in paths:
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if data.empty:
            continue
        data["label"] = label
        data["color"] = color
        data["stage_index"] = np.arange(1, len(data) + 1)
        frames.append(data)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), dpi=220)
    metrics = [("loss", "Loss"), ("head_acc", "Head accuracy"), ("joint_acc", "Joint accuracy")]
    for ax, (metric, title) in zip(axes, metrics):
        for label, part in data.groupby("label", sort=False):
            color = str(part["color"].iloc[0])
            ax.plot(part["stage_index"], part[metric], marker="o", markersize=4.2, linewidth=2.0, color=color, label=label)
        prepare_axis(ax, show_zero=False)
        ax.set_title(title, loc="left", fontsize=10.5, color="#111827", pad=7)
        ax.set_xlabel("stage")
        ax.set_ylabel(title)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.2)
    fig.suptitle("Factorized DQN / DAgger Training Diagnostics", x=0.02, ha="left", fontsize=13, fontweight="bold", color="#111827")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(OUT_DIR / "07_factorized_dagger_training_diagnostics_polished.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs(OUT_DIR, INDIVIDUAL_DIR)
    set_style()
    df = load_data()
    for group, filename in GROUP_ORDER:
        save_group_plot(df, group, filename)
    save_overview(df)
    save_small_multiples(df)
    save_individual(df)
    save_factorized_diagnostics()
    print("Saved polished training curves to:", OUT_DIR.relative_to(ROOT))
    print("Saved polished individual curves to:", INDIVIDUAL_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()

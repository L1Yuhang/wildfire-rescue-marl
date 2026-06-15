from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all multi-UGV DQN experiment summaries.")
    parser.add_argument("--metrics-dir", default="outputs/eval/metrics_csv")
    parser.add_argument("--out-dir", default="outputs/figures/multi_ugv")
    return parser.parse_args()


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_summaries(metrics_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(metrics_dir.glob("multi_summary_multi_*_seed*.csv")):
        frame = pd.read_csv(path)
        stem = path.stem
        # multi_summary_multi_easy_seed0 -> multi_easy
        difficulty = stem.replace("multi_summary_", "").rsplit("_seed", 1)[0]
        frame["difficulty"] = difficulty.replace("multi_", "")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.4), dpi=150)
    sns.barplot(data=df, x="difficulty", y=metric, hue="agent", order=["easy", "medium", "hard"])
    plt.title(title, fontsize=12, weight="bold")
    plt.xlabel("Difficulty")
    plt.ylabel(ylabel)
    plt.legend(title="Policy", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    metrics_dir = ROOT / args.metrics_dir
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    df = load_summaries(metrics_dir)
    if df.empty:
        raise FileNotFoundError("No multi_summary_multi_*_seed*.csv files found.")
    set_style()
    df.to_csv(metrics_dir / "multi_summary_all.csv", index=False)
    save_metric(df, "success", "Multi-UGV Success Rate", "Success Rate", out_dir / "multi_all_success_rate.png")
    save_metric(df, "reward", "Multi-UGV Average Reward", "Average Reward", out_dir / "multi_all_average_reward.png")
    save_metric(df, "steps", "Multi-UGV Average Steps", "Steps", out_dir / "multi_all_average_steps.png")
    save_metric(
        df,
        "delivered_count",
        "Multi-UGV Delivered Survivors",
        "Delivered Survivors",
        out_dir / "multi_all_delivered_count.png",
    )
    save_metric(
        df,
        "risk_exposure",
        "Multi-UGV Risk Exposure",
        "Risk Exposure",
        out_dir / "multi_all_risk_exposure.png",
    )
    print("Saved combined multi-UGV figures to:", out_dir.relative_to(ROOT))


if __name__ == "__main__":
    main()

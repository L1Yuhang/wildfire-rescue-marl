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
    parser = argparse.ArgumentParser(description="Plot multi-UGV initial experiment results.")
    parser.add_argument("--summary", default="outputs/eval/metrics_csv/multi_baselines_summary.csv")
    parser.add_argument("--detail", default="outputs/eval/metrics_csv/multi_baselines_detail.csv")
    parser.add_argument("--out-dir", default="outputs/figures/multi_ugv")
    return parser.parse_args()


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def save_bar(summary: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path) -> None:
    order = ["random", "greedy", "astar"]
    colors = ["#9ca3af", "#ef6c00", "#1565c0"]
    plt.figure(figsize=(6.4, 4.1), dpi=150)
    sns.barplot(data=summary, x="agent", y=metric, order=order, palette=colors, hue="agent", legend=False)
    plt.title(title, fontsize=12, weight="bold")
    plt.xlabel("Policy")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_delivery_box(detail: pd.DataFrame, path: Path) -> None:
    order = ["random", "greedy", "astar"]
    plt.figure(figsize=(6.4, 4.1), dpi=150)
    sns.boxplot(data=detail, x="agent", y="delivered_count", order=order, color="#d6d9de", width=0.55)
    sns.stripplot(data=detail, x="agent", y="delivered_count", order=order, color="#1565c0", size=3, alpha=0.55)
    plt.title("Delivered Survivors per Episode", fontsize=12, weight="bold")
    plt.xlabel("Policy")
    plt.ylabel("Delivered Survivors")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    summary_path = ROOT / args.summary
    detail_path = ROOT / args.detail
    out_dir = ROOT / args.out_dir
    ensure_dirs(out_dir)
    if not summary_path.exists() or not detail_path.exists():
        raise FileNotFoundError("Run scripts/eval_multi_baselines.py before plotting.")
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    set_style()
    save_bar(summary, "success", "Multi-UGV Success Rate", "Success Rate", out_dir / "multi_success_rate.png")
    save_bar(
        summary,
        "delivered_count",
        "Average Delivered Survivors",
        "Delivered Survivors",
        out_dir / "multi_delivered_count.png",
    )
    save_bar(summary, "steps", "Average Episode Steps", "Steps", out_dir / "multi_average_steps.png")
    save_bar(summary, "risk_exposure", "Average Risk Exposure", "Risk Exposure", out_dir / "multi_risk_exposure.png")
    save_delivery_box(detail, out_dir / "multi_delivery_distribution.png")
    print("Saved multi-UGV figures to:", out_dir.relative_to(ROOT))


if __name__ == "__main__":
    main()

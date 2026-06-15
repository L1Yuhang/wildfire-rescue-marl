from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from fire_rescue_rl.utils.config import ensure_dirs
from fire_rescue_rl.utils.metrics import summarize_metrics
from fire_rescue_rl.utils.plotting import save_metric_bars, save_training_curve


def main() -> None:
    figure_dir = ROOT / "outputs" / "figures"
    metrics_dir = ROOT / "outputs" / "eval" / "metrics_csv"
    ensure_dirs(figure_dir)
    metric_files = sorted(metrics_dir.glob("all_*_seed*.csv")) if metrics_dir.exists() else []
    if metric_files:
        full = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
        summarize_metrics(full).to_csv(metrics_dir / "summary_all.csv", index=False)
        save_metric_bars(full, figure_dir)
        print(f"Saved metric figures to {figure_dir}")
    else:
        print("No evaluation CSV files found.")
    log_files = sorted((ROOT / "outputs" / "logs").glob("*/*/step_rewards.csv"))
    if log_files:
        save_training_curve(log_files, figure_dir / "training_reward_curve.png")
        print("Saved training curve.")
    else:
        print("No training logs found.")


if __name__ == "__main__":
    main()


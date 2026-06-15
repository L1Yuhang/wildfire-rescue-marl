"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def save_metric_bars(df: pd.DataFrame, output_dir: str | Path) -> None:
    set_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    metrics = [
        ("success", "Success Rate", "eval_success_rate.png"),
        ("reward", "Average Reward", "eval_average_reward.png"),
        ("steps", "Average Steps", "eval_average_steps.png"),
        ("risk_exposure", "Risk Exposure", "eval_risk_exposure.png"),
        ("fire_collision", "Fire Collision Rate", "eval_fire_collision.png"),
        ("survivor_burned", "Survivor Burned Rate", "eval_survivor_burned.png"),
    ]
    for metric, title, filename in metrics:
        plt.figure(figsize=(7.0, 4.2), dpi=140)
        sns.barplot(data=df, x="agent", y=metric, hue="difficulty", errorbar="sd")
        plt.title(title)
        plt.xlabel("Agent")
        plt.ylabel(title)
        plt.tight_layout()
        plt.savefig(output / filename)
        plt.close()


def save_training_curve(log_files: Iterable[Path], output_path: str | Path) -> None:
    set_style()
    frames = []
    for path in log_files:
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame["run"] = path.parent.name
                frame["rolling_reward"] = frame["reward"].rolling(500, min_periods=1).mean()
                if len(frame) > 5000:
                    frame = frame.iloc[:: max(1, len(frame) // 5000)].copy()
                frames.append(frame)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    plt.figure(figsize=(8.0, 4.4), dpi=140)
    sns.lineplot(data=df, x="timesteps", y="rolling_reward", hue="run", linewidth=1.3)
    plt.title("Training Reward Curve")
    plt.xlabel("Timesteps")
    plt.ylabel("Rolling Step Reward")
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()

"""Metrics for fire rescue evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class EpisodeMetrics:
    agent: str
    difficulty: str
    episode: int
    reward: float
    steps: int
    success: int
    discovered: int
    picked: int
    fire_collision: int
    survivor_burned: int
    timeout: int
    invalid_uav: int
    invalid_ugv: int
    risk_exposure: float
    uav_path_length: int
    ugv_path_length: int


class MetricsAccumulator:
    def __init__(self, agent: str, difficulty: str) -> None:
        self.agent = agent
        self.difficulty = difficulty
        self.rows: List[EpisodeMetrics] = []

    def add_episode(self, episode: int, rollout: Dict[str, Any]) -> None:
        infos = rollout["infos"]
        last = infos[-1] if infos else {}
        reason = last.get("terminated_reason", "")
        row = EpisodeMetrics(
            agent=self.agent,
            difficulty=self.difficulty,
            episode=episode,
            reward=float(rollout["reward"]),
            steps=int(rollout["steps"]),
            success=1 if last.get("success") else 0,
            discovered=1 if any(info.get("discover") for info in infos) or last.get("survivor_discovered") else 0,
            picked=1 if any(info.get("pickup") for info in infos) or last.get("survivor_picked") else 0,
            fire_collision=1 if reason == "ugv_fire_collision" else 0,
            survivor_burned=1 if reason == "survivor_burned" else 0,
            timeout=1 if reason == "timeout" or rollout.get("truncated") else 0,
            invalid_uav=sum(1 for info in infos if info.get("invalid_uav")),
            invalid_ugv=sum(1 for info in infos if info.get("invalid_ugv")),
            risk_exposure=float(last.get("risk_exposure", 0.0)),
            uav_path_length=_path_length(last.get("uav_path", [])),
            ugv_path_length=_path_length(last.get("ugv_path", [])),
        )
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(row) for row in self.rows])


def _path_length(path) -> int:
    if len(path) < 2:
        return 0
    return sum(1 for a, b in zip(path[:-1], path[1:]) if a != b)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "reward",
        "steps",
        "success",
        "discovered",
        "picked",
        "fire_collision",
        "survivor_burned",
        "timeout",
        "invalid_uav",
        "invalid_ugv",
        "risk_exposure",
        "uav_path_length",
        "ugv_path_length",
    ]
    return df.groupby(["agent", "difficulty"], as_index=False)[metrics].mean()


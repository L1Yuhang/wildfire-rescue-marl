"""Reward weights."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict


@dataclass
class RewardWeights:
    discover: float = 30.0
    pickup: float = 45.0
    success: float = 120.0
    uav_progress: float = 0.8
    ugv_progress: float = 1.0
    regress: float = -0.8
    time: float = -0.08
    wait: float = -0.05
    invalid_uav: float = -3.0
    invalid_ugv: float = -6.0
    fire_collision: float = -80.0
    survivor_burned: float = -100.0
    smoke_risk: float = -0.8
    near_fire_risk: float = -2.5
    coordination: float = -0.1

    @classmethod
    def from_dict(cls, values: Dict[str, float] | None) -> "RewardWeights":
        if not values:
            return cls()
        names = {field.name for field in fields(cls)}
        clean = {key: float(value) for key, value in values.items() if key in names}
        return cls(**clean)


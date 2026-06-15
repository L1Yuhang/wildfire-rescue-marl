"""Fire and smoke dynamics."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

GridPos = Tuple[int, int]

NEIGHBORS: Tuple[GridPos, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def spread_fire(
    fire_cells: set[GridPos],
    blocked: set[GridPos],
    water: set[GridPos],
    height: int,
    width: int,
    spread_prob: float,
    max_fire_cells: int,
    rng: np.random.Generator,
) -> set[GridPos]:
    if len(fire_cells) >= max_fire_cells or spread_prob <= 0:
        return set(fire_cells)
    new_fire = set(fire_cells)
    candidates: list[GridPos] = []
    for row, col in fire_cells:
        for dr, dc in NEIGHBORS:
            pos = (row + dr, col + dc)
            if pos in new_fire or pos in blocked or pos in water:
                continue
            if pos[0] <= 0 or pos[0] >= height - 1 or pos[1] <= 0 or pos[1] >= width - 1:
                continue
            candidates.append(pos)
    candidates = sorted(set(candidates))
    for pos in candidates:
        if len(new_fire) >= max_fire_cells:
            break
        if rng.random() <= spread_prob:
            new_fire.add(pos)
    return new_fire


def smoke_from_fire(fire_cells: Iterable[GridPos], blocked: set[GridPos], height: int, width: int, radius: int) -> set[GridPos]:
    smoke: set[GridPos] = set()
    for row, col in fire_cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                pos = (row + dr, col + dc)
                if pos in blocked or pos in fire_cells:
                    continue
                if 0 <= pos[0] < height and 0 <= pos[1] < width:
                    smoke.add(pos)
    return smoke


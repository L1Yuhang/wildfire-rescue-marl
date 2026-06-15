"""Procedural maps for the multi-UGV rescue environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

EMPTY = 0
WALL = 1
ROAD = 2
DEBRIS = 3
BURNED = 4

GridPos = tuple[int, int]


@dataclass(frozen=True)
class GeneratedRescueMap:
    terrain: np.ndarray
    base_pos: GridPos
    survivor_positions: list[GridPos]
    fire_positions: list[GridPos]
    seed: int

    @property
    def height(self) -> int:
        return int(self.terrain.shape[0])

    @property
    def width(self) -> int:
        return int(self.terrain.shape[1])


def generate_valid_rescue_map(
    *,
    width: int = 28,
    height: int = 28,
    seed: int = 0,
    num_survivors: int = 3,
    num_fire_sources: int = 3,
    road_spacing: int = 6,
    wall_block_prob: float = 0.38,
    debris_ratio: float = 0.07,
    min_survivor_base_distance: int = 10,
    min_survivor_fire_distance: int = 6,
    max_attempts: int = 200,
) -> GeneratedRescueMap:
    """Generate a sparse city map and reject unsolvable samples.

    The generator intentionally keeps obstacles less continuous than the fixed
    maps used by the first project stage.  A road skeleton is laid down first,
    then small building/debris patches are added.  Each candidate map is kept
    only if every survivor is reachable from the base by a ground vehicle before
    fire spreads.
    """
    if width < 20 or height < 20:
        raise ValueError("width and height should both be at least 20")

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt * 9973)
        terrain = np.full((height, width), EMPTY, dtype=np.int16)
        terrain[0, :] = WALL
        terrain[-1, :] = WALL
        terrain[:, 0] = WALL
        terrain[:, -1] = WALL

        base_pos = (height - 3, 2)
        _add_roads(terrain, base_pos=base_pos, spacing=road_spacing)
        _add_sparse_blocks(terrain, rng=rng, spacing=road_spacing, wall_block_prob=wall_block_prob)
        _add_sparse_debris(terrain, rng=rng, ratio=debris_ratio, base_pos=base_pos)
        _clear_base_area(terrain, base_pos)

        reachable = _reachable_cells(terrain, base_pos, blocked_values={WALL, DEBRIS})
        if len(reachable) < max(30, width * height // 3):
            continue

        fire_positions = _choose_fire_positions(
            terrain,
            rng=rng,
            reachable=reachable,
            base_pos=base_pos,
            count=num_fire_sources,
        )
        if len(fire_positions) < num_fire_sources:
            continue

        survivor_positions = _choose_survivor_positions(
            terrain,
            rng=rng,
            reachable=reachable,
            base_pos=base_pos,
            fire_positions=fire_positions,
            count=num_survivors,
            min_base_distance=min_survivor_base_distance,
            min_fire_distance=min_survivor_fire_distance,
        )
        if len(survivor_positions) < num_survivors:
            continue

        if _all_reachable(terrain, base_pos, survivor_positions, fire_positions):
            return GeneratedRescueMap(
                terrain=terrain,
                base_pos=base_pos,
                survivor_positions=survivor_positions,
                fire_positions=fire_positions,
                seed=seed + attempt * 9973,
            )

    raise RuntimeError(f"Failed to generate a valid map after {max_attempts} attempts")


def save_generated_map_preview(
    rescue_map: GeneratedRescueMap,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    display = rescue_map.terrain.copy()
    for row, col in rescue_map.fire_positions:
        display[row, col] = 5
    for row, col in rescue_map.survivor_positions:
        display[row, col] = 6
    br, bc = rescue_map.base_pos
    display[br, bc] = 7

    cmap = mcolors.ListedColormap(
        [
            "#f8fafc",  # empty
            "#30343b",  # wall/building
            "#d4d7dc",  # road
            "#8a6f4d",  # debris
            "#5b6472",  # burned debris
            "#e53935",  # fire
            "#ffd54f",  # survivor
            "#43a047",  # base
        ]
    )
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8.6, 8.6), dpi=160, facecolor="#f4f6f8")
    ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title or f"Generated Multi-UGV Rescue Map (seed={rescue_map.seed})")
    ax.set_xticks(np.arange(-0.5, rescue_map.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rescue_map.height, 1), minor=True)
    ax.set_xticks(np.arange(-0.5, rescue_map.width, 5))
    ax.set_yticks(np.arange(-0.5, rescue_map.height, 5))
    ax.grid(which="minor", color="#ffffff", linewidth=0.25, alpha=0.35)
    ax.grid(which="major", color="#ffffff", linewidth=0.6, alpha=0.7)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#d4d7dc", markersize=8, label="road"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#30343b", markersize=8, label="wall"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#8a6f4d", markersize=8, label="debris"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#e53935", markersize=8, label="fire"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd54f", markeredgecolor="#111827", markersize=11, label="survivor"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#43a047", markersize=8, label="base"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=6, frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)


def passable_for_ugv(terrain_value: int) -> bool:
    return terrain_value not in {WALL, DEBRIS}


def passable_for_uav(terrain_value: int) -> bool:
    return terrain_value != WALL


def _add_roads(terrain: np.ndarray, *, base_pos: GridPos, spacing: int) -> None:
    height, width = terrain.shape
    rows = sorted(set([2, base_pos[0], height - 3] + list(range(spacing, height - 1, spacing))))
    cols = sorted(set([2, base_pos[1], width - 3] + list(range(spacing, width - 1, spacing))))
    for row in rows:
        if 0 < row < height - 1:
            terrain[row, 1:-1] = ROAD
    for col in cols:
        if 0 < col < width - 1:
            terrain[1:-1, col] = ROAD


def _add_sparse_blocks(
    terrain: np.ndarray,
    *,
    rng: np.random.Generator,
    spacing: int,
    wall_block_prob: float,
) -> None:
    height, width = terrain.shape
    for top in range(3, height - 5, spacing):
        for left in range(3, width - 5, spacing):
            if rng.random() > wall_block_prob:
                continue
            block_h = int(rng.integers(1, min(4, height - top - 1) + 1))
            block_w = int(rng.integers(1, min(4, width - left - 1) + 1))
            block = terrain[top : top + block_h, left : left + block_w]
            block[block == EMPTY] = WALL


def _add_sparse_debris(
    terrain: np.ndarray,
    *,
    rng: np.random.Generator,
    ratio: float,
    base_pos: GridPos,
) -> None:
    candidates = [
        tuple(cell)
        for cell in np.argwhere(terrain == EMPTY)
        if _manhattan(tuple(cell), base_pos) > 5
    ]
    rng.shuffle(candidates)
    target = max(1, int(len(candidates) * ratio))
    added = 0
    for pos in candidates:
        if added >= target:
            break
        if _neighbor_count(terrain, pos, DEBRIS) >= 2:
            continue
        terrain[pos] = DEBRIS
        if _component_size(terrain, pos, DEBRIS, limit=9) > 8:
            terrain[pos] = EMPTY
            continue
        added += 1


def _clear_base_area(terrain: np.ndarray, base_pos: GridPos) -> None:
    height, width = terrain.shape
    br, bc = base_pos
    for row in range(max(1, br - 1), min(height - 1, br + 2)):
        for col in range(max(1, bc - 1), min(width - 1, bc + 3)):
            terrain[row, col] = ROAD


def _choose_fire_positions(
    terrain: np.ndarray,
    *,
    rng: np.random.Generator,
    reachable: set[GridPos],
    base_pos: GridPos,
    count: int,
) -> list[GridPos]:
    candidates = [
        cell
        for cell in reachable
        if terrain[cell] in {EMPTY, ROAD}
        and _manhattan(cell, base_pos) > 12
        and 4 < cell[0] < terrain.shape[0] - 4
        and 4 < cell[1] < terrain.shape[1] - 4
    ]
    rng.shuffle(candidates)
    chosen: list[GridPos] = []
    for cell in candidates:
        if all(_manhattan(cell, other) >= 5 for other in chosen):
            chosen.append(cell)
        if len(chosen) >= count:
            break
    return chosen


def _choose_survivor_positions(
    terrain: np.ndarray,
    *,
    rng: np.random.Generator,
    reachable: set[GridPos],
    base_pos: GridPos,
    fire_positions: list[GridPos],
    count: int,
    min_base_distance: int,
    min_fire_distance: int,
) -> list[GridPos]:
    fire_set = set(fire_positions)
    candidates = [
        cell
        for cell in reachable
        if terrain[cell] in {EMPTY, ROAD}
        and cell not in fire_set
        and _manhattan(cell, base_pos) >= min_base_distance
        and all(_manhattan(cell, fire) >= min_fire_distance for fire in fire_positions)
    ]
    rng.shuffle(candidates)
    chosen: list[GridPos] = []
    for cell in candidates:
        if all(_manhattan(cell, other) >= 4 for other in chosen):
            chosen.append(cell)
        if len(chosen) >= count:
            break
    return chosen


def _all_reachable(
    terrain: np.ndarray,
    base_pos: GridPos,
    survivor_positions: list[GridPos],
    fire_positions: Iterable[GridPos],
) -> bool:
    blocked = {WALL, DEBRIS}
    reachable = _reachable_cells(terrain, base_pos, blocked_values=blocked, extra_blocked=set(fire_positions))
    return all(pos in reachable for pos in survivor_positions)


def _reachable_cells(
    terrain: np.ndarray,
    start: GridPos,
    *,
    blocked_values: set[int],
    extra_blocked: set[GridPos] | None = None,
) -> set[GridPos]:
    extra_blocked = extra_blocked or set()
    height, width = terrain.shape
    seen = {start}
    queue: deque[GridPos] = deque([start])
    while queue:
        row, col = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (row + dr, col + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if nxt in seen or nxt in extra_blocked or int(terrain[nxt]) in blocked_values:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def _neighbor_count(terrain: np.ndarray, pos: GridPos, value: int) -> int:
    row, col = pos
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < terrain.shape[0] and 0 <= nc < terrain.shape[1] and terrain[nr, nc] == value:
            count += 1
    return count


def _component_size(terrain: np.ndarray, start: GridPos, value: int, *, limit: int) -> int:
    seen = {start}
    queue: deque[GridPos] = deque([start])
    while queue and len(seen) <= limit:
        row, col = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (row + dr, col + dc)
            if not (0 <= nxt[0] < terrain.shape[0] and 0 <= nxt[1] < terrain.shape[1]):
                continue
            if nxt not in seen and terrain[nxt] == value:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def _manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

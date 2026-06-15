"""Map specifications for dynamic fire rescue tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

GridPos = Tuple[int, int]


@dataclass(frozen=True)
class FireMapSpec:
    name: str
    layout: Tuple[str, ...]

    @property
    def height(self) -> int:
        return len(self.layout)

    @property
    def width(self) -> int:
        return len(self.layout[0])


EASY = FireMapSpec(
    name="easy",
    layout=(
        "############",
        "#B.....X...#",
        "#.XXX..X...#",
        "#...X......#",
        "#...X.XX...#",
        "#.....X..S.#",
        "#.XXX.X....#",
        "#.....X....#",
        "#..F.......#",
        "#....XXX...#",
        "#..........#",
        "############",
    ),
)


MEDIUM = FireMapSpec(
    name="medium",
    layout=(
        "################",
        "#B.....X.......#",
        "#.XXX..X.XXX...#",
        "#...X......X...#",
        "#...X.XXX..X...#",
        "#.......X..X.S.#",
        "#.XXXXX.X..X...#",
        "#.......X......#",
        "#..F....XXXX...#",
        "#.......X......#",
        "#.XXX...X.XXX..#",
        "#...X..........#",
        "#...X.XXXXXX...#",
        "#..............#",
        "#..............#",
        "################",
    ),
)


HARD = FireMapSpec(
    name="hard",
    layout=(
        "####################",
        "#B.....X...........#",
        "#.XXXX.X.XXXXXX....#",
        "#......X......X....#",
        "#.XXXX.XXXXX..X.S..#",
        "#....X.....X..X....#",
        "#....X.XXX.X..X....#",
        "#....X.X...X.......#",
        "#.XXXX.XXX.XXXXXX..#",
        "#..................#",
        "#.XXXXXX.XXXX.XXX..#",
        "#..F...X....X...X..#",
        "#.XXXX.X.XX.X.X.X..#",
        "#....X.X....X.X.X..#",
        "#....X.XXXXXX.X.X..#",
        "#....X........X.X..#",
        "#.XXXXXXXXXX..X.X..#",
        "#..................#",
        "#..................#",
        "####################",
    ),
)


MAPS: Dict[str, FireMapSpec] = {"easy": EASY, "medium": MEDIUM, "hard": HARD}


def get_map_spec(difficulty: str) -> FireMapSpec:
    if difficulty not in MAPS:
        raise ValueError(f"Unknown difficulty {difficulty!r}; choose one of {sorted(MAPS)}")
    spec = MAPS[difficulty]
    validate_map(spec)
    return spec


def validate_map(spec: FireMapSpec) -> None:
    widths = {len(row) for row in spec.layout}
    if len(widths) != 1:
        raise ValueError(f"Map {spec.name!r} has inconsistent widths: {widths}")
    required = {"B": 0, "S": 0, "F": 0}
    for row in spec.layout:
        for char in row:
            if char in required:
                required[char] += 1
    missing = [key for key, value in required.items() if value < 1]
    if missing:
        raise ValueError(f"Map {spec.name!r} missing markers: {missing}")


def find_marker(spec: FireMapSpec, marker: str) -> GridPos:
    for row_idx, row in enumerate(spec.layout):
        col_idx = row.find(marker)
        if col_idx >= 0:
            return row_idx, col_idx
    raise ValueError(f"Marker {marker!r} not found in {spec.name!r}")


def find_all(spec: FireMapSpec, marker: str) -> set[GridPos]:
    positions: set[GridPos] = set()
    for row_idx, row in enumerate(spec.layout):
        for col_idx, char in enumerate(row):
            if char == marker:
                positions.add((row_idx, col_idx))
    return positions


def static_obstacles(spec: FireMapSpec) -> set[GridPos]:
    obstacles: set[GridPos] = set()
    for row_idx, row in enumerate(spec.layout):
        for col_idx, char in enumerate(row):
            if char in {"#", "X"}:
                obstacles.add((row_idx, col_idx))
    return obstacles


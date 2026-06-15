"""A* baseline for the multi-UGV fire rescue environment."""

from __future__ import annotations

import heapq
from typing import Optional

from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import ACTION_TO_DELTA, FireRescueMultiUGVEnv

GridPos = tuple[int, int]
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}


def astar_path(
    env: FireRescueMultiUGVEnv,
    start: GridPos,
    goal: GridPos,
    *,
    aerial: bool,
    avoid_smoke: bool = False,
) -> list[GridPos]:
    """Return a low-risk path from start to goal, or [] if no path exists."""
    if start == goal:
        return [start]
    frontier: list[tuple[float, int, GridPos]] = []
    heapq.heappush(frontier, (float(_manhattan(start, goal)), 0, start))
    came_from: dict[GridPos, Optional[GridPos]] = {start: None}
    cost_so_far: dict[GridPos, float] = {start: 0.0}
    counter = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            return _reconstruct(came_from, current)

        for action in [1, 2, 3, 4]:
            delta = ACTION_TO_DELTA[action]
            nxt = (current[0] + delta[0], current[1] + delta[1])
            if not _can_enter(env, nxt, aerial=aerial, goal=goal):
                continue
            risk_cost = 0.0
            if nxt in env.smoke_cells:
                risk_cost += 1.2 if avoid_smoke else 0.4
            if _near_fire(env, nxt):
                risk_cost += 3.0
            new_cost = cost_so_far[current] + 1.0 + risk_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                counter += 1
                priority = new_cost + _manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, counter, nxt))
                came_from[nxt] = current
    return []


class AStarMultiUGVAgent:
    """Rule-based coordinator used for solvability checks and demos."""

    def predict(self, env: FireRescueMultiUGVEnv) -> int:
        uav_action = self._uav_action(env)
        ugv_actions = self._ugv_actions(env)
        return env.encode_action([uav_action, *ugv_actions])

    def _uav_action(self, env: FireRescueMultiUGVEnv) -> int:
        targets = [
            survivor.position
            for survivor in env.survivors
            if not survivor.discovered and not survivor.delivered and not survivor.burned
        ]
        if not targets:
            # Keep the UAV close enough to act as an overwatch visual marker.
            target = env.base_pos
        else:
            target = min(targets, key=lambda pos: _manhattan(env.uav_pos, pos))
        return _next_action(env, env.uav_pos, target, aerial=True)

    def _ugv_actions(self, env: FireRescueMultiUGVEnv) -> list[int]:
        actions = [0 for _ in range(env.num_ugv)]
        reserved_survivors: set[int] = set()
        planned_targets: list[GridPos | None] = [None for _ in range(env.num_ugv)]

        for idx in range(env.num_ugv):
            carrying = env.ugv_carrying[idx]
            if carrying is not None:
                planned_targets[idx] = env.base_pos
                continue
            target_id = self._assign_survivor(env, idx, reserved_survivors)
            if target_id is None:
                planned_targets[idx] = None
            else:
                reserved_survivors.add(target_id)
                planned_targets[idx] = env.survivors[target_id].position

        for idx, target in enumerate(planned_targets):
            if target is None:
                actions[idx] = 0
                continue
            action = _next_action(env, env.ugv_positions[idx], target, aerial=False, avoid_smoke=True)
            actions[idx] = action
        return actions

    def _assign_survivor(
        self,
        env: FireRescueMultiUGVEnv,
        ugv_idx: int,
        reserved: set[int],
    ) -> int | None:
        candidates: list[tuple[int, int]] = []
        start = env.ugv_positions[ugv_idx]
        for survivor_id, survivor in enumerate(env.survivors):
            if survivor_id in reserved:
                continue
            if (
                not survivor.discovered
                or survivor.delivered
                or survivor.burned
                or survivor.picked_by is not None
            ):
                continue
            path = astar_path(env, start, survivor.position, aerial=False, avoid_smoke=True)
            if path:
                candidates.append((len(path), survivor_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]


class CoverageAStarMultiUGVAgent(AStarMultiUGVAgent):
    """A* rescue expert whose UAV searches by coverage, not hidden targets.

    The UGV assignment is inherited from ``AStarMultiUGVAgent`` and only uses
    discovered survivors.  The UAV targets coverage waypoints exposed by the
    environment, so its demonstrations are learnable from observations.
    """

    def _uav_action(self, env: FireRescueMultiUGVEnv) -> int:
        undiscovered_alive = any(
            not survivor.discovered and not survivor.delivered and not survivor.burned
            for survivor in env.survivors
        )
        if undiscovered_alive:
            target = env.exploration_target()
            if target == env.base_pos:
                target = _nearest_unexplored_cell(env) or env.base_pos
        else:
            target = env.base_pos
        return _next_action(env, env.uav_pos, target, aerial=True)


def _next_action(
    env: FireRescueMultiUGVEnv,
    start: GridPos,
    goal: GridPos,
    *,
    aerial: bool,
    avoid_smoke: bool = False,
) -> int:
    if start == goal:
        return 0
    path = astar_path(env, start, goal, aerial=aerial, avoid_smoke=avoid_smoke)
    if len(path) < 2:
        return 0
    nxt = path[1]
    delta = (nxt[0] - start[0], nxt[1] - start[1])
    return DELTA_TO_ACTION.get(delta, 0)


def _can_enter(env: FireRescueMultiUGVEnv, pos: GridPos, *, aerial: bool, goal: GridPos) -> bool:
    if aerial:
        return env.can_uav_enter(pos)
    if pos == goal:
        return env.in_bounds(pos) and int(env.terrain[pos]) != 1 and pos not in env.fire_cells
    return env.can_ugv_enter(pos)


def _near_fire(env: FireRescueMultiUGVEnv, pos: GridPos) -> bool:
    return any(_manhattan(pos, fire) == 1 for fire in env.fire_cells)


def _nearest_unexplored_cell(env: FireRescueMultiUGVEnv) -> GridPos | None:
    candidates: list[GridPos] = []
    for row in range(env.height):
        for col in range(env.width):
            pos = (row, col)
            if pos not in env.explored_cells and env.can_uav_enter(pos):
                candidates.append(pos)
    if not candidates:
        return None
    return min(candidates, key=lambda pos: _manhattan(env.uav_pos, pos))


def _reconstruct(came_from: dict[GridPos, Optional[GridPos]], current: GridPos) -> list[GridPos]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

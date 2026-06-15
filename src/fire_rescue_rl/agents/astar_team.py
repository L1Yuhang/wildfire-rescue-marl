"""A* team baseline for UAV/UGV fire rescue."""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from fire_rescue_rl.envs.fire_rescue_env import ACTION_TO_DELTA, FireRescueMAEnv

GridPos = Tuple[int, int]
DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}


def astar(env: FireRescueMAEnv, start: GridPos, goal: GridPos, aerial: bool) -> List[GridPos]:
    frontier: list[tuple[int, int, GridPos]] = []
    heapq.heappush(frontier, (0, 0, start))
    came_from: Dict[GridPos, Optional[GridPos]] = {start: None}
    cost: Dict[GridPos, int] = {start: 0}
    counter = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for action in [1, 2, 3, 4]:
            delta = ACTION_TO_DELTA[action]
            nxt = (current[0] + delta[0], current[1] + delta[1])
            if env._outside(nxt):
                continue
            if aerial:
                if nxt in env.blocked and env.layout[nxt[0]][nxt[1]] == "#":
                    continue
            elif nxt in env.blocked or nxt in env.fire_cells:
                continue
            new_cost = cost[current] + 1
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                counter += 1
                priority = new_cost + env.manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, counter, nxt))
                came_from[nxt] = current
    if goal not in came_from:
        return []
    path = []
    current: Optional[GridPos] = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


class AStarTeamAgent:
    def predict(self, env: FireRescueMAEnv) -> int:
        uav_target = env.survivor_pos if not env.survivor_discovered else env.base_pos
        ugv_target = env.ugv_target()
        uav_action = self._next_action(env, env.uav_pos, uav_target, aerial=True)
        ugv_action = self._next_action(env, env.ugv_pos, ugv_target, aerial=False)
        if not env.survivor_discovered:
            ugv_action = 0
        return uav_action * 5 + ugv_action

    def _next_action(self, env: FireRescueMAEnv, start: GridPos, goal: GridPos, aerial: bool) -> int:
        if start == goal:
            return 0
        path = astar(env, start, goal, aerial=aerial)
        if len(path) < 2:
            return 0
        nxt = path[1]
        delta = (nxt[0] - start[0], nxt[1] - start[1])
        return DELTA_TO_ACTION.get(delta, 0)


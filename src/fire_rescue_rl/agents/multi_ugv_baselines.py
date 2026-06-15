"""Simple baselines for the multi-UGV rescue environment."""

from __future__ import annotations

import numpy as np

from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import ACTION_TO_DELTA, FireRescueMultiUGVEnv

GridPos = tuple[int, int]


class RandomMultiUGVAgent:
    """Uniform random joint-action baseline."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, env: FireRescueMultiUGVEnv) -> int:
        return int(self.rng.integers(0, env.action_space.n))


class GreedyMultiUGVAgent:
    """Manhattan-distance greedy baseline without global path planning."""

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
        target = min(targets, key=lambda pos: _manhattan(env.uav_pos, pos)) if targets else env.base_pos
        return _greedy_action(env, env.uav_pos, target, aerial=True)

    def _ugv_actions(self, env: FireRescueMultiUGVEnv) -> list[int]:
        actions = [0 for _ in range(env.num_ugv)]
        reserved: set[int] = set()
        for idx in range(env.num_ugv):
            carrying = env.ugv_carrying[idx]
            if carrying is not None:
                target = env.base_pos
            else:
                target_id = self._nearest_available_survivor(env, idx, reserved)
                if target_id is None:
                    actions[idx] = 0
                    continue
                reserved.add(target_id)
                target = env.survivors[target_id].position
            actions[idx] = _greedy_action(env, env.ugv_positions[idx], target, aerial=False)
        return actions

    def _nearest_available_survivor(
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
                survivor.discovered
                and not survivor.delivered
                and not survivor.burned
                and survivor.picked_by is None
            ):
                candidates.append((_manhattan(start, survivor.position), survivor_id))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]


def _greedy_action(env: FireRescueMultiUGVEnv, start: GridPos, target: GridPos, *, aerial: bool) -> int:
    if start == target:
        return 0
    candidates: list[tuple[int, int]] = []
    current_distance = _manhattan(start, target)
    for action in [1, 2, 3, 4, 0]:
        delta = ACTION_TO_DELTA[action]
        nxt = (start[0] + delta[0], start[1] + delta[1])
        if aerial:
            valid = env.can_uav_enter(nxt)
        else:
            valid = env.can_ugv_enter(nxt)
        if not valid:
            continue
        distance = _manhattan(nxt, target)
        candidates.append((distance, action))
    if not candidates:
        return 0
    candidates.sort()
    best_distance, best_action = candidates[0]
    if best_distance > current_distance:
        return 0
    return int(best_action)


def _manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

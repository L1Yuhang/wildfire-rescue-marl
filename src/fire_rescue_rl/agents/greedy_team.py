"""Random and greedy baselines."""

from __future__ import annotations

import numpy as np

from fire_rescue_rl.envs.fire_rescue_env import ACTION_TO_DELTA, FireRescueMAEnv


class RandomTeamAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, env: FireRescueMAEnv) -> int:
        return int(self.rng.integers(0, env.action_space.n))


class GreedyTeamAgent:
    def predict(self, env: FireRescueMAEnv) -> int:
        uav_action = self._best_action(env, env.uav_pos, env.survivor_pos, aerial=True)
        ugv_action = 0
        if env.survivor_discovered:
            ugv_action = self._best_action(env, env.ugv_pos, env.ugv_target(), aerial=False)
        return uav_action * 5 + ugv_action

    def _best_action(self, env: FireRescueMAEnv, pos, target, aerial: bool) -> int:
        best_action = 0
        best_dist = env.manhattan(pos, target)
        for action, delta in ACTION_TO_DELTA.items():
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if env._outside(nxt):
                continue
            if aerial and nxt in env.blocked and env.layout[nxt[0]][nxt[1]] == "#":
                continue
            if not aerial and (nxt in env.blocked or nxt in env.fire_cells):
                continue
            dist = env.manhattan(nxt, target)
            if dist < best_dist:
                best_dist = dist
                best_action = action
        return best_action


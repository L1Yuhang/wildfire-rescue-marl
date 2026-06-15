"""PPO-friendly wrappers for hierarchical multi-agent rescue experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, Wrapper

from fire_rescue_rl.agents.astar_multi_ugv import CoverageAStarMultiUGVAgent
from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import ACTION_TO_DELTA, FireRescueMultiUGVEnv


class RescuePPOWrapper(Wrapper):
    """Base wrapper that keeps FireRescueMultiUGVEnv metrics intact."""

    def __init__(self, env: FireRescueMultiUGVEnv, safe_projection: bool = True) -> None:
        super().__init__(env)
        self.env: FireRescueMultiUGVEnv
        self.safe_projection = bool(safe_projection)
        self.expert = CoverageAStarMultiUGVAgent()

    def _safe_action(self, agent_idx: int, action: int) -> int:
        action = int(action)
        if not self.safe_projection or action == 0:
            return action
        if agent_idx == 0:
            pos = self.env.uav_pos
            can_enter = self.env.can_uav_enter
        else:
            pos = self.env.ugv_positions[agent_idx - 1]
            can_enter = self.env.can_ugv_enter
        delta = ACTION_TO_DELTA[action]
        nxt = (pos[0] + delta[0], pos[1] + delta[1])
        return action if can_enter(nxt) else 0

    def _encode_safe_joint(self, actions: list[int]) -> int:
        projected = [self._safe_action(idx, action) for idx, action in enumerate(actions)]
        return self.env.encode_action(projected)

    def _valid_action_mask(self, agent_idx: int) -> np.ndarray:
        mask = np.zeros(5, dtype=bool)
        mask[0] = True
        if agent_idx == 0:
            pos = self.env.uav_pos
            can_enter = self.env.can_uav_enter
        else:
            pos = self.env.ugv_positions[agent_idx - 1]
            can_enter = self.env.can_ugv_enter
        for action in [1, 2, 3, 4]:
            delta = ACTION_TO_DELTA[action]
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            mask[action] = bool(can_enter(nxt))
        return mask


class PPOUAVHybridEnv(RescuePPOWrapper):
    """PPO controls the UAV; Coverage A* controls both UGVs."""

    def __init__(self, env: FireRescueMultiUGVEnv, safe_projection: bool = True) -> None:
        super().__init__(env, safe_projection=safe_projection)
        self.action_space = spaces.Discrete(5)
        self.observation_space = env.observation_space

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        uav_action = self._safe_action(0, int(action))
        ugv_actions = self.expert._ugv_actions(self.env)
        return self.env.step(self.env.encode_action([uav_action, *ugv_actions]))

    def action_masks(self) -> np.ndarray:
        return self._valid_action_mask(0)


class PPOUGVHybridEnv(RescuePPOWrapper):
    """Coverage A* controls the UAV; PPO controls both UGVs."""

    def __init__(self, env: FireRescueMultiUGVEnv, safe_projection: bool = True) -> None:
        super().__init__(env, safe_projection=safe_projection)
        self.action_space = spaces.MultiDiscrete([5 for _ in range(env.num_ugv)])
        self.observation_space = env.observation_space

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        if len(action_array) != self.env.num_ugv:
            raise ValueError(f"Expected {self.env.num_ugv} UGV actions, got {len(action_array)}")
        uav_action = self.expert._uav_action(self.env)
        ugv_actions = [
            self._safe_action(idx + 1, int(ugv_action))
            for idx, ugv_action in enumerate(action_array.tolist())
        ]
        return self.env.step(self.env.encode_action([uav_action, *ugv_actions]))

    def action_masks(self) -> np.ndarray:
        return np.concatenate(
            [self._valid_action_mask(idx + 1) for idx in range(self.env.num_ugv)]
        )


class FullMultiDiscretePPOEnv(RescuePPOWrapper):
    """PPO controls UAV, UGV0, and UGV1 with a factorized action space."""

    def __init__(self, env: FireRescueMultiUGVEnv, safe_projection: bool = True) -> None:
        super().__init__(env, safe_projection=safe_projection)
        self.action_space = spaces.MultiDiscrete([5 for _ in range(1 + env.num_ugv)])
        self.observation_space = env.observation_space

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        expected = 1 + self.env.num_ugv
        if len(action_array) != expected:
            raise ValueError(f"Expected {expected} actions, got {len(action_array)}")
        return self.env.step(self._encode_safe_joint(action_array.tolist()))

    def action_masks(self) -> np.ndarray:
        return np.concatenate(
            [self._valid_action_mask(idx) for idx in range(1 + self.env.num_ugv)]
        )


def make_ppo_env(
    env_config: dict[str, Any],
    mode: str,
    *,
    render_mode: str | None = None,
    safe_projection: bool = True,
):
    env = FireRescueMultiUGVEnv.from_config(env_config, render_mode=render_mode)
    if mode == "uav":
        return PPOUAVHybridEnv(env, safe_projection=safe_projection)
    if mode == "ugv":
        return PPOUGVHybridEnv(env, safe_projection=safe_projection)
    if mode == "full":
        return FullMultiDiscretePPOEnv(env, safe_projection=safe_projection)
    raise ValueError(f"Unknown PPO mode: {mode}")

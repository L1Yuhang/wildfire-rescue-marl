"""Centralized joint-action Gymnasium environment for UAV/UGV fire rescue."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required in the RLearning environment.") from exc

from fire_rescue_rl.envs.fire_dynamics import smoke_from_fire, spread_fire
from fire_rescue_rl.envs.maps import FireMapSpec, find_all, find_marker, get_map_spec, static_obstacles
from fire_rescue_rl.envs.renderer import FireRescueRenderer
from fire_rescue_rl.envs.reward import RewardWeights

GridPos = Tuple[int, int]

ACTION_TO_DELTA: Dict[int, GridPos] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

ACTION_NAMES = {0: "wait", 1: "up", 2: "down", 3: "left", 4: "right"}


class FireRescueMAEnv(gym.Env):
    """Two-entity UAV/UGV rescue task trained as a centralized DQN problem."""

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 6}

    def __init__(
        self,
        difficulty: str = "easy",
        render_mode: str | None = None,
        view_size: int = 7,
        max_steps: int = 120,
        fire_spread_interval: int = 999,
        fire_spread_prob: float = 0.0,
        smoke_radius: int = 1,
        random_survivor: bool = False,
        random_fire: bool = False,
        max_fire_cells: int = 12,
        reward: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.map_spec: FireMapSpec = get_map_spec(difficulty)
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.view_size = int(view_size)
        if self.view_size % 2 != 1:
            raise ValueError("view_size must be odd")
        self.max_steps = int(max_steps)
        self.fire_spread_interval = int(fire_spread_interval)
        self.fire_spread_prob = float(fire_spread_prob)
        self.smoke_radius = int(smoke_radius)
        self.random_survivor = bool(random_survivor)
        self.random_fire = bool(random_fire)
        self.max_fire_cells = int(max_fire_cells)
        self.reward_weights = RewardWeights.from_dict(reward)

        self.layout = self.map_spec.layout
        self.height = self.map_spec.height
        self.width = self.map_spec.width
        self.base_pos = find_marker(self.map_spec, "B")
        self.default_survivor_pos = find_marker(self.map_spec, "S")
        self.default_fire_cells = find_all(self.map_spec, "F")
        self.blocked = static_obstacles(self.map_spec)
        self.water = find_all(self.map_spec, "W")

        self.action_space = spaces.Discrete(25)
        obs_dim = 23 + 2 * self.view_size * self.view_size
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.renderer = FireRescueRenderer()
        self.rng = np.random.default_rng()

        self.uav_pos = self.base_pos
        self.ugv_pos = self.base_pos
        self.survivor_pos = self.default_survivor_pos
        self.fire_cells: set[GridPos] = set(self.default_fire_cells)
        self.smoke_cells: set[GridPos] = set()
        self.survivor_discovered = False
        self.survivor_picked = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.risk_exposure = 0.0
        self.uav_path: List[GridPos] = []
        self.ugv_path: List[GridPos] = []
        self.distance_maps: Dict[GridPos, Dict[GridPos, int]] = {}
        self.explored: set[GridPos] = set()

    @classmethod
    def from_config(cls, config: Dict[str, Any], render_mode: str | None = None) -> "FireRescueMAEnv":
        values = dict(config)
        if render_mode is not None:
            values["render_mode"] = render_mode
        return cls(**values)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.uav_pos = self.base_pos
        self.ugv_pos = self.base_pos
        self.fire_cells = self._sample_initial_fire()
        self.survivor_pos = self._sample_survivor()
        self.smoke_cells = smoke_from_fire(self.fire_cells, self.blocked, self.height, self.width, self.smoke_radius)
        self.survivor_discovered = False
        self.survivor_picked = False
        self.step_count = 0
        self.episode_reward = 0.0
        self.risk_exposure = 0.0
        self.uav_path = [self.uav_pos]
        self.ugv_path = [self.ugv_pos]
        self.explored = set()
        self._update_explored()
        self.distance_maps = {
            self.survivor_pos: self._build_distance_map(self.survivor_pos, ugv=True),
            self.base_pos: self._build_distance_map(self.base_pos, ugv=True),
        }
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action: int):
        action = int(action)
        uav_action = action // 5
        ugv_action = action % 5
        components: Dict[str, float] = {"time": self.reward_weights.time}
        if uav_action == 0 and ugv_action == 0:
            components["wait"] = self.reward_weights.wait

        old_uav_dist = self.manhattan(self.uav_pos, self.survivor_pos)
        old_ugv_target = self.ugv_target()
        old_ugv_dist = self.navigation_distance(self.ugv_pos, old_ugv_target)

        invalid_uav = False
        invalid_ugv = False
        self.uav_pos, invalid_uav = self._move_uav(self.uav_pos, uav_action)
        self.ugv_pos, invalid_ugv = self._move_ugv(self.ugv_pos, ugv_action)
        if invalid_uav:
            components["invalid_uav"] = self.reward_weights.invalid_uav
        if invalid_ugv:
            components["invalid_ugv"] = self.reward_weights.invalid_ugv

        self._update_explored()
        if not self.survivor_discovered and self.manhattan(self.uav_pos, self.survivor_pos) <= self.view_size // 2:
            self.survivor_discovered = True
            components["discover"] = self.reward_weights.discover

        if self.survivor_discovered and not self.survivor_picked and self.ugv_pos == self.survivor_pos:
            self.survivor_picked = True
            components["pickup"] = self.reward_weights.pickup

        success = False
        if self.survivor_picked and self.ugv_pos == self.base_pos:
            success = True
            components["success"] = self.reward_weights.success

        new_uav_dist = self.manhattan(self.uav_pos, self.survivor_pos)
        if not self.survivor_discovered:
            delta = old_uav_dist - new_uav_dist
            if delta > 0:
                components["uav_progress"] = self.reward_weights.uav_progress * delta
            elif delta < 0:
                components["uav_regress"] = self.reward_weights.regress * abs(delta)

        new_ugv_target = self.ugv_target()
        new_ugv_dist = self.navigation_distance(self.ugv_pos, old_ugv_target)
        if old_ugv_target == new_ugv_target:
            delta = old_ugv_dist - new_ugv_dist
            if delta > 0:
                phase_factor = 2.0 if self.survivor_picked else 1.0
                components["ugv_progress"] = self.reward_weights.ugv_progress * phase_factor * delta
            elif delta < 0:
                phase_factor = 1.4 if self.survivor_picked else 1.0
                components["ugv_regress"] = self.reward_weights.regress * phase_factor * abs(delta)

        risk = self._risk_penalty()
        if risk:
            components.update(risk)

        self.step_count += 1
        if self._should_spread_fire():
            self.fire_cells = spread_fire(
                self.fire_cells,
                self.blocked,
                self.water,
                self.height,
                self.width,
                self.fire_spread_prob,
                self.max_fire_cells,
                self.rng,
            )
            self.smoke_cells = smoke_from_fire(self.fire_cells, self.blocked, self.height, self.width, self.smoke_radius)

        terminated = False
        truncated = False
        reason = ""
        if self.ugv_pos in self.fire_cells:
            components["fire_collision"] = self.reward_weights.fire_collision
            terminated = True
            reason = "ugv_fire_collision"
        elif self.survivor_pos in self.fire_cells and not self.survivor_picked:
            components["survivor_burned"] = self.reward_weights.survivor_burned
            terminated = True
            reason = "survivor_burned"
        elif success:
            terminated = True
            reason = "success"
        elif self.step_count >= self.max_steps:
            truncated = True
            reason = "timeout"

        if not self.survivor_discovered and self.step_count > self.max_steps * 0.45:
            components["coordination"] = self.reward_weights.coordination

        reward = float(sum(components.values()))
        self.episode_reward += reward
        self.uav_path.append(self.uav_pos)
        self.ugv_path.append(self.ugv_pos)

        obs = self._get_obs()
        info = self._get_info()
        info.update(
            {
                "reward_components": components,
                "uav_action": ACTION_NAMES[uav_action],
                "ugv_action": ACTION_NAMES[ugv_action],
                "success": success,
                "discover": "discover" in components,
                "pickup": "pickup" in components,
                "invalid_uav": invalid_uav,
                "invalid_ugv": invalid_ugv,
                "terminated_reason": reason,
            }
        )
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        return self.renderer.render(self.render_state())

    def close(self) -> None:
        return None

    def ugv_target(self) -> GridPos:
        if self.survivor_picked:
            return self.base_pos
        if self.survivor_discovered:
            return self.survivor_pos
        return self.base_pos

    def render_state(self) -> Dict[str, Any]:
        return {
            "layout": self.layout,
            "uav_pos": self.uav_pos,
            "ugv_pos": self.ugv_pos,
            "base_pos": self.base_pos,
            "survivor_pos": self.survivor_pos,
            "survivor_discovered": self.survivor_discovered,
            "survivor_picked": self.survivor_picked,
            "fire_cells": set(self.fire_cells),
            "smoke_cells": set(self.smoke_cells),
            "uav_path": list(self.uav_path),
            "ugv_path": list(self.ugv_path),
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "risk_exposure": self.risk_exposure,
            "stage": self.stage_name(),
        }

    def stage_name(self) -> str:
        if not self.survivor_discovered:
            return "search survivor"
        if not self.survivor_picked:
            return "ground rescue"
        return "return to base"

    def manhattan(self, a: GridPos, b: GridPos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def navigation_distance(self, a: GridPos, b: GridPos) -> int:
        if b not in self.distance_maps:
            self.distance_maps[b] = self._build_distance_map(b, ugv=True)
        return self.distance_maps[b].get(a, self.height * self.width)

    def _move_uav(self, pos: GridPos, action: int) -> tuple[GridPos, bool]:
        delta = ACTION_TO_DELTA[action]
        nxt = (pos[0] + delta[0], pos[1] + delta[1])
        if self._outside(nxt) or nxt in self.blocked and self.layout[nxt[0]][nxt[1]] == "#":
            return pos, action != 0
        return nxt, False

    def _move_ugv(self, pos: GridPos, action: int) -> tuple[GridPos, bool]:
        delta = ACTION_TO_DELTA[action]
        nxt = (pos[0] + delta[0], pos[1] + delta[1])
        if self._outside(nxt) or nxt in self.blocked or nxt in self.fire_cells:
            return pos, action != 0
        return nxt, False

    def _outside(self, pos: GridPos) -> bool:
        return pos[0] < 0 or pos[0] >= self.height or pos[1] < 0 or pos[1] >= self.width

    def _risk_penalty(self) -> Dict[str, float]:
        components: Dict[str, float] = {}
        risk = 0.0
        for prefix, pos in [("uav", self.uav_pos), ("ugv", self.ugv_pos)]:
            if pos in self.smoke_cells:
                components[f"{prefix}_smoke"] = self.reward_weights.smoke_risk
                risk += abs(self.reward_weights.smoke_risk)
            if self._near_fire(pos):
                components[f"{prefix}_near_fire"] = self.reward_weights.near_fire_risk
                risk += abs(self.reward_weights.near_fire_risk)
        self.risk_exposure += risk
        return components

    def _near_fire(self, pos: GridPos) -> bool:
        return any(self.manhattan(pos, fire) == 1 for fire in self.fire_cells)

    def _should_spread_fire(self) -> bool:
        return self.fire_spread_interval > 0 and self.step_count > 0 and self.step_count % self.fire_spread_interval == 0

    def _sample_initial_fire(self) -> set[GridPos]:
        if not self.random_fire:
            return set(self.default_fire_cells)
        candidates = [pos for pos in self._free_cells() if self.manhattan(pos, self.base_pos) > 5 and pos != self.default_survivor_pos]
        chosen = candidates[int(self.rng.integers(0, len(candidates)))]
        return {chosen}

    def _sample_survivor(self) -> GridPos:
        if not self.random_survivor:
            return self.default_survivor_pos
        candidates = [pos for pos in self._free_cells() if self.manhattan(pos, self.base_pos) > 8 and pos not in self.fire_cells]
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def _free_cells(self) -> list[GridPos]:
        cells = []
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col)
                if pos not in self.blocked and pos not in self.water:
                    cells.append(pos)
        return cells

    def _update_explored(self) -> None:
        radius = self.view_size // 2
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                pos = (self.uav_pos[0] + dr, self.uav_pos[1] + dc)
                if not self._outside(pos):
                    self.explored.add(pos)

    def _build_distance_map(self, target: GridPos, ugv: bool) -> Dict[GridPos, int]:
        distances = {target: 0}
        queue: deque[GridPos] = deque([target])
        while queue:
            current = queue.popleft()
            for action in [1, 2, 3, 4]:
                delta = ACTION_TO_DELTA[action]
                nxt = (current[0] + delta[0], current[1] + delta[1])
                if self._outside(nxt) or nxt in distances:
                    continue
                if ugv and (nxt in self.blocked or nxt in self.fire_cells):
                    continue
                if not ugv and nxt in self.blocked and self.layout[nxt[0]][nxt[1]] == "#":
                    continue
                distances[nxt] = distances[current] + 1
                queue.append(nxt)
        return distances

    def _get_obs(self) -> np.ndarray:
        max_row = max(1, self.height - 1)
        max_col = max(1, self.width - 1)
        max_dist = max(1, self.height * self.width)
        features: List[float] = []
        features.extend([self.uav_pos[0] / max_row, self.uav_pos[1] / max_col])
        features.extend([self.ugv_pos[0] / max_row, self.ugv_pos[1] / max_col])
        features.extend([self.base_pos[0] / max_row, self.base_pos[1] / max_col])
        if self.survivor_discovered:
            features.extend([self.survivor_pos[0] / max_row, self.survivor_pos[1] / max_col])
        else:
            features.extend([0.0, 0.0])
        ugv_target = self.ugv_target()
        features.extend([ugv_target[0] / max_row, ugv_target[1] / max_col])
        features.append(1.0 if self.survivor_discovered else 0.0)
        features.append(1.0 if self.survivor_picked else 0.0)
        features.extend(
            [
                1.0 if not self.survivor_discovered else 0.0,
                1.0 if self.survivor_discovered and not self.survivor_picked else 0.0,
                1.0 if self.survivor_picked else 0.0,
            ]
        )
        features.append(self.step_count / max(1, self.max_steps))
        features.append(min(1.0, len(self.fire_cells) / max(1, self.max_fire_cells)))
        features.append(min(1.0, self.risk_exposure / 50.0))
        features.append(self.manhattan(self.uav_pos, self.survivor_pos) / max_dist)
        features.append(min(1.0, self.navigation_distance(self.ugv_pos, self.ugv_target()) / max_dist))
        features.append(len(self.explored) / max(1, self.height * self.width))
        features.extend([1.0 if self.ugv_pos in self.smoke_cells else 0.0, 1.0 if self.uav_pos in self.smoke_cells else 0.0])

        features.extend(self._local_view(self.uav_pos, aerial=True))
        features.extend(self._local_view(self.ugv_pos, aerial=False))
        return np.asarray(features, dtype=np.float32)

    def _local_view(self, center: GridPos, aerial: bool) -> List[float]:
        radius = self.view_size // 2
        values: List[float] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                pos = (center[0] + dr, center[1] + dc)
                values.append(self._cell_code(pos, aerial) / 8.0)
        return values

    def _cell_code(self, pos: GridPos, aerial: bool) -> int:
        if self._outside(pos):
            return 8
        if pos in self.fire_cells:
            return 6
        if pos in self.smoke_cells:
            return 5
        if pos == self.base_pos:
            return 2
        if pos == self.survivor_pos and self.survivor_discovered:
            return 3
        if pos in self.blocked:
            if aerial and self.layout[pos[0]][pos[1]] != "#":
                return 1
            return 7
        if pos in self.water:
            return 4
        return 0

    def _get_info(self) -> Dict[str, Any]:
        return {
            "uav_pos": self.uav_pos,
            "ugv_pos": self.ugv_pos,
            "base_pos": self.base_pos,
            "survivor_pos": self.survivor_pos,
            "survivor_discovered": self.survivor_discovered,
            "survivor_picked": self.survivor_picked,
            "fire_cells": set(self.fire_cells),
            "smoke_cells": set(self.smoke_cells),
            "risk_exposure": self.risk_exposure,
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "stage": self.stage_name(),
            "uav_path": list(self.uav_path),
            "ugv_path": list(self.ugv_path),
        }

    def _render_ansi(self) -> str:
        chars = [list(row) for row in self.layout]
        for row, col in self.smoke_cells:
            if chars[row][col] == ".":
                chars[row][col] = "~"
        for row, col in self.fire_cells:
            chars[row][col] = "F"
        sr, sc = self.survivor_pos
        chars[sr][sc] = "S" if self.survivor_discovered else "?"
        ur, uc = self.uav_pos
        gr, gc = self.ugv_pos
        chars[ur][uc] = "A"
        chars[gr][gc] = "G"
        return "\n".join("".join(row) for row in chars)

"""Centralized multi-agent rescue environment with one UAV and two UGVs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required in the RLearning environment.") from exc

from fire_rescue_rl.envs.generated_maps import (
    BURNED,
    DEBRIS,
    EMPTY,
    ROAD,
    WALL,
    GeneratedRescueMap,
    generate_valid_rescue_map,
    passable_for_uav,
    passable_for_ugv,
)

GridPos = tuple[int, int]

ACTION_TO_DELTA: dict[int, GridPos] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}
ACTION_NAMES = {0: "wait", 1: "up", 2: "down", 3: "left", 4: "right"}


@dataclass
class SurvivorState:
    position: GridPos
    discovered: bool = False
    picked_by: int | None = None
    delivered: bool = False
    burned: bool = False


class FireRescueMultiUGVEnv(gym.Env):
    """Gymnasium env for dynamic fire rescue with cooperative ground robots.

    The environment is intentionally centralized so it can be trained with
    ordinary single-agent algorithms while still modeling multi-agent
    coordination.  One UAV searches for survivors, and two UGVs rescue and
    deliver discovered survivors to the base.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 6}

    def __init__(
        self,
        width: int = 28,
        height: int = 28,
        render_mode: str | None = None,
        view_size: int = 7,
        max_steps: int = 260,
        num_ugv: int = 2,
        num_survivors: int = 3,
        max_survivors: int | None = None,
        num_fire_sources: int = 3,
        fire_spread_interval: int = 7,
        fire_spread_prob: float = 0.055,
        smoke_radius: int = 2,
        max_fire_cells: int = 45,
        debris_burn_steps: int = 3,
        map_seed: int = 42,
        random_map: bool = True,
        road_spacing: int = 6,
        wall_block_prob: float = 0.35,
        debris_ratio: float = 0.06,
        uav_view_radius: int | None = None,
        ugv_view_radius: int = 2,
        include_exploration_features: bool = False,
        include_guidance_features: bool = False,
        exploration_waypoint_spacing: int = 6,
        reward: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.render_mode = render_mode
        self.view_size = int(view_size)
        if self.view_size % 2 != 1:
            raise ValueError("view_size must be odd")
        self.max_steps = int(max_steps)
        self.num_ugv = int(num_ugv)
        if self.num_ugv < 1:
            raise ValueError("num_ugv must be at least 1")
        self.num_survivors = int(num_survivors)
        self.max_survivors = int(max_survivors or num_survivors)
        if self.num_survivors > self.max_survivors:
            raise ValueError("num_survivors cannot exceed max_survivors")
        self.num_fire_sources = int(num_fire_sources)
        self.fire_spread_interval = int(fire_spread_interval)
        self.fire_spread_prob = float(fire_spread_prob)
        self.smoke_radius = int(smoke_radius)
        self.max_fire_cells = int(max_fire_cells)
        self.debris_burn_steps = int(debris_burn_steps)
        self.map_seed = int(map_seed)
        self.random_map = bool(random_map)
        self.road_spacing = int(road_spacing)
        self.wall_block_prob = float(wall_block_prob)
        self.debris_ratio = float(debris_ratio)
        self.uav_view_radius = int(uav_view_radius if uav_view_radius is not None else self.view_size // 2)
        self.ugv_view_radius = int(ugv_view_radius)
        self.include_exploration_features = bool(include_exploration_features)
        self.include_guidance_features = bool(include_guidance_features)
        self.exploration_waypoint_spacing = int(exploration_waypoint_spacing)
        self.reward_weights = self._default_reward()
        if reward:
            self.reward_weights.update({key: float(value) for key, value in reward.items()})

        self.action_space = spaces.Discrete(5 ** (1 + self.num_ugv))
        base_dim = 2 + 2 * self.num_ugv + 2 + 6 * self.max_survivors + 2 * self.num_ugv + 8
        if self.include_exploration_features:
            base_dim += 3
        if self.include_guidance_features:
            base_dim += 2 + 3 * self.num_ugv
        local_dim = (1 + self.num_ugv) * self.view_size * self.view_size
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(base_dim + local_dim,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(self.map_seed)
        self.rescue_map: GeneratedRescueMap | None = None
        self.terrain = np.zeros((self.height, self.width), dtype=np.int16)
        self.base_pos: GridPos = (self.height - 3, 2)
        self.uav_pos: GridPos = self.base_pos
        self.ugv_positions: list[GridPos] = [self.base_pos for _ in range(self.num_ugv)]
        self.survivors: list[SurvivorState] = []
        self.fire_cells: set[GridPos] = set()
        self.smoke_cells: set[GridPos] = set()
        self.debris_heat: dict[GridPos, int] = {}
        self.burned_debris: set[GridPos] = set()
        self.ugv_carrying: list[int | None] = [None for _ in range(self.num_ugv)]
        self.step_count = 0
        self.episode_reward = 0.0
        self.risk_exposure = 0.0
        self.uav_path: list[GridPos] = []
        self.ugv_paths: list[list[GridPos]] = []
        self.last_components: dict[str, float] = {}
        self.reset_count = 0
        self.explored_cells: set[GridPos] = set()

    @classmethod
    def from_config(cls, config: dict[str, Any], render_mode: str | None = None) -> "FireRescueMultiUGVEnv":
        values = dict(config)
        values.pop("experiment_name", None)
        if render_mode is not None:
            values["render_mode"] = render_mode
        return cls(**values)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if not self.random_map:
            episode_seed = self.map_seed
        elif seed is not None:
            episode_seed = self.map_seed + int(seed) * 1009
            self.reset_count = 0
        else:
            self.reset_count += 1
            episode_seed = self.map_seed + self.reset_count * 1009
        self.rng = np.random.default_rng(episode_seed)
        self.rescue_map = generate_valid_rescue_map(
            width=self.width,
            height=self.height,
            seed=episode_seed,
            num_survivors=self.num_survivors,
            num_fire_sources=self.num_fire_sources,
            road_spacing=self.road_spacing,
            wall_block_prob=self.wall_block_prob,
            debris_ratio=self.debris_ratio,
        )
        self.terrain = self.rescue_map.terrain.copy()
        self.base_pos = self.rescue_map.base_pos
        self.uav_pos = self.base_pos
        self.ugv_positions = self._initial_ugv_positions()
        self.survivors = [SurvivorState(position=pos) for pos in self.rescue_map.survivor_positions]
        self.fire_cells = set(self.rescue_map.fire_positions)
        self.smoke_cells = self._compute_smoke()
        self.debris_heat = {}
        self.burned_debris = set()
        self.ugv_carrying = [None for _ in range(self.num_ugv)]
        self.step_count = 0
        self.episode_reward = 0.0
        self.risk_exposure = 0.0
        self.uav_path = [self.uav_pos]
        self.ugv_paths = [[pos] for pos in self.ugv_positions]
        self.last_components = {}
        self.explored_cells = set()
        self._update_explored_cells()
        self._discover_visible_survivors()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        actions = self.decode_action(action)
        components: dict[str, float] = {"time": self.reward_weights["time"]}

        old_uav_target = self._uav_progress_target()
        old_uav_distance = self.manhattan(self.uav_pos, old_uav_target) if old_uav_target else 0
        old_explored_count = len(self.explored_cells)
        old_ugv_targets = [self.ugv_target(idx) for idx in range(self.num_ugv)]
        old_ugv_distances = [
            self.manhattan(self.ugv_positions[idx], target) if target else 0
            for idx, target in enumerate(old_ugv_targets)
        ]

        self.uav_pos, invalid_uav = self._move_uav(self.uav_pos, actions[0])
        if invalid_uav:
            components["invalid_uav"] = self.reward_weights["invalid_uav"]
        self._update_explored_cells()
        if self.include_exploration_features:
            new_explored = len(self.explored_cells) - old_explored_count
            if new_explored > 0:
                components["explore"] = self.reward_weights["explore"] * float(new_explored)

        invalid_ugv_count = 0
        for idx, ugv_action in enumerate(actions[1:]):
            new_pos, invalid = self._move_ugv(self.ugv_positions[idx], ugv_action)
            self.ugv_positions[idx] = new_pos
            if invalid:
                invalid_ugv_count += 1
        if invalid_ugv_count:
            components["invalid_ugv"] = self.reward_weights["invalid_ugv"] * invalid_ugv_count

        discoveries = self._discover_visible_survivors()
        if discoveries:
            components["discover"] = self.reward_weights["discover"] * discoveries

        pickups = self._pickup_survivors()
        if pickups:
            components["pickup"] = self.reward_weights["pickup"] * pickups

        deliveries = self._deliver_survivors()
        if deliveries:
            components["deliver"] = self.reward_weights["deliver"] * deliveries

        if old_uav_target is not None:
            new_uav_distance = self.manhattan(self.uav_pos, old_uav_target)
            components["uav_progress"] = self.reward_weights["uav_progress"] * float(old_uav_distance - new_uav_distance)

        for idx, old_target in enumerate(old_ugv_targets):
            if old_target is None:
                continue
            new_distance = self.manhattan(self.ugv_positions[idx], old_target)
            progress = float(old_ugv_distances[idx] - new_distance)
            components[f"ugv{idx}_progress"] = self.reward_weights["ugv_progress"] * progress

        risk_penalty = self._risk_penalty()
        if risk_penalty:
            components["risk"] = risk_penalty

        self.step_count += 1
        if self._should_spread_fire():
            burned_now = self._spread_fire()
            if burned_now:
                components["debris_burned"] = self.reward_weights["debris_burned"] * burned_now
            self.smoke_cells = self._compute_smoke()

        burned_survivors = self._burn_exposed_survivors()
        if burned_survivors:
            components["survivor_burned"] = self.reward_weights["survivor_burned"] * burned_survivors

        ugv_fire_hits = sum(1 for pos in self.ugv_positions if pos in self.fire_cells)
        if ugv_fire_hits:
            components["fire_collision"] = self.reward_weights["fire_collision"] * ugv_fire_hits

        success = self._all_survivors_delivered()
        terminated = False
        truncated = False
        reason = ""
        if ugv_fire_hits:
            terminated = True
            reason = "ugv_fire_collision"
        elif self._all_survivors_terminal():
            terminated = True
            reason = "success" if success else "survivor_burned"
            if success:
                components["success"] = self.reward_weights["success"]
        elif self.step_count >= self.max_steps:
            truncated = True
            reason = "timeout"

        reward = float(sum(components.values()))
        self.episode_reward += reward
        self.uav_path.append(self.uav_pos)
        for idx, pos in enumerate(self.ugv_positions):
            self.ugv_paths[idx].append(pos)
        self.last_components = components

        info = self._get_info()
        info.update(
            {
                "uav_action": ACTION_NAMES[actions[0]],
                "ugv_actions": [ACTION_NAMES[a] for a in actions[1:]],
                "invalid_uav": invalid_uav,
                "invalid_ugv_count": invalid_ugv_count,
                "success": bool(success),
                "terminated_reason": reason,
                "reward_components": components,
            }
        )
        return self._get_obs(), reward, bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        return self._render_rgb()

    def close(self) -> None:
        return None

    def decode_action(self, action: int) -> list[int]:
        actions = [0 for _ in range(1 + self.num_ugv)]
        value = int(action)
        for idx in range(len(actions) - 1, -1, -1):
            actions[idx] = value % 5
            value //= 5
        return actions

    def encode_action(self, actions: list[int] | tuple[int, ...]) -> int:
        value = 0
        for action in actions:
            value = value * 5 + int(action)
        return int(value)

    def ugv_target(self, idx: int) -> GridPos | None:
        carrying = self.ugv_carrying[idx]
        if carrying is not None:
            return self.base_pos
        candidates = [
            survivor.position
            for survivor in self.survivors
            if survivor.discovered
            and not survivor.delivered
            and not survivor.burned
            and survivor.picked_by is None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pos: self.manhattan(self.ugv_positions[idx], pos))

    def render_state(self) -> dict[str, Any]:
        return self._get_info()

    def manhattan(self, a: GridPos, b: GridPos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def in_bounds(self, pos: GridPos) -> bool:
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def can_uav_enter(self, pos: GridPos) -> bool:
        return self.in_bounds(pos) and passable_for_uav(int(self.terrain[pos]))

    def can_ugv_enter(self, pos: GridPos) -> bool:
        return self.in_bounds(pos) and passable_for_ugv(int(self.terrain[pos])) and pos not in self.fire_cells

    def _default_reward(self) -> dict[str, float]:
        return {
            "time": -0.05,
            "discover": 14.0,
            "pickup": 26.0,
            "deliver": 62.0,
            "success": 120.0,
            "explore": 0.02,
            "uav_progress": 0.18,
            "ugv_progress": 0.24,
            "invalid_uav": -2.0,
            "invalid_ugv": -4.0,
            "risk_smoke": -0.35,
            "risk_near_fire": -1.0,
            "fire_collision": -90.0,
            "survivor_burned": -90.0,
            "debris_burned": 1.0,
        }

    def _initial_ugv_positions(self) -> list[GridPos]:
        offsets = [(0, 0), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]
        positions: list[GridPos] = []
        for offset in offsets:
            if len(positions) >= self.num_ugv:
                break
            pos = (self.base_pos[0] + offset[0], self.base_pos[1] + offset[1])
            if self.can_ugv_enter(pos):
                positions.append(pos)
        while len(positions) < self.num_ugv:
            positions.append(self.base_pos)
        return positions

    def _move_uav(self, pos: GridPos, action: int) -> tuple[GridPos, bool]:
        delta = ACTION_TO_DELTA.get(int(action), (0, 0))
        nxt = (pos[0] + delta[0], pos[1] + delta[1])
        if not self.can_uav_enter(nxt):
            return pos, int(action) != 0
        return nxt, False

    def _move_ugv(self, pos: GridPos, action: int) -> tuple[GridPos, bool]:
        delta = ACTION_TO_DELTA.get(int(action), (0, 0))
        nxt = (pos[0] + delta[0], pos[1] + delta[1])
        if not self.can_ugv_enter(nxt):
            return pos, int(action) != 0
        return nxt, False

    def _discover_visible_survivors(self) -> int:
        count = 0
        observer_positions = [(self.uav_pos, self.uav_view_radius)]
        observer_positions.extend((pos, self.ugv_view_radius) for pos in self.ugv_positions)
        for survivor in self.survivors:
            if survivor.discovered or survivor.delivered or survivor.burned:
                continue
            if any(self.manhattan(pos, survivor.position) <= radius for pos, radius in observer_positions):
                survivor.discovered = True
                count += 1
        return count

    def _pickup_survivors(self) -> int:
        pickups = 0
        for idx, pos in enumerate(self.ugv_positions):
            if self.ugv_carrying[idx] is not None:
                continue
            for survivor_id, survivor in enumerate(self.survivors):
                if (
                    survivor.discovered
                    and not survivor.delivered
                    and not survivor.burned
                    and survivor.picked_by is None
                    and survivor.position == pos
                ):
                    survivor.picked_by = idx
                    self.ugv_carrying[idx] = survivor_id
                    pickups += 1
                    break
        return pickups

    def _deliver_survivors(self) -> int:
        deliveries = 0
        for idx, pos in enumerate(self.ugv_positions):
            survivor_id = self.ugv_carrying[idx]
            if survivor_id is None or pos != self.base_pos:
                continue
            survivor = self.survivors[survivor_id]
            survivor.delivered = True
            survivor.picked_by = None
            self.ugv_carrying[idx] = None
            deliveries += 1
        return deliveries

    def _nearest_undiscovered_position(self, pos: GridPos) -> GridPos | None:
        candidates = [
            survivor.position
            for survivor in self.survivors
            if not survivor.discovered and not survivor.delivered and not survivor.burned
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda target: self.manhattan(pos, target))

    def _uav_progress_target(self) -> GridPos | None:
        if not self.include_exploration_features:
            return self._nearest_undiscovered_position(self.uav_pos)
        undiscovered_alive = any(
            not survivor.discovered and not survivor.delivered and not survivor.burned
            for survivor in self.survivors
        )
        if not undiscovered_alive:
            return self.base_pos
        target = self.exploration_target()
        if target != self.base_pos:
            return target
        return self._nearest_unexplored_cell()

    def _nearest_unexplored_cell(self) -> GridPos | None:
        candidates: list[GridPos] = []
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col)
                if pos not in self.explored_cells and self.can_uav_enter(pos):
                    candidates.append(pos)
        if not candidates:
            return None
        return min(candidates, key=lambda target: self.manhattan(self.uav_pos, target))

    def _risk_penalty(self) -> float:
        penalty = 0.0
        exposure = 0.0
        for pos in [self.uav_pos, *self.ugv_positions]:
            if pos in self.smoke_cells:
                penalty += self.reward_weights["risk_smoke"]
                exposure += abs(self.reward_weights["risk_smoke"])
            if self._near_fire(pos):
                penalty += self.reward_weights["risk_near_fire"]
                exposure += abs(self.reward_weights["risk_near_fire"])
        self.risk_exposure += exposure
        return float(penalty)

    def _near_fire(self, pos: GridPos) -> bool:
        return any(self.manhattan(pos, fire) == 1 for fire in self.fire_cells)

    def _should_spread_fire(self) -> bool:
        return (
            self.fire_spread_interval > 0
            and self.fire_spread_prob > 0.0
            and self.step_count > 0
            and self.step_count % self.fire_spread_interval == 0
        )

    def _spread_fire(self) -> int:
        if not self.fire_cells:
            return 0
        new_fire = set(self.fire_cells)
        burned_now = 0
        protected = self._protected_base_cells()
        for fire in sorted(self.fire_cells):
            for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (fire[0] + delta[0], fire[1] + delta[1])
                if not self.in_bounds(nxt) or nxt in protected or nxt in self.fire_cells:
                    continue
                terrain_value = int(self.terrain[nxt])
                if terrain_value == WALL:
                    continue
                if terrain_value == DEBRIS:
                    self.debris_heat[nxt] = self.debris_heat.get(nxt, 0) + 1
                    if self.debris_heat[nxt] >= self.debris_burn_steps:
                        self.terrain[nxt] = BURNED
                        self.burned_debris.add(nxt)
                        burned_now += 1
                    continue
                if len(new_fire) >= self.max_fire_cells:
                    continue
                if self.rng.random() < self.fire_spread_prob:
                    new_fire.add(nxt)
        self.fire_cells = set(list(new_fire)[: self.max_fire_cells])
        return burned_now

    def _protected_base_cells(self) -> set[GridPos]:
        protected = {self.base_pos}
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                pos = (self.base_pos[0] + dr, self.base_pos[1] + dc)
                if self.in_bounds(pos):
                    protected.add(pos)
        return protected

    def _compute_smoke(self) -> set[GridPos]:
        smoke: set[GridPos] = set()
        for fire in self.fire_cells:
            for dr in range(-self.smoke_radius, self.smoke_radius + 1):
                for dc in range(-self.smoke_radius, self.smoke_radius + 1):
                    if abs(dr) + abs(dc) > self.smoke_radius:
                        continue
                    pos = (fire[0] + dr, fire[1] + dc)
                    if self.in_bounds(pos) and int(self.terrain[pos]) != WALL and pos not in self.fire_cells:
                        smoke.add(pos)
        return smoke

    def _burn_exposed_survivors(self) -> int:
        burned = 0
        for survivor in self.survivors:
            if survivor.delivered or survivor.burned or survivor.picked_by is not None:
                continue
            if survivor.position in self.fire_cells:
                survivor.burned = True
                survivor.discovered = True
                burned += 1
        return burned

    def _all_survivors_delivered(self) -> bool:
        return bool(self.survivors) and all(survivor.delivered for survivor in self.survivors)

    def _all_survivors_terminal(self) -> bool:
        return bool(self.survivors) and all(
            survivor.delivered or survivor.burned for survivor in self.survivors
        )

    def _get_obs(self) -> np.ndarray:
        max_row = max(1, self.height - 1)
        max_col = max(1, self.width - 1)
        features: list[float] = []
        features.extend([self.uav_pos[0] / max_row, self.uav_pos[1] / max_col])
        for pos in self.ugv_positions:
            features.extend([pos[0] / max_row, pos[1] / max_col])
        features.extend([self.base_pos[0] / max_row, self.base_pos[1] / max_col])
        for idx in range(self.max_survivors):
            if idx < len(self.survivors):
                survivor = self.survivors[idx]
                visible_pos = survivor.discovered or survivor.delivered or survivor.burned
                features.extend(
                    [
                        survivor.position[0] / max_row if visible_pos else 0.0,
                        survivor.position[1] / max_col if visible_pos else 0.0,
                        1.0 if survivor.discovered else 0.0,
                        1.0 if survivor.picked_by is not None else 0.0,
                        1.0 if survivor.delivered else 0.0,
                        1.0 if survivor.burned else 0.0,
                    ]
                )
            else:
                features.extend([0.0] * 6)
        denom_survivor = max(1, self.max_survivors - 1)
        for carrying in self.ugv_carrying:
            features.append(1.0 if carrying is not None else 0.0)
            features.append((float(carrying) / denom_survivor) if carrying is not None else 0.0)
        total_survivors = max(1, len(self.survivors))
        features.extend(
            [
                self.step_count / max(1, self.max_steps),
                min(1.0, len(self.fire_cells) / max(1, self.max_fire_cells)),
                min(1.0, len(self.smoke_cells) / max(1, self.width * self.height)),
                min(1.0, len(self.burned_debris) / max(1, self.width * self.height)),
                self.delivered_count / total_survivors,
                self.discovered_count / total_survivors,
                self.burned_count / total_survivors,
                min(1.0, self.risk_exposure / 80.0),
            ]
        )
        if self.include_exploration_features:
            target = self.exploration_target()
            features.extend(
                [
                    target[0] / max_row,
                    target[1] / max_col,
                    len(self.explored_cells) / max(1, self.width * self.height),
                ]
            )
        if self.include_guidance_features:
            uav_target = self._uav_progress_target() or self.base_pos
            features.extend(
                [
                    (uav_target[0] - self.uav_pos[0]) / max_row,
                    (uav_target[1] - self.uav_pos[1]) / max_col,
                ]
            )
            for idx, pos in enumerate(self.ugv_positions):
                ugv_target = self.ugv_target(idx)
                if ugv_target is None:
                    features.extend([0.0, 0.0, 0.0])
                else:
                    features.extend(
                        [
                            1.0,
                            (ugv_target[0] - pos[0]) / max_row,
                            (ugv_target[1] - pos[1]) / max_col,
                        ]
                    )
        features.extend(self._local_view(self.uav_pos, aerial=True))
        for pos in self.ugv_positions:
            features.extend(self._local_view(pos, aerial=False))
        obs = np.asarray(features, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _local_view(self, center: GridPos, aerial: bool) -> list[float]:
        radius = self.view_size // 2
        values: list[float] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                pos = (center[0] + dr, center[1] + dc)
                values.append(self._cell_code(pos, aerial) / 10.0)
        return values

    def _cell_code(self, pos: GridPos, aerial: bool) -> int:
        if not self.in_bounds(pos):
            return 10
        if pos in self.fire_cells:
            return 7
        if pos in self.smoke_cells:
            return 6
        if pos == self.base_pos:
            return 5
        if any(
            survivor.position == pos
            and survivor.discovered
            and not survivor.delivered
            and not survivor.burned
            for survivor in self.survivors
        ):
            return 8
        if pos == self.uav_pos or pos in self.ugv_positions:
            return 9
        terrain_value = int(self.terrain[pos])
        if terrain_value == WALL:
            return 4
        if terrain_value == DEBRIS and not aerial:
            return 3
        if terrain_value == BURNED:
            return 2
        if terrain_value == ROAD:
            return 1
        return 0

    def _update_explored_cells(self) -> None:
        radius = self.uav_view_radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                pos = (self.uav_pos[0] + dr, self.uav_pos[1] + dc)
                if self.in_bounds(pos) and self.can_uav_enter(pos):
                    self.explored_cells.add(pos)

    def exploration_target(self) -> GridPos:
        candidates = [
            waypoint
            for waypoint in self._exploration_waypoints()
            if waypoint not in self.explored_cells and self.can_uav_enter(waypoint)
        ]
        if not candidates:
            return self.base_pos
        return min(candidates, key=lambda pos: self.manhattan(self.uav_pos, pos))

    def _exploration_waypoints(self) -> list[GridPos]:
        spacing = max(2, self.exploration_waypoint_spacing)
        rows = list(range(2, self.height - 2, spacing))
        cols = list(range(2, self.width - 2, spacing))
        waypoints: list[GridPos] = []
        for idx, row in enumerate(rows):
            ordered_cols = cols if idx % 2 == 0 else list(reversed(cols))
            for col in ordered_cols:
                pos = (row, col)
                if self.can_uav_enter(pos):
                    waypoints.append(pos)
        return waypoints

    @property
    def delivered_count(self) -> int:
        return sum(1 for survivor in self.survivors if survivor.delivered)

    @property
    def discovered_count(self) -> int:
        return sum(1 for survivor in self.survivors if survivor.discovered)

    @property
    def burned_count(self) -> int:
        return sum(1 for survivor in self.survivors if survivor.burned)

    def _get_info(self) -> dict[str, Any]:
        return {
            "uav_pos": self.uav_pos,
            "ugv_positions": list(self.ugv_positions),
            "base_pos": self.base_pos,
            "survivors": [
                {
                    "position": survivor.position,
                    "discovered": survivor.discovered,
                    "picked_by": survivor.picked_by,
                    "delivered": survivor.delivered,
                    "burned": survivor.burned,
                }
                for survivor in self.survivors
            ],
            "fire_cells": set(self.fire_cells),
            "smoke_cells": set(self.smoke_cells),
            "burned_debris": set(self.burned_debris),
            "debris_heat": dict(self.debris_heat),
            "delivered_count": self.delivered_count,
            "discovered_count": self.discovered_count,
            "burned_count": self.burned_count,
            "total_survivors": len(self.survivors),
            "success": self._all_survivors_delivered(),
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "risk_exposure": self.risk_exposure,
            "explored_fraction": len(self.explored_cells) / max(1, self.width * self.height),
            "exploration_target": self.exploration_target(),
            "uav_guidance_target": self._uav_progress_target(),
            "ugv_guidance_targets": [self.ugv_target(idx) for idx in range(self.num_ugv)],
            "map_seed": self.rescue_map.seed if self.rescue_map else self.map_seed,
            "uav_path": list(self.uav_path),
            "ugv_paths": [list(path) for path in self.ugv_paths],
        }

    def _render_rgb(self) -> np.ndarray:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.colors as mcolors
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        display = self.terrain.copy()
        for row, col in self.fire_cells:
            display[row, col] = 5
        for row, col in self.smoke_cells:
            if display[row, col] in {EMPTY, ROAD, BURNED}:
                display[row, col] = 6
        cmap = mcolors.ListedColormap(
            [
                "#f7f8fa",
                "#2d3138",
                "#d6d9de",
                "#8c724f",
                "#626b78",
                "#e53935",
                "#8d8aa6",
            ]
        )
        norm = mcolors.BoundaryNorm(np.arange(-0.5, 7.5, 1), cmap.N)

        fig, (ax, panel) = plt.subplots(
            1,
            2,
            figsize=(9.8, 8.3),
            dpi=130,
            gridspec_kw={"width_ratios": [4.8, 1.35]},
            facecolor="#f3f5f7",
        )
        fig.suptitle(
            "Dynamic Fire Rescue: UAV Search + 2 UGV Delivery",
            fontsize=13,
            weight="bold",
            color="#111827",
            y=0.985,
        )
        ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=0.25, alpha=0.35)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        self._draw_paths(ax)
        self._draw_survivors(ax)
        self._draw_agents(ax, patches)
        br, bc = self.base_pos
        ax.scatter(bc, br, marker="s", s=180, color="#43a047", edgecolors="white", linewidths=1.1, zorder=5)
        ax.text(bc, br - 0.55, "BASE", ha="center", va="center", fontsize=7, color="#111827", weight="bold")

        panel.axis("off")
        self._draw_status_panel(panel)
        fig.tight_layout(pad=0.7, rect=[0.0, 0.0, 1.0, 0.955])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    def _draw_paths(self, ax) -> None:
        colors = ["#2e7d32", "#ef6c00", "#7b1fa2", "#00838f"]
        if len(self.uav_path) > 1:
            path = np.asarray(self.uav_path)
            ax.plot(path[:, 1], path[:, 0], color="#1565c0", linewidth=1.2, alpha=0.65, linestyle="--")
        for idx, path_list in enumerate(self.ugv_paths):
            if len(path_list) <= 1:
                continue
            path = np.asarray(path_list)
            ax.plot(path[:, 1], path[:, 0], color=colors[idx % len(colors)], linewidth=1.7, alpha=0.75)

    def _draw_survivors(self, ax) -> None:
        for idx, survivor in enumerate(self.survivors):
            row, col = survivor.position
            if survivor.delivered:
                color = "#66bb6a"
                marker = "P"
                label = f"S{idx}:done"
            elif survivor.burned:
                color = "#424242"
                marker = "X"
                label = f"S{idx}:lost"
            elif survivor.picked_by is not None:
                color = "#ffb300"
                marker = "*"
                label = f"S{idx}:carried"
            elif survivor.discovered:
                color = "#ffd54f"
                marker = "*"
                label = f"S{idx}"
            else:
                color = "#b0bec5"
                marker = "o"
                label = f"?S{idx}"
            ax.scatter(col, row, marker=marker, s=145, color=color, edgecolors="#111827", linewidths=0.8, zorder=6)
            ax.text(col + 0.34, row + 0.22, label, fontsize=7, color="#111827", weight="bold", zorder=7)

    def _draw_agents(self, ax, patches) -> None:
        row, col = self.uav_pos
        ax.scatter(col, row, marker="^", s=175, color="#1565c0", edgecolors="white", linewidths=1.1, zorder=8)
        ax.text(col + 0.28, row - 0.28, "UAV", fontsize=7, color="#0d47a1", weight="bold", zorder=9)
        colors = ["#2e7d32", "#ef6c00", "#7b1fa2", "#00838f"]
        for idx, pos in enumerate(self.ugv_positions):
            row, col = pos
            color = colors[idx % len(colors)]
            car = patches.FancyBboxPatch(
                (col - 0.36, row - 0.23),
                0.72,
                0.46,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.1,
                edgecolor="white",
                facecolor=color,
                zorder=8,
            )
            ax.add_patch(car)
            for wheel_x in [col - 0.22, col + 0.22]:
                wheel = patches.Circle((wheel_x, row + 0.27), 0.08, facecolor="#111827", edgecolor="none", zorder=9)
                ax.add_patch(wheel)
            ax.text(col, row - 0.45, f"UGV{idx}", fontsize=7, color=color, weight="bold", ha="center", zorder=9)

    def _draw_status_panel(self, panel) -> None:
        lines = [
            ("step", f"{self.step_count}/{self.max_steps}"),
            ("delivered", f"{self.delivered_count}/{len(self.survivors)}"),
            ("discovered", f"{self.discovered_count}/{len(self.survivors)}"),
            ("burned", f"{self.burned_count}"),
            ("fire cells", f"{len(self.fire_cells)}"),
            ("debris opened", f"{len(self.burned_debris)}"),
            ("risk", f"{self.risk_exposure:.1f}"),
            ("reward", f"{self.episode_reward:.1f}"),
        ]
        panel.text(0.0, 0.98, "Mission", fontsize=12, weight="bold", color="#111827", va="top")
        y = 0.91
        for key, value in lines:
            panel.text(0.0, y, key, fontsize=8, color="#64748b", va="top")
            panel.text(0.96, y, value, fontsize=9, color="#111827", va="top", ha="right", weight="bold")
            y -= 0.058
        y -= 0.014
        panel.text(0.0, y, "UGV load", fontsize=10, weight="bold", color="#111827", va="top")
        y -= 0.055
        for idx, carrying in enumerate(self.ugv_carrying):
            value = "empty" if carrying is None else f"S{carrying}"
            panel.text(0.0, y, f"UGV{idx}", fontsize=8, color="#64748b", va="top")
            panel.text(0.96, y, value, fontsize=9, color="#111827", va="top", ha="right", weight="bold")
            y -= 0.052
        y -= 0.014
        panel.text(0.0, y, "Legend", fontsize=10, weight="bold", color="#111827", va="top")
        y -= 0.052
        legend = [
            ("road", "#d6d9de"),
            ("building", "#2d3138"),
            ("debris", "#8c724f"),
            ("burned", "#626b78"),
            ("fire", "#e53935"),
            ("smoke", "#8d8aa6"),
        ]
        import matplotlib.patches as mpatches

        for name, color in legend:
            panel.add_patch(
                mpatches.Rectangle((0.0, y - 0.018), 0.09, 0.028, color=color, transform=panel.transAxes)
            )
            panel.text(0.12, y, name, fontsize=8, color="#111827", va="center")
            y -= 0.043

    def _render_ansi(self) -> str:
        chars = np.full((self.height, self.width), ".", dtype="<U1")
        chars[self.terrain == WALL] = "#"
        chars[self.terrain == ROAD] = "="
        chars[self.terrain == DEBRIS] = "d"
        chars[self.terrain == BURNED] = "x"
        for row, col in self.smoke_cells:
            if chars[row, col] == ".":
                chars[row, col] = "~"
        for row, col in self.fire_cells:
            chars[row, col] = "F"
        br, bc = self.base_pos
        chars[br, bc] = "B"
        for idx, survivor in enumerate(self.survivors):
            row, col = survivor.position
            if survivor.delivered:
                continue
            chars[row, col] = "S" if survivor.discovered else "?"
        ur, uc = self.uav_pos
        chars[ur, uc] = "A"
        for idx, (row, col) in enumerate(self.ugv_positions):
            chars[row, col] = str(idx)
        return "\n".join("".join(row) for row in chars)

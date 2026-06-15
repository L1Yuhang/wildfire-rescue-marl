"""Factorized neural policy for centralized multi-agent rescue control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
from torch import nn

from fire_rescue_rl.envs.fire_rescue_multi_ugv_env import ACTION_TO_DELTA


class FactorizedQNetwork(nn.Module):
    """Shared encoder with one 5-action Q head per controlled agent."""

    def __init__(self, obs_dim: int, num_agents: int, hidden_sizes: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden), nn.ReLU()])
            last_dim = hidden
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList(nn.Linear(last_dim, 5) for _ in range(num_agents))

    def forward(self, obs: th.Tensor) -> list[th.Tensor]:
        features = self.encoder(obs)
        return [head(features) for head in self.heads]


class FactorizedDQNPolicy:
    """Inference wrapper that composes per-agent heads into one joint action."""

    def __init__(self, model: FactorizedQNetwork, device: str | th.device = "cpu") -> None:
        self.device = th.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.num_agents = len(model.heads)

    @classmethod
    def load(cls, path: str | Path, device: str | th.device = "cpu") -> "FactorizedDQNPolicy":
        checkpoint = th.load(Path(path), map_location=device)
        model = FactorizedQNetwork(
            obs_dim=int(checkpoint["obs_dim"]),
            num_agents=int(checkpoint["num_agents"]),
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, device=device)

    def save(self, path: str | Path, obs_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        th.save(
            {
                "obs_dim": int(obs_dim),
                "num_agents": int(self.num_agents),
                "hidden_sizes": tuple(hidden_sizes),
                "state_dict": self.model.state_dict(),
            },
            Path(path),
        )

    def predict_obs(self, obs: np.ndarray) -> int:
        with th.no_grad():
            q_values = self._q_values(obs)
            actions = [int(q.argmax(dim=1).item()) for q in q_values]
        return encode_factorized_action(actions)

    def predict_env(self, env) -> int:
        obs = env._get_obs()
        with th.no_grad():
            q_values = self._q_values(obs)
            actions = [
                self._best_valid_action(q_values[0][0], env.uav_pos, env.can_uav_enter),
            ]
            for ugv_idx, pos in enumerate(env.ugv_positions):
                actions.append(self._best_valid_action(q_values[ugv_idx + 1][0], pos, env.can_ugv_enter))
        return encode_factorized_action(actions)

    def predict_uav_with_astar_ugv(self, env) -> int:
        from fire_rescue_rl.agents.astar_multi_ugv import CoverageAStarMultiUGVAgent

        obs = env._get_obs()
        with th.no_grad():
            q_values = self._q_values(obs)
            uav_action = self._best_valid_action(q_values[0][0], env.uav_pos, env.can_uav_enter)
        ugv_actions = CoverageAStarMultiUGVAgent()._ugv_actions(env)
        return encode_factorized_action([uav_action, *ugv_actions])

    def predict_astar_uav_with_ugv(self, env) -> int:
        from fire_rescue_rl.agents.astar_multi_ugv import CoverageAStarMultiUGVAgent

        uav_action = CoverageAStarMultiUGVAgent()._uav_action(env)
        obs = env._get_obs()
        with th.no_grad():
            q_values = self._q_values(obs)
            offset = 1 if len(q_values) == 1 + len(env.ugv_positions) else 0
            ugv_actions = [
                self._best_valid_action(q_values[ugv_idx + offset][0], pos, env.can_ugv_enter)
                for ugv_idx, pos in enumerate(env.ugv_positions)
            ]
        return encode_factorized_action([uav_action, *ugv_actions])

    def _q_values(self, obs: np.ndarray) -> list[th.Tensor]:
        obs_tensor = th.as_tensor(obs, dtype=th.float32, device=self.device).unsqueeze(0)
        return self.model(obs_tensor)

    @staticmethod
    def _best_valid_action(q_values: th.Tensor, pos: tuple[int, int], can_enter) -> int:
        ranked_actions = th.argsort(q_values, descending=True).detach().cpu().tolist()
        for action in ranked_actions:
            if int(action) == 0:
                return 0
            delta = ACTION_TO_DELTA[int(action)]
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if can_enter(nxt):
                return int(action)
        return 0


def encode_factorized_action(actions: list[int] | tuple[int, ...]) -> int:
    value = 0
    for action in actions:
        value = value * 5 + int(action)
    return int(value)

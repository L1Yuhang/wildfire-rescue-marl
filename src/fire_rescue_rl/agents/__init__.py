"""Baseline team agents."""

from fire_rescue_rl.agents.astar_multi_ugv import AStarMultiUGVAgent, CoverageAStarMultiUGVAgent
from fire_rescue_rl.agents.astar_team import AStarTeamAgent
from fire_rescue_rl.agents.factorized_q_policy import FactorizedDQNPolicy
from fire_rescue_rl.agents.multi_ugv_baselines import GreedyMultiUGVAgent, RandomMultiUGVAgent

__all__ = [
    "AStarTeamAgent",
    "AStarMultiUGVAgent",
    "CoverageAStarMultiUGVAgent",
    "FactorizedDQNPolicy",
    "GreedyMultiUGVAgent",
    "RandomMultiUGVAgent",
]

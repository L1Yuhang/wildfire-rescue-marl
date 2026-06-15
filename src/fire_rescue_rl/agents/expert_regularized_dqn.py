"""DQN variant with an expert behavior-cloning regularizer."""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class ExpertRegularizedDQN(DQN):
    """SB3 DQN with an auxiliary cross-entropy loss on expert samples.

    The TD objective is still the main RL update.  The expert term acts as a
    stabilizer so fine-tuning does not erase a previously learned cooperative
    rescue policy in the 125-action joint space.
    """

    def set_expert_dataset(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        bc_coef: float = 0.1,
        bc_batch_size: int = 128,
    ) -> None:
        self.expert_observations = th.as_tensor(observations, dtype=th.float32, device=self.device)
        self.expert_actions = th.as_tensor(actions, dtype=th.long, device=self.device)
        self.bc_coef = float(bc_coef)
        self.bc_batch_size = int(bc_batch_size)
        self._expert_rng = np.random.default_rng(self.seed)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        td_losses = []
        bc_losses = []
        has_expert = hasattr(self, "expert_observations") and len(self.expert_actions) > 0
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            td_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            loss = td_loss

            bc_loss = th.zeros((), device=self.device)
            if has_expert and self.bc_coef > 0.0:
                indices = self._expert_rng.integers(
                    0,
                    len(self.expert_actions),
                    size=min(self.bc_batch_size, len(self.expert_actions)),
                )
                expert_q = self.q_net(self.expert_observations[indices])
                bc_loss = F.cross_entropy(expert_q, self.expert_actions[indices])
                loss = loss + self.bc_coef * bc_loss

            losses.append(float(loss.detach().cpu()))
            td_losses.append(float(td_loss.detach().cpu()))
            bc_losses.append(float(bc_loss.detach().cpu()))

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/td_loss", np.mean(td_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))

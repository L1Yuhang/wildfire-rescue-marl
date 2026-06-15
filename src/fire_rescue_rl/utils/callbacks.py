"""Callbacks."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class RewardCsvCallback(BaseCallback):
    def __init__(self, csv_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.csv_path = Path(csv_path)
        self.rows = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        reward = float(rewards[0]) if rewards is not None and len(rewards) else 0.0
        done = int(bool(dones[0])) if dones is not None and len(dones) else 0
        self.rows.append((self.num_timesteps, reward, done))
        if len(self.rows) >= 2000:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8") as handle:
            if write_header:
                handle.write("timesteps,reward,done\n")
            for row in self.rows:
                handle.write(f"{row[0]},{row[1]},{row[2]}\n")
        self.rows.clear()


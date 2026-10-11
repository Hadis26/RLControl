"""Validation and checkpoint utilities for DQN training."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


def evaluate_model(
    model: torch.nn.Module,
    episodes: int = 20,
    seed: int = 20_000,
) -> dict:
    """Evaluate an in-memory DQN with greedy actions."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    env = gym.make("CartPole-v1")
    returns = []

    try:
        with torch.inference_mode():
            for episode in range(episodes):
                state, _ = env.reset(seed=seed + episode)

                terminated = False
                truncated = False
                episode_return = 0.0

                while not (terminated or truncated):
                    state_tensor = torch.as_tensor(
                        state,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    q_values = model(state_tensor)
                    action = int(q_values.argmax(dim=1).item())

                    state, reward, terminated, truncated, _ = env.step(action)

                    episode_return += reward

                returns.append(episode_return)
    finally:
        env.close()
        model.train(was_training)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }


class CheckpointManager:
    """Track the best validation score and save checkpoints."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.best_score = float("-inf")
        self.best_episode = None

    def save(
        self,
        model: torch.nn.Module,
        episode: int,
        config: dict,
        filename: str,
        score: float | None = None,
    ) -> Path:
        path = self.directory / filename

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "episode": episode,
                "validation_score": score,
                "config": config,
            },
            path,
        )

        return path

    def update_best(
        self,
        model: torch.nn.Module,
        episode: int,
        score: float,
        config: dict,
    ) -> bool:
        """Save only when validation performance improves."""

        if not np.isfinite(score):
            raise ValueError("Validation score must be finite.")

        if score <= self.best_score:
            return False

        self.best_score = score
        self.best_episode = episode

        self.save(
            model=model,
            episode=episode,
            config=config,
            filename="dqn_best.pt",
            score=score,
        )

        return True

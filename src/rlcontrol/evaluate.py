"""Evaluate random and trained DQN policies on CartPole-v1."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rlcontrol.network import DQN


def evaluate_policy(
    policy,
    n_episodes: int = 100,
    seed: int = 10_000,
) -> dict:
    """Evaluate a policy with reproducible episode seeds."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")

    env = gym.make("CartPole-v1")
    rewards = []

    try:
        for episode in range(n_episodes):
            episode_seed = seed + episode

            state, _ = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)

            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = policy(state, env)

                state, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward

            rewards.append(total_reward)
    finally:
        env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "episode_rewards": [float(r) for r in rewards],
    }


def random_policy(state, env) -> int:
    return int(env.action_space.sample())


def load_dqn(checkpoint_path: str) -> DQN:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = DQN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def make_dqn_policy(model):
    def policy(state, env) -> int:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)

        return int(q_values.argmax(dim=1).item())

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/dqn_final.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    model = load_dqn(args.checkpoint)

    baseline = evaluate_policy(
        random_policy,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    trained = evaluate_policy(
        make_dqn_policy(model),
        n_episodes=args.episodes,
        seed=args.seed,
    )

    print("\nEvaluation — CartPole-v1")
    print("-" * 48)

    for name, result in [
        ("Random", baseline),
        ("DQN", trained),
    ]:
        print(
            f"{name:8s} | "
            f"Mean: {result['mean_reward']:7.2f} | "
            f"Std: {result['std_reward']:7.2f} | "
            f"Min: {result['min_reward']:5.0f} | "
            f"Max: {result['max_reward']:5.0f}"
        )

    output = Path("results/metrics/evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as file:
        json.dump(
            {
                "environment": "CartPole-v1",
                "episodes": args.episodes,
                "evaluation_seed": args.seed,
                "checkpoint": args.checkpoint,
                "random": baseline,
                "dqn": trained,
            },
            file,
            indent=2,
        )

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()

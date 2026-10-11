"""Train a DQN agent on Gymnasium CartPole-v1."""

import argparse
import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rlcontrol.agent import DQNAgent
from rlcontrol.validation import evaluate_model, CheckpointManager


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_schedule(
    step: int,
    start: float = 1.0,
    end: float = 0.05,
    decay_steps: int = 20_000,
) -> float:
    """Linearly decrease epsilon over environment steps."""
    fraction = min(step / decay_steps, 1.0)
    return start + fraction * (end - start)


def save_learning_curve(rewards, output_path: Path) -> None:
    """Plot episode returns and a moving average."""
    window = 20

    moving_average = [
        np.mean(rewards[max(0, i - window + 1) : i + 1]) for i in range(len(rewards))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, label="Episode return")
    plt.plot(
        moving_average,
        linewidth=2,
        label=f"Moving average ({window} episodes)",
    )

    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("DQN Training — CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_validation_curve(records, output_path: Path):
    """Plot greedy-policy validation performance."""

    episodes = [r["episode"] for r in records]
    means = [r["mean_return"] for r in records]
    stds = [r["std_return"] for r in records]

    means = np.asarray(means)
    stds = np.asarray(stds)

    plt.figure(figsize=(10, 5))

    plt.plot(
        episodes,
        means,
        marker="o",
        markersize=3,
        label="Validation mean",
    )

    plt.fill_between(
        episodes,
        np.maximum(0, means - stds),
        np.minimum(500, means + stds),
        alpha=0.2,
        label="Mean ± 1 standard deviation",
    )

    plt.axhline(
        500,
        linestyle="--",
        color="gray",
        label="Maximum CartPole return",
    )

    plt.xlabel("Training episode")
    plt.ylabel("Greedy-policy return")
    plt.title("DQN Validation Performance")
    plt.ylim(0, 520)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def train(
    episodes: int = 600,
    seed: int = 42,
    output_dir: str = "results",
) -> None:
    if episodes <= 0:
        raise ValueError("Episodes must be positive.")

    set_seed(seed)

    output = Path(output_dir)
    figures_dir = output / "figures"
    metrics_dir = output / "metrics"
    checkpoints_dir = output / "checkpoints"

    for directory in (
        figures_dir,
        metrics_dir,
        checkpoints_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "environment": "CartPole-v1",
        "episodes": episodes,
        "seed": seed,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_capacity": 100_000,
        "target_update_frequency": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 20_000,
        "device": "cpu",
        "validation_frequency": 20,
        "validation_episodes": 20,
        "validation_seed": 20_000,
    }

    agent = DQNAgent(
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        buffer_capacity=config["buffer_capacity"],
        target_update_frequency=config["target_update_frequency"],
        seed=seed,
        device=config["device"],
    )

    checkpoint_manager = CheckpointManager(checkpoints_dir)
    validation_records = []

    env = gym.make(config["environment"])
    env.action_space.seed(seed)

    episode_rewards = []
    records = []
    global_step = 0

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)

            terminated = False
            truncated = False
            total_reward = 0.0
            losses = []

            while not (terminated or truncated):
                epsilon = epsilon_schedule(
                    global_step,
                    start=config["epsilon_start"],
                    end=config["epsilon_end"],
                    decay_steps=config["epsilon_decay_steps"],
                )

                action = agent.select_action(state, epsilon)

                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                )

                loss = agent.learn()

                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += reward
                global_step += 1

            episode_rewards.append(total_reward)

            record = {
                "episode": episode,
                "reward": total_reward,
                "epsilon": epsilon,
                "mean_loss": (float(np.mean(losses)) if losses else None),
                "global_step": global_step,
            }
            records.append(record)

            if episode % 20 == 0 or episode == 1:
                recent_mean = np.mean(episode_rewards[-20:])

                print(
                    f"Episode {episode:4d}/{episodes} | "
                    f"Mean20: {recent_mean:7.2f} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Steps: {global_step}"
                )

            if episode % config["validation_frequency"] == 0:
                validation = evaluate_model(
                    agent.policy_network,
                    episodes=config["validation_episodes"],
                    seed=config["validation_seed"],
                )

                score = validation["mean_return"]

                improved = checkpoint_manager.update_best(
                    model=agent.policy_network,
                    episode=episode,
                    score=score,
                    config=config,
                )

                validation_records.append(
                    {
                        "episode": episode,
                        **validation,
                        "is_best": improved,
                    }
                )

                print(
                    f"  Validation | Mean: {score:.2f} | "
                    f"Std: {validation['std_return']:.2f} | "
                    f"Best: {checkpoint_manager.best_score:.2f}"
                    + (" | NEW BEST" if improved else "")
                )

    finally:
        env.close()

    checkpoint_path = checkpoints_dir / "dqn_final.pt"

    torch.save(
        {
            "model_state_dict": (agent.policy_network.state_dict()),
            "config": config,
            "training_steps": agent.training_steps,
        },
        checkpoint_path,
    )

    with (metrics_dir / "training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "reward",
                "epsilon",
                "mean_loss",
                "global_step",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    validation_csv = metrics_dir / "validation.csv"

    with validation_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "mean_return",
                "std_return",
                "min_return",
                "max_return",
                "is_best",
            ],
        )
        writer.writeheader()
        writer.writerows(validation_records)

    with (metrics_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    save_learning_curve(
        episode_rewards,
        figures_dir / "training_curve.png",
    )


    if validation_records:
        save_validation_curve(
            validation_records,
            figures_dir / "validation_curve.png",
        )

        print(
            f"Best validation mean: "
            f"{checkpoint_manager.best_score:.2f}"
        )
        print(
            f"Best checkpoint episode: "
            f"{checkpoint_manager.best_episode}"
        )


    print("\nTraining complete.")
    print(f"Mean reward (last 20): {np.mean(episode_rewards[-20:]):.2f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent on CartPole-v1.")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )

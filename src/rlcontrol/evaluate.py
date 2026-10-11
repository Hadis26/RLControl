import gymnasium as gym
import numpy as np


def evaluate_random_policy(
    env_name: str = "CartPole-v1",
    n_episodes: int = 100,
    seed: int = 42,
) -> dict:
    """Evaluate a uniformly random policy on CartPole."""

    env = gym.make(env_name)
    rewards = []

    try:
        for episode in range(n_episodes):
            observation, info = env.reset(seed=seed + episode)
            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()

                observation, reward, terminated, truncated, info = env.step(
                    action
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
    }


if __name__ == "__main__":
    results = evaluate_random_policy()

    print("Random Policy Evaluation")
    print("------------------------")

    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")

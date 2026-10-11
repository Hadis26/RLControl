"""Analyze the control behavior of a trained DQN agent."""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlcontrol.evaluate import load_dqn, make_dqn_policy


def collect_trajectory(
    checkpoint_path: str,
    seed: int = 10_000,
):
    """Collect one deterministic greedy-policy rollout."""

    model = load_dqn(checkpoint_path)
    policy = make_dqn_policy(model)

    env = gym.make("CartPole-v1")

    try:
        state, _ = env.reset(seed=seed)

        states = [np.asarray(state, dtype=float).copy()]
        actions = []
        rewards = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy(state, env)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            actions.append(action)
            rewards.append(reward)
            states.append(
                np.asarray(next_state, dtype=float).copy()
            )

            state = next_state

    finally:
        env.close()

    final_state = states[-1]

    if truncated:
        termination_reason = "Time limit reached"
    elif abs(final_state[0]) > 2.4:
        termination_reason = "Cart position boundary exceeded"
    elif abs(final_state[2]) > np.deg2rad(12):
        termination_reason = "Pole angle boundary exceeded"
    elif terminated:
        termination_reason = "Other environment termination"
    else:
        termination_reason = "Unknown"

    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "seed": seed,
        "termination_reason": termination_reason,
    }


def plot_trajectory(trajectory, output_path: Path):
    """Visualize the state and control trajectories."""

    states = trajectory["states"]
    actions = trajectory["actions"]

    # CartPole-v1 uses a simulation step of 0.02 seconds.
    dt = 0.02
    time = np.arange(len(states)) * dt
    action_time = time[:-1]

    fig, axes = plt.subplots(
        5, 1, figsize=(11, 12), sharex=True
    )

    labels = [
        "Cart position [m]",
        "Cart velocity [m/s]",
        "Pole angle [deg]",
        "Pole angular velocity [rad/s]",
    ]

    state_data = [
        states[:, 0],
        states[:, 1],
        np.rad2deg(states[:, 2]),
        states[:, 3],
    ]

    for i in range(4):
        axes[i].plot(time, state_data[i])
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)

    # Environment termination boundaries
    axes[0].axhline(2.4, color="red", linestyle="--")
    axes[0].axhline(-2.4, color="red", linestyle="--")

    axes[2].axhline(12, color="red", linestyle="--")
    axes[2].axhline(-12, color="red", linestyle="--")

    axes[4].step(
        action_time,
        actions,
        where="post",
    )
    axes[4].set_ylabel("Action")
    axes[4].set_yticks([0, 1])
    axes[4].set_xlabel("Time [s]")
    axes[4].grid(alpha=0.3)

    fig.suptitle(
        f"DQN Controller Analysis — Seed {trajectory['seed']}\n"
        f"{trajectory['termination_reason']}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_trajectory(trajectory, output_path: Path):
    """Export a trajectory to CSV for reproducibility."""

    states = trajectory["states"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "step",
            "time_s",
            "cart_position_m",
            "cart_velocity_m_s",
            "pole_angle_rad",
            "pole_angular_velocity_rad_s",
            "action",
            "reward",
        ])

        for i, state in enumerate(states):
            writer.writerow([
                i,
                i * 0.02,
                *state,
                int(actions[i]) if i < len(actions) else "",
                float(rewards[i]) if i < len(rewards) else "",
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/dqn_final.pt",
    )
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()

    output_figures = Path("results/figures")
    output_metrics = Path("results/metrics")

    output_figures.mkdir(parents=True, exist_ok=True)
    output_metrics.mkdir(parents=True, exist_ok=True)

    trajectory = collect_trajectory(
        args.checkpoint, args.seed
    )

    figure_path = (
        output_figures / f"trajectory_seed_{args.seed}.png"
    )
    csv_path = (
        output_metrics / f"trajectory_seed_{args.seed}.csv"
    )

    plot_trajectory(trajectory, figure_path)
    save_trajectory(trajectory, csv_path)

    print("\nDQN Controller Analysis")
    print("-" * 45)
    print(f"Seed: {args.seed}")
    print(f"Episode return: {trajectory['rewards'].sum():.0f}")
    print(
        f"Episode duration: "
        f"{len(trajectory['actions']) * 0.02:.2f} s"
    )
    print(f"Termination: {trajectory['termination_reason']}")
    print(f"Figure: {figure_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

"""Robustness evaluation for a trained CartPole DQN."""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rlcontrol.evaluate import load_dqn


# State order: x, x_dot, theta, theta_dot
NOISE_LEVELS = [0.0, 0.01, 0.025, 0.05, 0.1]
INITIAL_ANGLES_DEG = [-10, -5, 0, 5, 10]
INITIAL_POSITIONS_M = [-1.5, -0.75, 0, 0.75, 1.5]


def choose_action(model, observation):
    """Greedy DQN action."""
    device = next(model.parameters()).device

    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state)

    return int(q_values.argmax(dim=1).item())


def run_episode(
    model,
    seed: int,
    noise_std: float = 0.0,
    initial_state=None,
):
    """Evaluate one episode with optional observation noise."""

    if noise_std < 0:
        raise ValueError("Noise standard deviation cannot be negative.")

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed + 1_000_000)

    try:
        observation, _ = env.reset(seed=seed)

        if initial_state is not None:
            initial_state = np.asarray(
                initial_state, dtype=np.float32
            )
            if initial_state.shape != (4,):
                raise ValueError("Initial state must have shape (4,).")

            # Controlled reset for standard Gymnasium CartPole-v1.
            # Uses environment internals, not the public reset API.
            env.unwrapped.state = initial_state.copy()
            observation = initial_state.copy()

        states = [np.asarray(observation).copy()]
        actions = []

        terminated = False
        truncated = False
        total_return = 0.0

        while not (terminated or truncated):
            measured_state = (
                observation
                + rng.normal(
                    0.0, noise_std, size=4
                )
            )

            action = choose_action(model, measured_state)

            observation, reward, terminated, truncated, _ = (
                env.step(action)
            )

            states.append(np.asarray(observation).copy())
            actions.append(action)
            total_return += reward

        states = np.asarray(states)

        return {
            "return": float(total_return),
            "success": bool(truncated and total_return == 500),
            "max_abs_position": float(
                np.max(np.abs(states[:, 0]))
            ),
            "max_abs_angle_deg": float(
                np.max(np.abs(np.rad2deg(states[:, 2])))
            ),
            "final_position": float(states[-1, 0]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "states": states,
            "actions": np.asarray(actions),
        }
    finally:
        env.close()


def summarize(results):
    """Summarize a collection of evaluation episodes."""
    returns = np.asarray([r["return"] for r in results])

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "success_rate": float(
            np.mean([r["success"] for r in results])
        ),
        "mean_max_abs_position": float(
            np.mean([r["max_abs_position"] for r in results])
        ),
        "mean_max_abs_angle_deg": float(
            np.mean([r["max_abs_angle_deg"] for r in results])
        ),
    }


def evaluate_noise(model, episodes=100, seed=40_000):
    """Evaluate sensitivity to observation noise."""
    records = []

    for noise_std in NOISE_LEVELS:
        results = [
            run_episode(
                model,
                seed=seed + i,
                noise_std=noise_std,
            )
            for i in range(episodes)
        ]

        summary = summarize(results)
        records.append({
            "noise_std": noise_std,
            **summary,
        })

        print(
            f"Noise={noise_std:.3f} | "
            f"Mean={summary['mean_return']:.2f} | "
            f"Success={summary['success_rate']:.1%}"
        )

    return records


def evaluate_initial_conditions(model, seed=50_000):
    """Evaluate a grid of explicitly initialized states."""
    records = []

    for position in INITIAL_POSITIONS_M:
        for angle_deg in INITIAL_ANGLES_DEG:
            state = np.array([
                position,
                0.0,
                np.deg2rad(angle_deg),
                0.0,
            ], dtype=np.float32)

            result = run_episode(
                model,
                seed=seed,
                initial_state=state,
            )

            records.append({
                "initial_position": position,
                "initial_angle_deg": angle_deg,
                "return": result["return"],
                "success": result["success"],
                "max_abs_position": result["max_abs_position"],
                "max_abs_angle_deg": result["max_abs_angle_deg"],
            })

    return records


def save_csv(records, path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)


def plot_noise(records, path):
    noise = [r["noise_std"] for r in records]
    means = [r["mean_return"] for r in records]
    stds = [r["std_return"] for r in records]

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        noise,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
    )
    plt.axhline(
        500,
        color="gray",
        linestyle="--",
        label="Maximum return",
    )
    plt.xlabel("Observation noise std (per state component)")
    plt.ylabel("Episode return")
    plt.title("DQN Sensitivity to Observation Noise")
    plt.ylim(0, 530)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_initial_conditions(records, path):
    matrix = np.zeros((
        len(INITIAL_POSITIONS_M),
        len(INITIAL_ANGLES_DEG),
    ))

    for record in records:
        i = INITIAL_POSITIONS_M.index(
            record["initial_position"]
        )
        j = INITIAL_ANGLES_DEG.index(
            record["initial_angle_deg"]
        )
        matrix[i, j] = record["return"]

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        matrix,
        origin="lower",
        vmin=0,
        vmax=500,
        cmap="viridis",
        aspect="auto",
    )

    ax.set_xticks(range(len(INITIAL_ANGLES_DEG)))
    ax.set_xticklabels(INITIAL_ANGLES_DEG)
    ax.set_yticks(range(len(INITIAL_POSITIONS_M)))
    ax.set_yticklabels(INITIAL_POSITIONS_M)

    ax.set_xlabel("Initial pole angle [deg]")
    ax.set_ylabel("Initial cart position [m]")
    ax.set_title("DQN Initial-Condition Sensitivity")

    fig.colorbar(image, ax=ax, label="Episode return")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/reference_models/dqn_seed42_best.pt",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="results/experiments/exp008",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("episodes must be positive.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = load_dqn(args.checkpoint)

    print("\nObservation Noise Evaluation")
    print("-" * 48)
    noise_records = evaluate_noise(
        model, episodes=args.episodes
    )

    print("\nInitial-Condition Evaluation")
    print("-" * 48)
    initial_records = evaluate_initial_conditions(model)

    for record in initial_records:
        print(
            f"x0={record['initial_position']:+.2f} m | "
            f"theta0={record['initial_angle_deg']:+.0f} deg | "
            f"Return={record['return']:.0f}"
        )

    save_csv(
        noise_records,
        output / "noise_results.csv",
    )
    save_csv(
        initial_records,
        output / "initial_condition_results.csv",
    )

    plot_noise(
        noise_records,
        output / "noise_sensitivity.png",
    )
    plot_initial_conditions(
        initial_records,
        output / "initial_condition_heatmap.png",
    )

    with (output / "metadata.json").open("w") as file:
        json.dump({
            "checkpoint": args.checkpoint,
            "environment": "CartPole-v1",
            "episodes_per_noise_level": args.episodes,
            "noise_seed_start": 40000,
            "initial_condition_seed": 50000,
            "noise_levels": NOISE_LEVELS,
            "initial_angles_deg": INITIAL_ANGLES_DEG,
            "initial_positions_m": INITIAL_POSITIONS_M,
        }, file, indent=2)

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()

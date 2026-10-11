"""Run independent DQN training experiments across seeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rlcontrol.train import train
from rlcontrol.evaluate import (
    evaluate_policy,
    load_dqn,
    make_dqn_policy,
)


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes: int = 100,
    seed: int = 30_000,
) -> dict:
    model = load_dqn(str(checkpoint_path))

    return evaluate_policy(
        make_dqn_policy(model),
        n_episodes=episodes,
        seed=seed,
    )


def run_single_seed(
    seed: int,
    episodes: int,
    test_episodes: int,
    root: Path,
) -> list[dict]:
    """Train one seed and evaluate both saved checkpoints."""

    output = root / f"seed_{seed}"

    print(f"\n{'=' * 55}")
    print(f"Training seed: {seed}")
    print(f"{'=' * 55}")

    train(
        episodes=episodes,
        seed=seed,
        output_dir=str(output),
    )

    records = []

    for checkpoint_type in ("best", "final"):
        checkpoint_path = (
            output
            / "checkpoints"
            / f"dqn_{checkpoint_type}.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        result = evaluate_checkpoint(
            checkpoint_path,
            episodes=test_episodes,
        )

        record = {
            "training_seed": seed,
            "checkpoint": checkpoint_type,
            "checkpoint_episode": checkpoint.get("episode", episodes),
            "mean_return": result["mean_reward"],
            "std_return": result["std_reward"],
            "min_return": result["min_reward"],
            "max_return": result["max_reward"],
            "success_rate": sum(
                r >= 500 for r in result["episode_rewards"]
            ) / len(result["episode_rewards"]),
        }

        records.append(record)

        metrics_path = (
            output / "metrics"
            / f"evaluation_{checkpoint_type}.json"
        )

        with metrics_path.open("w") as file:
            json.dump(
                {
                    "training_seed": seed,
                    "checkpoint": checkpoint_type,
                    "test_seed": 30_000,
                    "test_episodes": test_episodes,
                    **result,
                },
                file,
                indent=2,
            )

        print(
            f"{checkpoint_type.upper():5s} | "
            f"Mean: {record['mean_return']:.2f} | "
            f"Success: {record['success_rate']:.0%}"
        )

    return records


def write_summary(records: list[dict], root: Path):
    """Save per-run metrics and aggregate statistics."""

    fieldnames = [
        "training_seed",
        "checkpoint",
        "checkpoint_episode",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "success_rate",
    ]

    with (root / "summary.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\nAggregate results")
    print("-" * 55)

    for checkpoint_type in ("best", "final"):
        subset = [
            r for r in records
            if r["checkpoint"] == checkpoint_type
        ]

        if not subset:
            continue

        means = np.array(
            [r["mean_return"] for r in subset]
        )
        success = np.array(
            [r["success_rate"] for r in subset]
        )

        print(
            f"{checkpoint_type.upper():5s} | "
            f"Across-run mean: {means.mean():.2f} | "
            f"Across-run std: {means.std():.2f} | "
            f"Mean success rate: {success.mean():.1%}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[123, 456, 789, 2026],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-dir",
        default="results/experiments/exp007",
    )

    args = parser.parse_args()

    if (
        args.episodes < 20
        or args.episodes % 20 != 0
        or args.test_episodes <= 0
    ):
        parser.error(
            "Episodes must be a positive multiple of 20; "
            "test episodes must be positive."
        )

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Training seeds must be unique.")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    records = []

    for seed in args.seeds:
        records.extend(
            run_single_seed(
                seed=seed,
                episodes=args.episodes,
                test_episodes=args.test_episodes,
                root=root,
            )
        )

    write_summary(records, root)


if __name__ == "__main__":
    main()

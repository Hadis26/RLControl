import csv

import pytest

from rlcontrol.multi_seed import (
    evaluate_checkpoint,
    write_summary,
)
from rlcontrol.network import DQN
import torch


def test_evaluate_checkpoint(tmp_path):
    model = DQN()

    path = tmp_path / "test_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "episode": 20,
            "config": {},
        },
        path,
    )

    result = evaluate_checkpoint(path, episodes=3, seed=30000)

    assert len(result["episode_rewards"]) == 3
    assert 0 < result["mean_reward"] <= 500


def test_summary_output(tmp_path):
    records = [
        {
            "training_seed": 123,
            "checkpoint": "best",
            "checkpoint_episode": 100,
            "mean_return": 450.0,
            "std_return": 20.0,
            "min_return": 400.0,
            "max_return": 500.0,
            "success_rate": 0.5,
        },
        {
            "training_seed": 123,
            "checkpoint": "final",
            "checkpoint_episode": 1200,
            "mean_return": 150.0,
            "std_return": 30.0,
            "min_return": 100.0,
            "max_return": 210.0,
            "success_rate": 0.0,
        },
    ]

    write_summary(records, tmp_path)

    with (tmp_path / "summary.csv").open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["checkpoint"] == "best"
    assert float(rows[0]["mean_return"]) == 450.0
    assert rows[1]["checkpoint"] == "final"

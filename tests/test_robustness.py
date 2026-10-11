import numpy as np
import pytest
import torch

from rlcontrol.network import DQN
from rlcontrol.robustness import (
    run_episode,
    summarize,
    evaluate_initial_conditions,
)


@pytest.fixture
def model():
    torch.manual_seed(42)
    network = DQN()
    network.eval()
    return network


def test_nominal_episode(model):
    result = run_episode(model, seed=40000)

    assert 0 < result["return"] <= 500
    assert len(result["actions"]) == result["return"]
    assert len(result["states"]) == len(result["actions"]) + 1


def test_noise_is_reproducible(model):
    first = run_episode(model, seed=40000, noise_std=0.025)
    second = run_episode(model, seed=40000, noise_std=0.025)

    assert first["return"] == second["return"]
    assert np.array_equal(
        first["states"],
        second["states"],
    )


def test_negative_noise_rejected(model):
    with pytest.raises(ValueError):
        run_episode(model, seed=42, noise_std=-0.1)


def test_initial_condition_grid(model):
    records = evaluate_initial_conditions(model)

    assert len(records) == 25
    assert all(0 < record["return"] <= 500 for record in records)


def test_summary(model):
    results = [run_episode(model, seed=40000 + i) for i in range(3)]

    summary = summarize(results)

    assert 0 < summary["mean_return"] <= 500
    assert 0 <= summary["success_rate"] <= 1
    assert summary["std_return"] >= 0

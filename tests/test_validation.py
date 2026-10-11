import pytest
import torch

from rlcontrol.network import DQN
from rlcontrol.validation import (
    CheckpointManager,
    evaluate_model,
)


def test_validation_returns_valid_statistics():
    torch.manual_seed(42)
    model = DQN()

    result = evaluate_model(model, episodes=3, seed=20000)

    assert 0 < result["min_return"] <= 500
    assert result["min_return"] <= result["mean_return"]
    assert result["mean_return"] <= result["max_return"]
    assert result["std_return"] >= 0


def test_validation_is_reproducible():
    torch.manual_seed(42)
    model = DQN()

    first = evaluate_model(model, episodes=3, seed=20000)
    second = evaluate_model(model, episodes=3, seed=20000)

    assert first == second


def test_validation_restores_model_mode():
    model = DQN()
    model.train()

    evaluate_model(model, episodes=1)

    assert model.training


def test_best_checkpoint_selection(tmp_path):
    model = DQN()
    manager = CheckpointManager(tmp_path)
    config = {"environment": "CartPole-v1"}

    assert manager.update_best(model, 20, 100.0, config)
    assert not manager.update_best(model, 40, 90.0, config)
    assert manager.update_best(model, 60, 150.0, config)

    assert manager.best_score == 150.0
    assert manager.best_episode == 60

    checkpoint = torch.load(
        tmp_path / "dqn_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["episode"] == 60
    assert checkpoint["validation_score"] == 150.0


def test_nonfinite_validation_score_rejected(tmp_path):
    model = DQN()
    manager = CheckpointManager(tmp_path)

    with pytest.raises(ValueError):
        manager.update_best(model, 20, float("nan"), {})

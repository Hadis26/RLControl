import numpy as np
import torch

from rlcontrol.agent import DQNAgent


def make_agent(**kwargs):
    defaults = {
        "batch_size": 4,
        "buffer_capacity": 20,
        "seed": 42,
    }
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def populate_memory(agent, n=8):
    for i in range(n):
        state = np.full(4, i / 10, dtype=np.float32)
        agent.store_transition(
            state=state,
            action=i % 2,
            reward=1.0,
            next_state=state + 0.1,
            terminated=(i % 3 == 0),
        )


def test_select_action_is_valid():
    agent = make_agent()
    state = np.zeros(4, dtype=np.float32)

    for epsilon in (0.0, 0.5, 1.0):
        action = agent.select_action(state, epsilon)
        assert action in (0, 1)


def test_learning_waits_for_enough_samples():
    agent = make_agent(batch_size=4)
    populate_memory(agent, n=3)

    assert agent.learn() is None
    assert agent.training_steps == 0


def test_learning_updates_policy_parameters():
    agent = make_agent()
    populate_memory(agent)

    before = [
        p.detach().clone()
        for p in agent.policy_network.parameters()
    ]

    loss = agent.learn()

    assert loss is not None
    assert np.isfinite(loss)
    assert agent.training_steps == 1

    assert any(
        not torch.equal(old, new)
        for old, new in zip(
            before, agent.policy_network.parameters()
        )
    )


def test_target_network_synchronization():
    agent = make_agent(target_update_frequency=1)
    populate_memory(agent)

    agent.learn()

    for policy_param, target_param in zip(
        agent.policy_network.parameters(),
        agent.target_network.parameters(),
    ):
        assert torch.equal(policy_param, target_param)


def test_invalid_epsilon():
    import pytest

    agent = make_agent()
    state = np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError):
        agent.select_action(state, epsilon=-0.1)

    with pytest.raises(ValueError):
        agent.select_action(state, epsilon=1.1)

import torch

from rlcontrol.network import DQN


def test_dqn_output_shape():
    model = DQN(state_dim=4, action_dim=2)

    states = torch.randn(8, 4)
    q_values = model(states)

    assert q_values.shape == (8, 2)


def test_dqn_supports_backpropagation():
    model = DQN()

    states = torch.randn(8, 4)
    q_values = model(states)

    loss = q_values.mean()
    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
    )

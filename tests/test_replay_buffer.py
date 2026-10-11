import numpy as np
import pytest
import torch

from rlcontrol.replay_buffer import ReplayBuffer


def add_transitions(buffer, count):
    for i in range(count):
        state = np.full(4, i, dtype=np.float32)
        next_state = state + 1

        buffer.push(
            state=state,
            action=i % 2,
            reward=1.0,
            next_state=next_state,
            terminated=False,
        )


def test_buffer_capacity():
    buffer = ReplayBuffer(capacity=5)
    add_transitions(buffer, 10)

    assert len(buffer) == 5


def test_sample_shapes():
    buffer = ReplayBuffer(capacity=20)
    add_transitions(buffer, 10)

    batch = buffer.sample(batch_size=4)

    assert batch["states"].shape == (4, 4)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["next_states"].shape == (4, 4)
    assert batch["terminated"].shape == (4,)
    assert batch["actions"].dtype == torch.long


def test_sampling_too_many_transitions():
    buffer = ReplayBuffer(capacity=10)
    add_transitions(buffer, 3)

    with pytest.raises(ValueError):
        buffer.sample(batch_size=5)


def test_state_is_copied():
    buffer = ReplayBuffer(capacity=10)

    state = np.zeros(4, dtype=np.float32)

    buffer.push(state, 0, 1.0, state, False)

    state[0] = 99.0

    batch = buffer.sample(batch_size=1)

    assert batch["states"][0, 0].item() == 0.0


def test_sampling_is_reproducible():
    first = ReplayBuffer(capacity=10, seed=123)
    second = ReplayBuffer(capacity=10, seed=123)

    add_transitions(first, 10)
    add_transitions(second, 10)

    first_batch = first.sample(batch_size=5)
    second_batch = second.sample(batch_size=5)

    assert torch.equal(
        first_batch["states"],
        second_batch["states"],
    )

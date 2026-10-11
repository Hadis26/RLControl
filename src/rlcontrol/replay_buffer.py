from collections import deque, namedtuple
import random

import numpy as np
import torch


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "terminated"],
)


class ReplayBuffer:
    """Fixed-capacity experience replay buffer for DQN."""

    def __init__(self, capacity: int, seed: int = 42):
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")

        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        terminated: bool,
    ) -> None:
        transition = Transition(
            state=np.array(state, dtype=np.float32, copy=True),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32, copy=True),
            terminated=bool(terminated),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int, device="cpu") -> dict:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if batch_size > len(self.buffer):
            raise ValueError("Not enough transitions to sample.")

        batch = self.rng.sample(list(self.buffer), batch_size)

        return {
            "states": torch.as_tensor(
                np.stack([t.state for t in batch]),
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.tensor(
                [t.action for t in batch],
                dtype=torch.long,
                device=device,
            ),
            "rewards": torch.tensor(
                [t.reward for t in batch],
                dtype=torch.float32,
                device=device,
            ),
            "next_states": torch.as_tensor(
                np.stack([t.next_state for t in batch]),
                dtype=torch.float32,
                device=device,
            ),
            "terminated": torch.tensor(
                [t.terminated for t in batch],
                dtype=torch.bool,
                device=device,
            ),
        }

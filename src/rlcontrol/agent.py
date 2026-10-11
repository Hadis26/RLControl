import random

import numpy as np
import torch
from torch import nn

from rlcontrol.network import DQN
from rlcontrol.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_frequency: int = 500,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.policy_network = DQN(
            state_dim, action_dim, hidden_dim
        ).to(self.device)

        self.target_network = DQN(
            state_dim, action_dim, hidden_dim
        ).to(self.device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(
            capacity=buffer_capacity,
            seed=seed,
        )

        self.training_steps = 0

    def select_action(self, state, epsilon: float = 0.0) -> int:
        """Select an action using an epsilon-greedy policy."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1.")

        if self.rng.random() < epsilon:
            return self.rng.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self, state, action, reward, next_state, terminated
    ):
        """Store one transition in the replay buffer."""
        self.memory.push(
            state, action, reward, next_state, terminated
        )

    def learn(self) -> float | None:
        """Perform one gradient update using a replay minibatch."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(
            self.batch_size, device=self.device
        )

        # Predicted Q-values for the actions actually taken
        q_values = self.policy_network(
            batch["states"]
        ).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

        # Bellman targets computed without gradient tracking
        with torch.no_grad():
            next_q_values = self.target_network(
                batch["next_states"]
            ).max(dim=1).values

            targets = (
                batch["rewards"]
                + self.gamma
                * (~batch["terminated"]).float()
                * next_q_values
            )

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), max_norm=10.0
        )

        self.optimizer.step()
        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        """Copy policy-network weights into the target network."""
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

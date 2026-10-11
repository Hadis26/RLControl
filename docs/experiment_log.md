# Experiment Log

## EXP-001 — Random Policy Baseline

**Environment:** CartPole-v1

**Policy:** Uniform random action

**Episodes:** 100

**Evaluation seeds:** 42

Results:
- Mean reward: 23.29
- Standard reward: 14.25
- Minimum reward: 8.00
- Maximum reward: 91.00

### Purpose
Establish a reference performance level before
training the DQN agent.

## EXP-002 — Initial DQN Training

### Objective
Train a Deep Q-Network to balance CartPole-v1
and investigate whether training returns improve
with reinforcement learning.

### Method
- Algorithm: DQN
- Network: 4 → 128 → 128 → 2
- Activation: ReLU
- Optimizer: Adam
- Loss: Huber
- Replay capacity: 100,000
- Batch size: 64
- Learning rate: 0.001
- Discount factor: 0.99
- Target update interval: 500 gradient steps
- Exploration: Linear epsilon decay
- Epsilon: 1.0 → 0.05
- Decay steps: 20,000
- Episodes: 600
- Random seed: 42

### Results
To be recorded after execution.

### Observations
To be recorded after inspecting the learning curve.

### Limitations
A single training seed is insufficient to establish
reliable algorithm performance. A separate evaluation
on held-out seeds is required.
## EXP-001 — Random Policy Baseline

**Environment:** CartPole-v1  
**Policy:** Uniform Random  
**Episodes:** 100  
**Seed:** 42 (environment resets)

### Results
- Mean reward: 23.29
- Standard deviation: 14.25
- Minimum reward: 8.00
- Maximum reward: 91.00

### Purpose
Establish a reference performance level before training the DQN agent.

### Observations
The random policy achieved an average episode return of 23.29, indicating that an agent selecting actions without considering the environment state cannot consistently balance the pole.

The standard deviation of 14.25 indicates considerable variation in episode duration.

### Limitations
The environment resets were seeded, but the action space was not explicitly seeded. Consequently, the experiment is not fully reproducible.

A reproducible random-policy evaluation is performed separately in EXP-003.

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

- Training episodes: 600
- Training seed: 42
- Total environment interactions: 33,739
- Mean return (first 20 episodes): 19.00
- Mean return (last 20 episodes): 114.85
- Highest reported 20-episode mean: 149.15 (episode 480)
- Final exploration epsilon: 0.05
- Final model checkpoint: `results/checkpoints/dqn_final.pt`
- Training curve: `results/figures/training_curve.png`

### Observations

The DQN agent demonstrated a clear improvement in training performance over 600 episodes.

During the first 400 episodes, mean returns generally remained between 20 and 40, indicating limited but gradually developing control performance.

A substantial improvement occurred between approximately episodes 420 and 480, during which the moving-average return increased rapidly. At episode 480, the reported 20-episode mean reached 149.15.

After episode 480, training performance fluctuated and eventually declined to a final 20-episode mean of 114.85.

The training results indicate that the agent learned a useful control policy, although its performance remained variable and did not converge to the maximum CartPole-v1 return of 500.

### Limitations

Training was conducted using only one random seed. Therefore, the observed learning behavior may not represent the algorithm's performance across independent runs.

Furthermore, training returns were collected using epsilon-greedy action selection and should not be interpreted as the final greedy policy's evaluation performance.

Independent evaluation is conducted in EXP-003.


## EXP-003 — Trained DQN vs Random Baseline

### Objective
Evaluate the trained DQN policy against a uniformly
random controller on held-out CartPole initial states.

### Evaluation protocol
- Environment: CartPole-v1
- Episodes per policy: 100
- Evaluation seeds: 10000–10099
- Training seed: 42
- DQN exploration epsilon: 0.0
- DQN checkpoint: dqn_final.pt
- Random action space: seeded

### Results

| Metric | Random | DQN |
|---|---:|---:|
| Mean return | 22.52 | 98.63 |
| Standard deviation | 11.41 | 1.88 |
| Minimum return | 9 | 94 |
| Maximum return | 61 | 103 |

### Interpretation

The trained DQN significantly outperformed the random baseline in the descriptive evaluation results.

Across 100 held-out evaluation episodes, the DQN achieved a mean return of 98.63 compared with 22.52 for the random policy, representing approximately a 4.38-fold improvement.

The DQN also demonstrated substantially lower variability in episode returns, with a standard deviation of 1.88 compared with 11.41 for the random policy.

These results indicate that the DQN learned a state-dependent control strategy that consistently outperformed random action selection.

However, evaluation returns were concentrated between 94 and 103, substantially below the maximum possible return of 500. The agent therefore demonstrates useful but limited balancing performance.

### Limitations

The evaluation used one trained DQN checkpoint and 100 held-out environment seeds. Multiple independent training runs are necessary to assess the reliability of the learning algorithm.

The narrow distribution of DQN returns should also be investigated to determine whether the learned policy consistently fails at a similar point in each episode.

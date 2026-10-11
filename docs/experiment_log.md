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


## EXP-004 — DQN Controller Behavior Analysis

### Objective

Investigate why the trained DQN achieves episode returns concentrated around 100 steps despite significantly outperforming a random policy.

### Method

- Environment: CartPole-v1
- Controller: Trained DQN, greedy policy (epsilon = 0)
- Checkpoint: `results/checkpoints/dqn_final.pt`
- Evaluation seeds: 10000, 10025, 10050
- Simulation time step: 0.02 s
- Recorded signals: Cart position, cart velocity, pole angle, pole angular velocity, and control action

### Analysis

The controller will be examined for:

1. Pole-angle instability.
2. Cart-position drift.
3. Oscillatory control decisions.
4. Consistency of termination causes across initial conditions.

### Results

Three deterministic evaluation episodes were analyzed using the trained DQN checkpoint.

| Seed | Episode return | Duration | Termination reason |
|---|---:|---:|---|
| 10000 | 99 | 1.98 s | Cart position boundary |
| 10025 | 96 | 1.92 s | Cart position boundary |
| 10050 | 99 | 1.98 s | Cart position boundary |

### Observations

**Cart position and velocity:** All three trajectories exhibit progressive rightward cart drift. The cart position eventually exceeds the +2.4 m termination boundary. Cart velocity remains predominantly positive during the later stages of each episode.

**Pole angle:** The controller maintains the pole angle within the ±12° termination boundaries. Following an initial increase to approximately +11°, the angle decreases and subsequently remains around +5° to +10°.

**Control actions:** The DQN frequently alternates between left and right actions. While these corrections help constrain the pole angle, they do not sufficiently reduce cart velocity or prevent horizontal drift.

**Consistency:** All three episodes exhibit the same failure mechanism, explaining the narrow distribution of returns observed during EXP-003.

### Conclusion

The trained DQN demonstrates partial stabilization of the CartPole system but fails to regulate the complete system state.

Specifically, the policy prevents pole-angle divergence during the analyzed trajectories while allowing positive cart velocity and horizontal displacement to accumulate.

The resulting cart-position constraint violation terminates episodes after approximately 100 simulation steps.

This behavior suggests that the learned policy is incomplete rather than that the DQN implementation necessarily contains an error.

### Next Steps

1. Preserve the initial trained checkpoint as a reference.
2. Extend training while keeping the algorithm and hyperparameters unchanged.
3. Evaluate the new checkpoint using the same held-out seeds.
4. Compare episode returns and termination mechanisms.
5. Investigate alternative training configurations only if necessary.

### Limitations

Only three selected trajectories were inspected. The findings establish a consistent failure mechanism for these episodes but do not establish that every evaluation episode terminates for the same reason.

## EXP-005 — Extended DQN Training

### Objective

Investigate whether increasing the training duration from 600 to 1,200 episodes improves the DQN controller's performance and stability.

### Hypothesis

Additional environment interactions may allow the agent to develop a more effective control strategy, improving cart-position regulation and extending episode duration.

### Experimental Setup

The DQN architecture, learning rate, discount factor, replay buffer, batch size, target-update frequency, exploration schedule, and random seed were kept unchanged from EXP-002.

Only the number of training episodes was increased from 600 to 1,200.

- Environment: CartPole-v1
- Algorithm: DQN
- Training episodes: 1,200
- Training seed: 42
- Total environment interactions: 136,904
- Evaluation episodes: 100
- Evaluation seeds: 10000–10099
- Evaluation policy: Greedy (epsilon = 0)

### Results

| Metric | EXP-002 (600) | EXP-005 (1200) |
|---|---:|---:|
| Final training mean (20 episodes) | 114.85 | 46.30 |
| Mean evaluation return | 98.63 | 154.16 |
| Evaluation standard deviation | 1.88 | 47.18 |
| Minimum evaluation return | 94 | 117 |
| Maximum evaluation return | 103 | 230 |

The extended-training checkpoint achieved approximately 56.3% higher mean evaluation return than the original 600-episode checkpoint.

### Observations

The training process exhibited substantial instability.

Around episodes 800–840 and 940–960, the agent achieved high returns, including episodes reaching the maximum return of 500. The reported 20-episode moving average reached 457.85 at episode 940.

However, these improvements were not maintained. Performance subsequently declined, with the final 20-episode mean reaching only 46.30 at episode 1200.

Despite this deterioration in training performance, the final greedy policy achieved a higher held-out evaluation return than the original 600-episode policy.

### Conclusion

Extending training improved final greedy-policy evaluation performance but did not produce stable convergence.

The results demonstrate that a longer training run does not necessarily result in consistently better policies.

Periodic checkpointing and independent validation are therefore necessary to identify strong intermediate policies and avoid relying exclusively on the final training checkpoint.

### Limitations

- Only one training seed was used.
- Intermediate checkpoints were not saved.
- The episodes responsible for high training returns were not independently evaluated using greedy action selection.
- The underlying cause of the observed learning instability has not been established.

### Next Steps

Implement periodic checkpoint saving, validation-based model selection, and independent final testing. Preserve the original DQN algorithm to provide a controlled comparison.

## EXP-006 — Validation-Based DQN Checkpoint Selection

### Objective

Investigate whether periodic greedy-policy validation can identify stronger intermediate DQN checkpoints and mitigate the effect of training instability.

### Hypothesis

A model selected by its validation performance may achieve better independent test performance than the final training checkpoint.

### Experimental Setup

- Environment: CartPole-v1
- Algorithm: DQN
- Training episodes: 1,200
- Training seed: 42
- Validation frequency: Every 20 episodes
- Validation episodes per checkpoint: 20
- Validation seeds: 20000–20019
- Final test seeds: 30000–30099
- Model selection criterion: Highest mean validation return
- Saved models: Best validation checkpoint and final checkpoint

### Results

To be completed after training and evaluation.

### Analysis

To be completed after comparing training returns, validation performance, and independent test results.

### Conclusion

To be completed after the experiment.
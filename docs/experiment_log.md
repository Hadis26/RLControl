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

## EXP-006 — Validation-Based Checkpoint Selection

### Objective

Investigate whether periodic validation can identify a stronger intermediate DQN policy than the model obtained at the end of training.

### Experimental Setup

- Algorithm: DQN
- Environment: CartPole-v1
- Training episodes: 1,200
- Training seed: 42
- Validation interval: Every 20 episodes
- Validation episodes: 20
- Validation seeds: 20000–20019
- Independent test episodes: 100
- Independent test seeds: 30000–30099
- Model selection: Highest greedy-policy validation mean

### Results

The best model was selected at training episode **780**, achieving a validation mean return of **500.00**.

Independent test results:

| Metric | Random | Final DQN | Best DQN |
|---|---:|---:|---:|
| Mean return | 22.95 | 158.08 | 500.00 |
| Standard deviation | 11.56 | 48.59 | 0.00 |
| Minimum return | 10 | 118 | 500 |
| Maximum return | 66 | 230 | 500 |

The validation-selected checkpoint achieved the maximum CartPole-v1 return in all 100 independent test episodes.

Its mean return was approximately 3.16 times that of the final DQN checkpoint.

### Observations

The best checkpoint occurred at episode 780, despite subsequent training episodes occasionally achieving very high returns.

This demonstrates that epsilon-greedy training returns are not reliable indicators of greedy-policy performance.

Furthermore, continued optimization after episode 780 produced a final policy with substantially worse independent test performance.

The experiment provides evidence that the DQN training process is susceptible to policy degradation and that periodic validation can identify intermediate policies worth preserving.

### Conclusion

Validation-based checkpoint selection substantially improved the quality of the retained controller in this experiment.

The best checkpoint achieved a mean return of 500.00 with zero observed variability across 100 held-out evaluation episodes.

The results support using periodic independent validation and best-checkpoint preservation rather than relying exclusively on the final training checkpoint.

### Limitations

The experiment involved only one independent training run.

The validation set contained 20 initial conditions, and model selection was based on these conditions. Although the selected policy subsequently achieved maximum performance on 100 unseen test seeds, broader reliability requires evaluation across multiple training seeds and potentially additional environment conditions.

A return of 500 represents the Gymnasium episode limit and does not establish stability beyond that horizon.

## EXP-007 — Multi-Seed Reproducibility Study

### Objective

Evaluate the reliability of the DQN learning algorithm across multiple independent training runs.

### Research Question

Does validation-based checkpoint selection consistently produce successful CartPole controllers across different training seeds?

### Experimental Setup

- Algorithm: DQN
- Environment: CartPole-v1
- Training seeds: 42, 123, 456, 789, 2026
- Training episodes: 1,200 per seed
- Validation frequency: Every 20 episodes
- Validation episodes: 20
- Validation seeds: 20000–20019
- Independent test episodes: 100 per checkpoint
- Test seeds: 30000–30099
- Model selection: Highest validation mean
- Checkpoints: Best and final

### Evaluation Metrics

1. Mean return per trained policy.
2. Standard deviation across evaluation episodes.
3. Success rate (fraction of episodes reaching 500).
4. Variability in mean returns across independent training runs.
5. Training episode at which the best checkpoint was selected.

### Results

Five independent DQN training runs were evaluated using 100 held-out CartPole-v1 episodes per checkpoint.

| Training Seed | Best Mean | Final Mean | Best Success Rate |
|---|---:|---:|---:|
| 42 | 500.00 | 158.08 | 100% |
| 123 | 500.00 | 500.00 | 100% |
| 456 | 455.47 | 124.77 | 82% |
| 789 | 228.37 | 60.32 | 0% |
| 2026 | 500.00 | 88.72 | 100% |

**Aggregate statistics across five independent training runs:**

- Best-checkpoint mean return: 436.77
- Best-checkpoint sample standard deviation: 118.08
- Final-checkpoint mean return: 186.38
- Final-checkpoint sample standard deviation: 179.15
- Mean paired improvement: 250.39 reward points
- Best-checkpoint aggregate success rate: 76.4%
- Final-checkpoint aggregate success rate: 20.0%
- Runs achieving a 100% success rate with best checkpoints: 3/5

Success is defined as reaching the CartPole-v1 maximum episode return of 500.

### Analysis

The results demonstrate considerable sensitivity to the random seed used during DQN training.

Three training runs (seeds 42, 123, and 2026) produced validation-selected controllers that reached the maximum return in all 100 independent test episodes.

Seed 456 produced a partially successful controller with a mean return of 455.47 and an 82% success rate.

Seed 789 produced a weaker controller, with a mean return of 228.37 and no episodes reaching the maximum return.

Validation-selected checkpoints outperformed final checkpoints in four runs and matched the final checkpoint in one run.

The across-run mean return increased from 186.38 for final checkpoints to 436.77 for validation-selected checkpoints.

This corresponds to an approximately 2.34-fold increase in mean evaluated performance.

The performance differences indicate that additional training does not necessarily preserve previously acquired control capabilities. Periodic model evaluation and checkpoint preservation substantially improve the reliability of the retained models under the tested conditions.

### Conclusion

The study provides evidence that the implemented DQN algorithm can learn successful CartPole-v1 control policies across multiple independent training seeds.

However, training outcomes remain variable, and successful convergence is not guaranteed.

Validation-based checkpoint selection is an effective practical method for preserving strong intermediate policies. It improved independent test performance in four of five runs and never performed worse than the final checkpoint in this study.

The results support the inclusion of independent validation and checkpoint management in reinforcement learning experiments.

### Limitations

- Only five independent training seeds were evaluated.
- All runs used the same algorithm configuration.
- Checkpoints were selected using a fixed set of 20 validation initial conditions.
- Independent testing used 100 fixed initial conditions.
- CartPole-v1 is a relatively simple simulated environment.
- The results do not establish robustness under external disturbances, observation noise, model uncertainty, or alternative system dynamics.
- Observed maximum returns are limited by the environment's 500-step episode horizon.

### Future Work

Investigate controller robustness to altered initial conditions and observation disturbances. Compare the original DQN against algorithmic improvements such as Double DQN only if a controlled comparison is warranted.

## EXP-008 — DQN Controller Robustness Analysis

### Objective

Evaluate the sensitivity of the validation-selected DQN controller to observation noise and variations in initial system states.

### Hypothesis

Although the selected DQN achieves the maximum return under nominal CartPole-v1 conditions, its performance may deteriorate under measurement uncertainty or initial conditions outside the standard reset distribution.

### Experimental Setup

**Controller**
- Algorithm: DQN
- Checkpoint: Best model from training seed 42
- Evaluation policy: Greedy (epsilon = 0)
- Environment: CartPole-v1

**Observation Noise**
- Distribution: Independent additive Gaussian noise
- Standard deviations: 0, 0.01, 0.025, 0.05, 0.1
- Episodes per noise level: 100
- Environment seeds: 40000–40099

**Initial-Condition Variation**
- Cart positions: −1.5, −0.75, 0, 0.75, 1.5 m
- Pole angles: −10°, −5°, 0°, 5°, 10°
- Initial velocities: 0
- Number of scenarios: 25

### Evaluation Metrics

- Mean episode return
- Standard deviation of episode returns
- Success rate (return = 500)
- Maximum absolute cart position
- Maximum absolute pole angle

### Results

Pending execution.

### Analysis

To be completed after inspecting the noise-sensitivity plot and initial-condition heatmap.

### Limitations

The observation-noise standard deviation is identical numerically across state components with different physical units. The initial-condition analysis covers a discrete grid, not the full state space. No physical disturbances or changes to the environment dynamics are introduced.
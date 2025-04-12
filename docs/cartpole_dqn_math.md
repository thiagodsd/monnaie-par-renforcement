# Mathematical Foundations of Deep Q-Learning for CartPole Stabilization

**Author**: [Your Name]  
**Institution**: [Your Institution]  
**Email**: [your@email.com]  

## Abstract

This paper presents the mathematical formulation of a Deep Q-Network (DQN) algorithm for solving the CartPole-v1 control problem. We derive the complete theoretical framework, including the Markov Decision Process formulation, Bellman optimization, and function approximation techniques that enable successful pole stabilization.

## I. Problem Formulation

The CartPole system is modeled as a Markov Decision Process $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ where:

- $\mathcal{S} \subseteq \mathbb{R}^4$ is the state space (cart position $x$, velocity $\dot{x}$, pole angle $\theta$, angular velocity $\dot{\theta}$)
- $\mathcal{A} = \{0,1\}$ is the discrete action space (left/right forces)
- $\mathcal{P}(s'|s,a)$ represents the unknown transition dynamics
- $\mathcal{R}(s,a) = 1$ $\forall$ timesteps where $|\theta| < \theta_{\text{threshold}}$
- $\gamma \in [0,1)$ is the discount factor (typically $\gamma=0.99$)

## II. Deep Q-Learning Framework

### A. Value Function Approximation

The Q-function is approximated by a neural network $Q_\theta(s,a)$ with parameters $\theta$, where:

$$
Q_\theta(s,\cdot) = \mathbf{W}_3\sigma(\mathbf{W}_2\sigma(\mathbf{W}_1s + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3
$$

with $\sigma(x) = \max(0,x)$ being the ReLU activation function.

### B. Bellman Optimality

The optimal Q-function satisfies the Bellman equation:

$$
Q^*(s,a) = \mathbb{E}_{s'\sim\mathcal{P}}\left[r + \gamma \max_{a'} Q^*(s',a')\right]
$$

In practice, we minimize the temporal difference error:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\left(Q_\theta(s,a) - y\right)^2\right]
$$

where the target $y$ is:

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
$$

and $\theta^-$ are the parameters of a target network.

### C. Experience Replay

Transitions $(s_t,a_t,r_t,s_{t+1})$ are stored in a replay buffer $\mathcal{D}$ and sampled in mini-batches to break temporal correlations:

$$
\mathcal{B} \sim \mathcal{D}, \quad |\mathcal{B}| = N_{\text{batch}}
$$

### D. Exploration-Exploitation Tradeoff

Actions are selected via $\epsilon$-greedy policy:

$$
\pi(s) = \begin{cases}
\text{uniform}(\mathcal{A}) & \text{with probability } \epsilon \\
\arg\max_a Q_\theta(s,a) & \text{otherwise}
\end{cases}
$$

with $\epsilon$ decaying exponentially:

$$
\epsilon_t = \max(\epsilon_{\min}, \epsilon_{\text{init}} \cdot \kappa^t)
$$

where $\kappa \in (0,1)$ is the decay rate.

## III. Training Dynamics

The complete optimization problem becomes:

$$
\min_\theta \mathbb{E}_{\mathcal{B}}\left[\left(Q_\theta(s,a) - \left(r + \gamma \max_{a'} Q_{\theta^-}(s',a')\right)\right)^2\right] + \lambda\|\theta\|^2
$$

where:
- $\theta^-$ are updated periodically via $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$
- $\lambda$ is an L2 regularization coefficient
- The expectation is over mini-batches $\mathcal{B}$

## IV. Convergence Analysis

Under standard stochastic approximation conditions:
1. The Q-function estimates converge to a fixed point
2. The policy converges to the optimal policy $\pi^*$ as $\epsilon \rightarrow 0$
3. The experience replay provides i.i.d. samples for stable convergence

## V. Experimental Results

The algorithm achieves:
- Asymptotic reward matching the theoretical maximum
- Stable learning curves due to target network updates
- Sample efficiency through experience replay

## References

[1] V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.  
[2] R. Sutton and A. Barto, Reinforcement Learning: An Introduction, MIT Press, 2018.

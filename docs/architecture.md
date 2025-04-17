# Deep Q-Network for Cryptocurrency Trading: Algorithmic Framework

This document describes the Deep Q-Network (DQN) architecture and learning process implemented for cryptocurrency trading. The explanation follows the sequential flow of information and decision-making in the reinforcement learning system.

## 1. State Observation

The agent observes the environment through a structured state representation:

- **Observation window**: The agent receives a fixed-length window of historical data (default: 20 time steps)
- **Feature encoding**: Each time step contains 8 features:
  - OHLC prices (Open, High, Low, Close)
  - Technical indicators (EMA-9, EMA-21, RSI-14, ATR)
- **Normalization**: All features are normalized within the observation window:
  - Price data and EMAs: min-max scaling using the price range in the window
  - RSI: already in [0-100] range, divided by 100 to scale to [0,1]
  - ATR: normalized as percentage of close price

The resulting state tensor has shape `(window_size, 8)` and provides a localized view of recent market behavior.

## 2. Action Selection

Given the current state, the agent selects one of three possible actions:

- **Action space**: Discrete(3) = {HOLD (0), BUY (1), SELL (2)}
- **Decision process**:
  - The agent receives the current observation tensor
  - This tensor is processed by the Q-network to produce Q-values for each action
  - The action with the highest Q-value is selected (when not exploring)

The agent also implements a position cooldown mechanism that enforces a minimum holding period between trades to reduce overtrading.

## 3. Environment Transition

When the agent takes an action, the environment processes it and transitions to a new state:

- **HOLD (0)**: No portfolio change, time advances
- **BUY (1)**: If not holding a position and cash is available:
  - Allocate 40% of available cash
  - Apply transaction fee (0.1%)
  - Record trade details (step, price, amount, value)
  - Set holding_position flag to True
- **SELL (2)**: If holding a position:
  - Sell entire cryptocurrency position
  - Calculate proceeds after transaction fee (0.1%)
  - Update cash balance
  - Reset holding_position flag to False
  - Record trade details

After any action, the environment:
1. Updates portfolio values (cash_balance, asset_value, net_worth)
2. Calculates the reward
3. Advances to the next time step
4. Delivers the next observation

## 4. Experience Storage

Each interaction with the environment generates an experience tuple that is stored in the replay buffer:

- **Experience tuple**: (state, action, reward, next_state, done)
- **Buffer implementation**:
  - Fixed-capacity circular buffer (default: 100,000 transitions)
  - FIFO replacement when capacity is reached
  - Stores experiences as raw tuples without preprocessing

The replay buffer fills gradually during training. The agent begins learning once the buffer reaches a threshold size (10% of capacity) to ensure sufficient diversity in the stored experiences.

## 5. Experience Sampling

During the training process, experiences are sampled from the replay buffer to update the Q-network:

- **Sampling strategy**: Uniform random sampling
- **Batch creation**:
  - Sample `batch_size` (default: 64) experience tuples randomly from buffer
  - Convert tuples to PyTorch tensors
  - Group states, actions, rewards, next_states, and dones into separate batches
  - Move tensors to GPU if available

This random sampling breaks the temporal correlation in the training data, helping to stabilize the learning process.

## 6. Q-Function Training

The Q-function is trained using a temporal difference (TD) learning approach:

- **Loss function**: Mean Squared Error (MSE) between:
  - Current Q-values: Q(s, a) - predicted by the Q-network
  - Target Q-values: r + γ * max_a'[Q(s', a')] - Bellman equation
- **Optimization process**:
  1. Forward pass: Compute current Q-values for taken actions
  2. Compute target Q-values using the Bellman equation
  3. Calculate MSE loss between current and target Q-values
  4. Backpropagate gradients
  5. Apply gradient clipping (max norm: 10.0)
  6. Update network parameters using Adam optimizer

The network architecture is a Multi-Layer Perceptron with the following structure:
- **Input**: Flattened observation window (window_size × 8 features)
- **Hidden layers**:
  - First layer: 2 × hidden_dim nodes, ReLU activation, Layer Normalization, Dropout (20%)
  - Second layer: hidden_dim nodes, ReLU activation, Layer Normalization, Dropout (20%)
  - Third layer: hidden_dim/2 nodes, ReLU activation, Layer Normalization
- **Output**: 3 nodes (one per action) with linear activation

## 7. Q-Value Computation

The Q-values represent the expected future discounted rewards for each action in the current state:

- **Input processing**:
  - State tensor is flattened to 1D (batch_size, window_size × 8)
  - Passed through the MLP network
- **Output interpretation**:
  - Each output node corresponds to an action (HOLD, BUY, SELL)
  - Values indicate expected future returns for each action
  - Higher values indicate more promising actions

The Q-values are used both for action selection and for computing the TD error during training.

## 8. Exploration Strategy

The agent balances exploration and exploitation using an ε-greedy strategy:

- **Initial exploration**: ε = 1.0 (100% random actions)
- **Exploration decay**: ε = max(ε_min, ε × decay_rate)
  - decay_rate = 0.995 (multiplicative decay)
  - ε_min = 0.05 (5% minimum exploration)
- **Action selection process**:
  1. Generate random number r ∈ [0,1]
  2. If r < ε: Select random action (exploration)
  3. If r ≥ ε: Select action with highest Q-value (exploitation)
  4. Apply cooldown constraints regardless of selection method

As training progresses, the agent gradually shifts from exploration to exploitation, focusing more on actions it has learned to be profitable while still maintaining a small probability of exploring to discover potentially better strategies.

## Summary

The DQN trading agent learns to make trading decisions through repeated interaction with the simulated environment. By observing market states, selecting actions, observing outcomes, and adjusting its Q-function, the agent gradually learns to identify profitable trading opportunities while managing risk appropriately. The combination of function approximation (via the neural network), experience replay, and controlled exploration enables the agent to develop complex trading strategies without explicit programming of trading rules.
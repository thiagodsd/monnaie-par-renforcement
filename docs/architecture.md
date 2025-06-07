# Deep Q-Network for Cryptocurrency Trading: Complete Architecture

This document describes the complete Deep Q-Network (DQN) architecture and learning process for cryptocurrency trading.

## 1. State Observation

The agent observes the environment through a structured state representation:

- **Observation window**: Fixed-length window of historical data (default: 21 time steps)
- **Feature encoding**: Each time step contains 12 features:
  - **Price data**: OHLC (Open, High, Low, Close)
  - **Technical indicators**: EMA-9, EMA-21, RSI-14, ATR
  - **Volume**: Trading volume
  - **Market benchmarks**: SP500 daily return, DJIA daily return
  - **Market sentiment**: Fear & Greed Index (FNG)
- **Normalization**: All features are normalized within the observation window:
  - Price data and EMAs: min-max scaling using the price range in the window
  - RSI: divided by 100 to scale to [0,1]
  - ATR: normalized as percentage of close price
  - Volume: min-max scaling within the window
  - Market returns: clipped to [-0.1, 0.1] and scaled to [0,1]
  - FNG: divided by 100 to scale to [0,1]

The resulting state tensor has shape `(window_size, 12)` providing market behavior and context.

## 2. Action Selection

The agent selects one of 9 possible actions:

- **Action space**: Discrete(9) = {
    - HOLD (0)
    - BUY 25% (1), BUY 50% (2), BUY 75% (3), BUY 100% (4)
    - SELL 25% (5), SELL 50% (6), SELL 75% (7), SELL 100% (8)
  }
- **Exploration strategy**: Epsilon-greedy
  - Initial epsilon: 1.0 (100% random actions)
  - Final epsilon: 0.01 (1% random actions)
  - Decay steps: 50,000 steps
  - Linear decay schedule

## 3. Network Architecture

- **Type**: Multi-Layer Perceptron (MLP)
- **Input**: Flattened observation tensor (21 × 12 = 252 features)
- **Hidden layers**: User-configurable (suggested: [512, 256, 128])
- **Activation**: ReLU for hidden layers
- **Output**: 9 Q-values (one per action)
- **Optimizer**: Adam with learning rate 0.001
- **Target network**: Updated every 1,000 steps

## 4. Reward Function

Simple portfolio-based reward encouraging profitable trading:

```
reward = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value

# Optional enhancements:
# - Subtract small penalty for transaction costs
# - Add small penalty for excessive trading (action != HOLD)
```

**Rationale**: Directly optimizes for portfolio growth while implicitly penalizing losses.

## 5. Environment Transition

When the agent takes an action:

**HOLD (0)**: No portfolio change, time advances

**BUY (1-4)**: Based on available cash:
- Calculate maximum buyable amount with available cash
- Apply percentage based on action (25%, 50%, 75%, 100%)
- Deduct transaction fee (2%)
- Update portfolio

**SELL (5-8)**: Based on current holdings:
- Calculate amount to sell based on current position
- Apply percentage based on action (25%, 50%, 75%, 100%)
- Apply transaction fee (5%)
- Update cash balance

After any action:
1. Update portfolio values (cash, crypto holdings, total value)
2. Calculate reward
3. Advance to next time step
4. Return next observation

## 6. Training Configuration

**Experience Replay**:
- Buffer size: 100,000 transitions
- Batch size: 32
- Minimum buffer size before training: 10,000

**Training Schedule**:
- Train every 4 steps
- Target network update: every 1,000 steps
- Total training episodes: 1,000

**Data Split**:
- Training: First 70% of time series data
- Validation: Last 30% of time series data (out-of-time validation)

## 7. Performance Evaluation

**Primary Metric**: Portfolio balance growth
- Final portfolio value vs. initial value
- Percentage return over evaluation period

**Secondary Metrics**:
- Total number of trades executed
- Win rate (percentage of profitable trades)
- Maximum drawdown during evaluation
- Comparison to buy-and-hold strategy

**Validation Process**:
1. Train agent on training data (first 70%)
2. Freeze trained model (epsilon = 0)
3. Evaluate on validation data (last 30%)
4. Compare final portfolio value to initial value
5. Report percentage return and trade statistics

## 8. Implementation Summary

The DQN agent learns cryptocurrency trading through:

1. **Observation**: 21-step windows with 12 market features
2. **Decision**: 9-action space with position sizing
3. **Learning**: Experience replay with epsilon-greedy exploration
4. **Evaluation**: Out-of-time validation measuring portfolio growth

The system is designed for simplicity while maintaining effectiveness, focusing on direct portfolio optimization without complex risk management constraints.
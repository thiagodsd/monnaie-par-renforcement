import random
import os
from typing import List, Tuple
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
import pandas as pd
from pathlib import Path
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import ta


class PriceActionQNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten to (batch_size, features)
        # forward pass through the network
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = list()
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_np = np.array(states) # converting to numpy arrays first for efficiency
        next_states_np = np.array(next_states)
        return (
            torch.FloatTensor(states_np),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states_np),
            torch.FloatTensor(dones),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def ready(self) -> bool:
        return len(self.buffer) >= self.capacity // 10


class PositionDQNAgent:
    def __init__(self, env: gym.Env, hyperparams: dict):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dim = hyperparams["hidden_dim"]
        #
        self.q_network = PriceActionQNetwork(state_dim, hidden_dim, action_dim)
        self.target_network = PriceActionQNetwork(state_dim, hidden_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        #
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=hyperparams["lr"]
        )
        self.buffer = ReplayBuffer(hyperparams["buffer_size"])
        self.hyperparams = hyperparams
        self.criterion = nn.MSELoss()
        #
        self.epsilon_start = hyperparams["epsilon_start"]
        self.epsilon_end = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        #
        # Position tracking
        self.position_cooldown = 0
        self.min_hold_period = hyperparams.get("min_hold_period", 5)
        #
        self.video_dir = "../data/videos"
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def act(self, state, exploration: bool = True) -> int:
        # Implement cooldown to prevent frequent trading
        if self.position_cooldown > 0:
            self.position_cooldown -= 1
            # If in cooldown period, only allow HOLD action
            return 0  # HOLD
            
        if exploration and random.random() < self.epsilon:
            action = self.env.action_space.sample()
            # If randomly selected BUY or SELL, set cooldown
            if action > 0:  # action is BUY or SELL
                self.position_cooldown = self.min_hold_period
            return action
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            
            # If selected BUY or SELL, set cooldown
            if action > 0:  # action is BUY or SELL
                self.position_cooldown = self.min_hold_period
                
            return action

    def update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.hyperparams["gamma"] * max_next_q_values
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes: int, eval_every: int = 20):
        portfolio_values = list()
        eval_portfolios = list()
        loss_history = list()
        reward_history = list()
        eval_rewards = list()
        self.train_trades = list()
        for episode in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result # new gym api
            else:
                state = reset_result # old gym api
            done = False
            episode_profit = 0
            self.position_cooldown = 0  # Reset cooldown at the start of each episode
            
            while not done:
                action = self.act(state)
                step_result = self.env.step(action)
                if len(step_result) == 5:  # new gym api
                    next_state, reward, done, truncated, _ = step_result
                    done = done or truncated
                else:  # old gym api
                    next_state, reward, done, _ = step_result
                self.total_steps += 1
                episode_profit += reward
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                if self.buffer.ready():
                    batch = self.buffer.sample(self.hyperparams["batch_size"])
                    loss = self.update(batch)
                    loss_history.append(loss)
                    self.update_epsilon()
                    reward_history.append(np.mean(batch[2].numpy()))
                if self.total_steps % self.hyperparams["target_update"] == 0:
                    self.update_target_network()
            portfolio_values.append(self.env.net_worth)
            if episode == num_episodes - 1:
                self.train_trades = self.env.trades.copy()
            if (episode + 1) % eval_every == 0:
                eval_portfolio, eval_reward = self.evaluate()
                eval_portfolios.append(eval_portfolio)
                eval_rewards.append(eval_reward)
                print(f"\nEvaluation after {episode+1} episodes:")
                print(f"  Avg Portfolio: ${np.mean(eval_portfolio):.2f}")
                print(f"  Avg Reward: {np.mean(eval_reward):.2f}\n")
            if episode % 10 == 0:
                avg_profit = np.mean(portfolio_values[-10:])
                print(f"Episode: {episode}, "
                      f"Portfolio: ${self.env.net_worth:.2f}, "
                      f"Avg Profit: ${avg_profit:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        self.env.close()
        self.save_training_plots(loss_history, reward_history, portfolio_values)
        return portfolio_values, loss_history, reward_history, eval_portfolios, eval_rewards

    def evaluate(self, num_episodes: int = 10):
        """Evaluate agent performance without exploration"""
        original_epsilon = self.epsilon
        original_cooldown = self.position_cooldown
        self.epsilon = 0  # disable exploration
        self.position_cooldown = 0  # reset cooldown
        eval_portfolios = list()
        eval_rewards = list()
        for _ in range(num_episodes):
            reset_result = self.env.reset()
            
            # Handle both gym API versions
            if isinstance(reset_result, tuple):
                state, _ = reset_result  # New Gym API
            else:
                state = reset_result  # Old Gym API
                
            done = False
            episode_rewards = []
            self.position_cooldown = 0  # Reset cooldown
            
            while not done:
                action = self.act(state, exploration=False)
                step_result = self.env.step(action)
                
                # Handle both gym API versions
                if len(step_result) == 5:  # New Gym API
                    next_state, reward, done, truncated, _ = step_result
                    done = done or truncated
                else:  # Old Gym API
                    next_state, reward, done, _ = step_result
                    
                state = next_state
                episode_rewards.append(reward)
            
            eval_portfolios.append(self.env.net_worth)
            eval_rewards.append(np.mean(episode_rewards))
        
        self.epsilon = original_epsilon  # Restore original epsilon
        self.position_cooldown = original_cooldown  # Restore original cooldown
        return eval_portfolios, eval_rewards
    
    def save_training_plots(self, loss_history, reward_history, portfolio_values):
        """Save training metrics plots"""
        # Create output directory if it doesn't exist
        plots_dir = Path("../data/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('DQN Loss History')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.savefig(plots_dir / 'loss_history.png')
        plt.close()
        
        # Plot reward history
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history)
        plt.title('Average Reward History')
        plt.xlabel('Update Steps')
        plt.ylabel('Average Reward')
        plt.savefig(plots_dir / 'reward_history.png')
        plt.close()
        
        # Plot portfolio value history
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values)
        plt.title('Portfolio Value History')
        plt.xlabel('Episodes')
        plt.ylabel('Portfolio Value ($)')
        plt.savefig(plots_dir / 'portfolio_history.png')
        plt.close()


class PriceActionTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000, window_size=20):
        super().__init__()
        self.df = self.add_technical_indicators(df)
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        
        # Action space: [HOLD, BUY, SELL]
        self.action_space = gym.spaces.Discrete(3)
        
        # Number of features in observation space
        num_features = 12  # price, volume, fng + technical indicators
        
        # Observation space normalized
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_features,),  # Flattened feature vector
            dtype=np.float32
        )
        
        # Trading history for visualization
        self.trades = []
        self.position_size = 0.4  # Use 40% of balance per position
        self.trade_fee = 0.001    # 0.1% trading fee
        self.holding_position = False  # Track if we're in a position
        
        self.reset()
    
    def add_technical_indicators(self, df):
        """Add technical indicators for price action analysis"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Add RSI (Relative Strength Index)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Add MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_pct'] = bollinger.bollinger_pband()  # Percentile within bands
        
        # Add ATR (Average True Range) for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Fill NaN values that may have been created
        df = df.fillna(method='bfill').fillna(0)
        
        return df
        
    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance  # Initialize net_worth
        self.current_step = self.window_size
        self.trades = []  # Reset trade history
        self.holding_position = False
        self.last_action_step = 0
        
        # Handle both old and new Gym API
        try:
            return self._next_observation(), {}  # New Gym API (obs, info)
        except Exception:
            return self._next_observation()  # Old Gym API (just obs)
        
    def _next_observation(self):
        """Get price action features for current state"""
        # Get current market data
        current_data = self.df.iloc[self.current_step]
        
        # Create feature vector
        features = np.array([
            # Price information (normalized)
            current_data['close'] / current_data['open'] - 1,  # Price change
            
            # Volume (normalized by div by max)
            current_data['volume'] / self.df['volume'].max(),
            
            # Fear and Greed Index (normalized)
            current_data['fng_value'] / 100 if 'fng_value' in current_data else 0.5,
            
            # Technical indicators
            current_data['rsi'] / 100,  # RSI (0-100 normalized to 0-1)
            current_data['macd'] / current_data['close'] * 100,  # MACD as % of price
            current_data['macd_signal'] / current_data['close'] * 100,  # Signal as % of price
            current_data['macd_diff'] / current_data['close'] * 100,  # Diff as % of price
            
            # Bollinger Bands
            current_data['bb_pct'],  # Already normalized (0-1)
            (current_data['close'] - current_data['bb_low']) / (current_data['bb_high'] - current_data['bb_low'] + 1e-8),
            
            # ATR (volatility) - normalize by price
            current_data['atr'] / current_data['close'] * 100,
            
            # Position information
            1.0 if self.holding_position else 0.0,  # Binary indicator if in position
            self.btc_held * current_data['close'] / self.initial_balance  # Position size relative to initial balance
        ], dtype=np.float32)
        
        return features
        
    def step(self, action):
        self.current_step += 1
        
        current_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Apply position trading logic with fees and larger position sizes
        # Only execute trades if not holding and action is BUY, or if holding and action is SELL
        steps_since_last_action = self.current_step - self.last_action_step
                
        if action == 1 and not self.holding_position and self.balance > 0:  # BUY
            position_size = self.balance * self.position_size  # Use percentage of balance
            cost = position_size * (1 - self.trade_fee)
            btc_bought = cost / current_price
            self.btc_held += btc_bought
            self.balance -= position_size
            self.holding_position = True
            self.last_action_step = self.current_step
            
            # Record buy trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'buy',
                'amount': btc_bought,
                'value': position_size
            })
            
        elif action == 2 and self.holding_position and self.btc_held > 0:  # SELL
            btc_sold = self.btc_held  # Sell entire position
            proceeds = btc_sold * current_price * (1 - self.trade_fee)
            self.balance += proceeds
            self.btc_held = 0
            self.holding_position = False
            self.last_action_step = self.current_step
            
            # Record sell trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'sell',
                'amount': btc_sold,
                'value': proceeds
            })
        
        # Calculate portfolio value
        self.net_worth = self.balance + self.btc_held * current_price
        
        # Enhanced reward function for position trading
        # Stronger rewards for holding winning positions and cutting losing ones
        if prev_net_worth == 0:
            reward = 0
        else:
            returns = (self.net_worth - prev_net_worth) / prev_net_worth
            
            # Scale returns into a reward
            reward = returns * 100  # Base reward is percent return
            
            # Penalize frequent trading
            if action > 0 and steps_since_last_action < 5:  # If trade and last trade was recent
                reward -= 0.5  # Penalty for frequent trading
                
            # Reward for successful position trades
            if action == 2 and returns > 0:  # Selling at a profit
                reward *= 1.5  # Bonus for profitable exits
                
            # Reduce excessive trading frequency by small penalty for any action
            if action > 0:
                reward -= 0.1  # Small cost for any trade
        
        # Check if done
        done = self.net_worth <= 0 or self.current_step >= len(self.df)-1
        
        # Gym API can be inconsistent, so handle both old and new versions
        try:
            return self._next_observation(), reward, done, False, {}  # New Gym API (done, truncated, info)
        except Exception:
            return self._next_observation(), reward, done, {}  # Old Gym API (done, info)

def load_data():
    """Load and preprocess market data"""
    df = pd.read_parquet("../data/04_feature/analytical_base_table_01.parquet")
    print(f"Loaded data with shape {df.shape}")
    
    # Sort by date and reset index
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split into training and validation sets (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"Split data into training ({train_df.shape}) and validation ({val_df.shape})")
    
    return train_df, val_df


def plot_trading_results(train_df, val_df, train_trades, val_trades, train_profit, val_profit):
    """Plot training and validation results side by side with buy/sell indicators"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # Plot training data and trades
    ax1.plot(train_df['close'].values, color='blue', alpha=0.6)
    ax1.set_title(f'Training Set - Profit: ${train_profit:.2f}')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Price')
    
    # Plot buy/sell points
    for trade in train_trades:
        if trade['type'] == 'buy':
            ax1.scatter(trade['step'] - train_df.index[0], trade['price'], 
                       marker='^', color='green', s=100, alpha=0.7)
        else:  # sell
            ax1.scatter(trade['step'] - train_df.index[0], trade['price'], 
                       marker='v', color='red', s=100, alpha=0.7)
    
    # Plot validation data and trades
    ax2.plot(val_df['close'].values, color='purple', alpha=0.6)
    ax2.set_title(f'Out-of-Time Validation - Profit: ${val_profit:.2f}')
    ax2.set_xlabel('Time Steps')
    
    # Plot buy/sell points
    for trade in val_trades:
        if trade['type'] == 'buy':
            ax2.scatter(trade['step'] - val_df.index[0], trade['price'], 
                       marker='^', color='green', s=100, alpha=0.7)
        else:  # sell
            ax2.scatter(trade['step'] - val_df.index[0], trade['price'], 
                       marker='v', color='red', s=100, alpha=0.7)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Training Price'),
        Line2D([0], [0], color='purple', lw=2, label='Validation Price'),
        Line2D([0], [0], marker='^', color='green', markersize=10, linestyle='None', label='Buy'),
        Line2D([0], [0], marker='v', color='red', markersize=10, linestyle='None', label='Sell')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    plots_dir = Path("../data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / 'trading_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_out_of_time_validation(agent, val_df, window_size, initial_balance=1000):
    """Run the trained agent on out-of-time validation data"""
    val_env = PriceActionTradingEnv(val_df, initial_balance=initial_balance, window_size=window_size)
    reset_result = val_env.reset()
    
    # Handle both gym API versions
    if isinstance(reset_result, tuple):
        state, _ = reset_result  # New Gym API
    else:
        state = reset_result  # Old Gym API
        
    done = False
    
    print("\nRunning out-of-time validation...")
    while not done:
        # Always use exploitation (no exploration)
        action = agent.act(state, exploration=False)
        step_result = val_env.step(action)
        
        # Handle both gym API versions
        if len(step_result) == 5:  # New Gym API
            next_state, reward, done, truncated, _ = step_result
            done = done or truncated
        else:  # Old Gym API
            next_state, reward, done, _ = step_result
            
        state = next_state
    
    # Calculate final profit
    profit = val_env.net_worth - val_env.initial_balance
    roi = (profit / val_env.initial_balance) * 100
    
    print(f"Validation Results:")
    print(f"  Final Portfolio: ${val_env.net_worth:.2f}")
    print(f"  Profit: ${profit:.2f} (ROI: {roi:.2f}%)")
    print(f"  Total Trades: {len(val_env.trades)}")
    
    return val_env.net_worth, val_env.trades

def main():
    # Load and prepare data
    train_df, val_df = load_data()
    
    # Create price action trading environment with training data
    window_size = 20  # Larger window for better trend analysis
    initial_balance = 1000  # Starting with 1000 USD
    env = PriceActionTradingEnv(train_df, initial_balance=initial_balance, window_size=window_size)
    print(f"Environment created successfully! Action space: {env.action_space}, Initial balance: ${initial_balance}")
    
    hyperparams = {
        "hidden_dim": 256,     # Deep network for complex patterns
        "lr": 0.00025,         # Slightly higher learning rate
        "buffer_size": 100000,
        "batch_size": 64,      # Smaller batches for more frequent updates
        "gamma": 0.99,         # Longer term horizon for position trading
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,   # Maintain some exploration
        "epsilon_decay": 0.995,  # Slower epsilon decay
        "target_update": 500,  # Less frequent target updates
        "min_hold_period": 10  # Minimum periods to hold a position
    }
    
    # Train the agent
    agent = PositionDQNAgent(env, hyperparams)
    portfolio_values, loss_history, reward_history, _, _ = agent.train(100)  # More episodes for better learning
    
    # Calculate training profit
    train_profit = env.net_worth - env.initial_balance
    train_roi = (train_profit / env.initial_balance) * 100
    print(f"\nTraining complete!")
    print(f"  Final portfolio: ${env.net_worth:.2f}")
    print(f"  Profit: ${train_profit:.2f} (ROI: {train_roi:.2f}%)")
    print(f"  Total trades: {len(agent.train_trades)}")
    
    # Run out-of-time validation using the same initial balance
    val_portfolio, val_trades = run_out_of_time_validation(agent, val_df, window_size, initial_balance)
    val_profit = val_portfolio - initial_balance
    
    # Plot training and validation results side by side
    plot_trading_results(train_df, val_df, agent.train_trades, val_trades, train_profit, val_profit)


if __name__ == "__main__":
    main()
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


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim * 3, hidden_dim)  # input is window_size x 3 features
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten to (batch_size, window_size*3)
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


class DQNAgent:
    def __init__(self, env: gym.Env, hyperparams: dict):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dim = hyperparams["hidden_dim"]
        #
        self.q_network = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_network = QNetwork(state_dim, hidden_dim, action_dim)
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
        self.video_dir = "../data/videos"
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def act(self, state, exploration: bool = True) -> int:
        if exploration and random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

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
        self.epsilon = 0 # disable exploration
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


class BitcoinTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000, window_size=10):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        
        # Action space: [HOLD, BUY, SELL]
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space normalized (close price, volume, fear & greed)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(window_size, 3),  # [close_pct, volume_pct, fng_normalized]
            dtype=np.float32
        )
        
        # Trading history for visualization
        self.trades = []
        
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance  # Initialize net_worth
        self.current_step = self.window_size
        self.trades = []  # Reset trade history
        
        # Handle both old and new Gym API
        try:
            return self._next_observation(), {}  # New Gym API (obs, info)
        except Exception:
            return self._next_observation()  # Old Gym API (just obs)
        
    def _next_observation(self):
        """Get window of market observations"""
        # Normalize features within the observation window
        window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        
        # Ensure window is not empty
        if len(window) < self.window_size:
            # Fill with the last rows repeated if needed
            padding = self.window_size - len(window)
            last_row = window.iloc[-1:].copy()
            padding_rows = pd.concat([last_row] * padding, ignore_index=True)
            window = pd.concat([window, padding_rows], ignore_index=True)
        
        # Normalize price and volume using window-specific min/max
        close_pct = (window['close'] - window['close'].min()) / (window['close'].max() - window['close'].min() + 1e-8)
        
        # Some datasets might use different volume column names
        if 'volume' in window.columns:
            volume_col = 'volume'
        elif 'Volume' in window.columns:
            volume_col = 'Volume'
        else:
            # Create synthetic volume if not available
            window['volume'] = np.random.randint(1e6, 1e7, len(window))
            volume_col = 'volume'
            
        volume_pct = (window[volume_col] - window[volume_col].min()) / (window[volume_col].max() - window[volume_col].min() + 1e-8)
        
        # Handle missing FNG data
        if 'fng_value' in window.columns:
            fng_normalized = window['fng_value'] / 100  # FNG is already 0-100
        else:
            # Use a synthetic sentiment indicator if not available
            fng_normalized = np.array([0.5] * len(window))
        
        obs = np.stack([close_pct.values, volume_pct.values, fng_normalized], axis=1)
        return obs
        
    def step(self, action):
        self.current_step += 1
        
        current_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Execute trade with 0.1% fee and position limits
        fee = 0.001  # 0.1% trading fee
        
        if action == 1 and self.balance > 0:  # Buy
            max_position_size = self.balance * 0.25  # Max 25% of balance per trade
            cost = min(max_position_size, self.balance) * (1 - fee)
            self.btc_held += cost / current_price
            self.balance -= cost
            # Record buy trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'buy',
                'amount': cost / current_price,
                'value': cost
            })
            
        elif action == 2 and self.btc_held > 0:  # Sell
            sell_amount = self.btc_held * 0.25  # Sell 25% of position
            proceeds = sell_amount * current_price * (1 - fee)
            self.balance += proceeds
            self.btc_held -= sell_amount
            # Record sell trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'sell',
                'amount': sell_amount,
                'value': proceeds
            })
            
        # Calculate percentage-based reward
        self.net_worth = self.balance + self.btc_held * current_price
        # Calculate reward based on portfolio returns only (without inactivity penalty)
        if prev_net_worth == 0:
            reward = 0
        else:
            returns = (self.net_worth - prev_net_worth) / prev_net_worth
            reward = returns * 100  # Scale to percentage points
        
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
    
    # Split into training and validation sets (70/30)
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
    val_env = BitcoinTradingEnv(val_df, initial_balance=initial_balance, window_size=window_size)
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
    
    # Create trading environment with training data
    window_size = 10
    initial_balance = 1000  # Starting with 1000 USD
    env = BitcoinTradingEnv(train_df, initial_balance=initial_balance, window_size=window_size)
    print(f"Environment created successfully! Action space: {env.action_space}, Initial balance: ${initial_balance}")
    
    hyperparams = {
        "hidden_dim": 256,     # Deeper network for price patterns
        "lr": 0.0001,         # More stable learning rate
        "buffer_size": 100000,
        "batch_size": 128,    # Larger batches for smoother updates
        "gamma": 0.95,        # Slightly shorter horizon
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,  # Maintain some exploration
        "epsilon_decay": 0.995,  # Slower epsilon decay
        "target_update": 200   # More stable target network
    }
    
    # Train the agent
    agent = DQNAgent(env, hyperparams)
    portfolio_values, loss_history, reward_history, _, _ = agent.train(50)
    
    # Calculate training profit
    train_profit = env.net_worth - env.initial_balance
    train_roi = (train_profit / env.initial_balance) * 100
    print(f"\nTraining complete!")
    print(f"  Final portfolio: ${env.net_worth:.2f}")
    print(f"  Profit: ${train_profit:.2f} (ROI: {train_roi:.2f}%)")
    
    # Run out-of-time validation using the same initial balance
    val_portfolio, val_trades = run_out_of_time_validation(agent, val_df, window_size, initial_balance)
    val_profit = val_portfolio - initial_balance
    
    # Plot training and validation results side by side
    plot_trading_results(train_df, val_df, agent.train_trades, val_trades, train_profit, val_profit)


if __name__ == "__main__":
    main()

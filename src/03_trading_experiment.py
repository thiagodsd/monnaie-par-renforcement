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
        self.fc1 = nn.Linear(state_dim * 3, hidden_dim)  # Input is window_size x 3 features
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the windowed data
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, window_size*3)
        
        # Process through FC layers
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
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
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
        
        # Q networks
        self.q_network = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_network = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=hyperparams["lr"]
        )
        self.buffer = ReplayBuffer(hyperparams["buffer_size"])
        self.hyperparams = hyperparams
        self.criterion = nn.MSELoss()
        
        # Epsilon for exploration
        self.epsilon_start = hyperparams["epsilon_start"]
        self.epsilon_end = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        
        # Create output directory for videos
        self.video_dir = "../data/videos"
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def act(self, state, exploration: bool = True) -> int:
        if exploration and random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        
        # Exploit: best action according to Q-values
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
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.hyperparams["gamma"] * max_next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes: int, eval_every: int = 20):
        portfolio_values = []
        eval_portfolios = []
        loss_history = []
        reward_history = []
        eval_rewards = []
        
        for episode in range(num_episodes):
            # Training phase
            state = test_env.reset() if test_env else self.env.reset()
            env_to_use = test_env if test_env else self.env
            done = False
            episode_profit = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env_to_use.step(action)
                self.total_steps += 1
                episode_profit += reward
                
                # Store transition in replay buffer
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                # Update networks if we have enough samples
                if self.buffer.ready():
                    batch = self.buffer.sample(self.hyperparams["batch_size"])
                    loss = self.update(batch)
                    loss_history.append(loss)
                    self.update_epsilon()
                    
                    # Track average reward for this batch
                    reward_history.append(np.mean(batch[2].numpy()))
                
                # Update target network periodically
                if self.total_steps % self.hyperparams["target_update"] == 0:
                    self.update_target_network()
            
            portfolio_values.append(self.env.net_worth)
            
            # Evaluation phase
            if (episode + 1) % eval_every == 0:
                eval_portfolio, eval_reward = self.evaluate()
                eval_portfolios.append(eval_portfolio)
                eval_rewards.append(eval_reward)
                print(f"\nEvaluation after {episode+1} episodes:")
                print(f"  Avg Portfolio: ${np.mean(eval_portfolio):.2f}")
                print(f"  Avg Reward: {np.mean(eval_reward):.2f}\n")
            
            # Print progress
            if episode % 10 == 0:
                avg_profit = np.mean(portfolio_values[-10:])
                print(f"Episode: {episode}, "
                      f"Portfolio: ${self.env.net_worth:.2f}, "
                      f"Avg Profit: ${avg_profit:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        self.env.close()
        
        # Save training metrics plots
        self.save_training_plots(loss_history, reward_history, portfolio_values)
        
        return portfolio_values, loss_history, reward_history, eval_portfolios, eval_rewards

    def evaluate(self, num_episodes: int = 10, test_env=None):
        """Evaluate agent performance without exploration"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # Disable exploration
        
        eval_portfolios = []
        eval_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                action = self.act(state, exploration=False)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_rewards.append(reward)
            
            eval_portfolios.append(env_to_use.net_worth)
            eval_rewards.append(np.mean(episode_rewards))
        
        self.epsilon = original_epsilon  # Restore original epsilon
        return eval_portfolios, eval_rewards

    def save_training_plots(self, loss_history, reward_history, portfolio_values, eval_portfolios=None, eval_rewards=None):
        """Save plots of training metrics to file"""
        plots_dir = "../data/plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Create figure with 2 rows of 3 columns
        plt.figure(figsize=(25, 10))

        # Loss plot
        plt.subplot(2, 3, 1)
        plt.plot(loss_history)
        plt.title('Training Loss History')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True)

        # Reward plot
        plt.subplot(2, 3, 2)
        plt.plot(reward_history)
        plt.title('Training Reward History')
        plt.xlabel('Update Step')
        plt.ylabel('Average Reward')
        plt.grid(True)

        # Portfolio value plot
        plt.subplot(2, 3, 3)
        plt.plot(portfolio_values)
        plt.title('Training Portfolio Value History')
        plt.xlabel('Episode')
        plt.ylabel('USD')
        plt.grid(True)

        # Evaluation portfolio plot
        plt.subplot(2, 3, 4)
        if eval_portfolios:
            plt.plot(eval_portfolios)
            plt.title('Evaluation Portfolio Value')
            plt.xlabel('Evaluation Epoch')
            plt.ylabel('USD')
            plt.grid(True)

        # Evaluation reward plot
        plt.subplot(2, 3, 5)
        if eval_rewards:
            plt.plot(eval_rewards)
            plt.title('Average Evaluation Reward')
            plt.xlabel('Evaluation Epoch')
            plt.ylabel('Reward')
            plt.grid(True)

        # Hide empty subplot
        plt.subplot(2, 3, 6).axis('off')

        # Save and close
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'training_metrics.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"\nSaved training metrics plots to {plot_path}")


    def record_video(self, filename="cartpole_solution.gif", num_episodes=1):
        """Record a video of the agent solving the environment"""
        # Create a new environment with render mode for recording
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        
        frames = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            
            print(f"Recording episode {episode+1}/{num_episodes}...")
            while not (done or truncated):
                # Use greedy policy (no exploration)
                action = self.act(state, exploration=False)
                state, _, done, truncated, _ = env.step(action)
                
                # Render and capture frame
                frame = env.render()
                frames.append(Image.fromarray(frame))
        
        env.close()
        
        # Save as GIF
        output_path = os.path.join(self.video_dir, filename)
        print(f"Saving video to {output_path}")
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Video saved successfully!")
        return output_path


class BitcoinTradingEnv(gym.Env):
    def __init__(self, df_train, df_test=None, initial_balance=10000, window_size=10, mode='train'):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.mode = mode
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
        
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance
        self.df = self.df_train if self.mode == 'train' else self.df_test
        self.current_step = self.window_size
        return self._next_observation()
        
    def _next_observation(self):
        """Get window of market observations"""
        # Normalize features within the observation window
        window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        
        # Normalize price and volume using window-specific min/max
        close_pct = (window['close'] - window['close'].min()) / (window['close'].max() - window['close'].min() + 1e-8)
        volume_pct = (window['volume'] - window['volume'].min()) / (window['volume'].max() - window['volume'].min() + 1e-8)
        fng_normalized = window['fng_value'] / 100  # FNG is already 0-100
        
        obs = np.stack([close_pct.values, volume_pct.values, fng_normalized.values], axis=1)
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
            
        elif action == 2 and self.btc_held > 0:  # Sell
            sell_amount = self.btc_held * 0.25  # Sell 25% of position
            proceeds = sell_amount * current_price * (1 - fee)
            self.balance += proceeds
            self.btc_held -= sell_amount
            
        # Calculate percentage-based reward
        self.net_worth = self.balance + self.btc_held * current_price
        # Calculate reward with penalty for inactivity
        if prev_net_worth == 0:
            reward = 0
        else:
            returns = (self.net_worth - prev_net_worth) / prev_net_worth
            reward = returns * 100  # Scale to percentage points
            
            # Penalize holding position during market moves
            price_change = (current_price - self.df.iloc[self.current_step-1]['close']) / self.df.iloc[self.current_step-1]['close']
            if abs(price_change) > 0.03 and action == 0:  # >3% move and did nothing
                reward -= abs(price_change) * 10
        
        # Check if done
        done = self.net_worth <= 0 or self.current_step >= len(self.df)-1
        
        return self._next_observation(), reward, done, {}

def load_data():
    """Load and preprocess market data"""
    try:
        # Try loading real data
        df = pd.read_parquet("../data/04_feature/analytical_base_table_01.parquet")
        # Split data into train/test (70/30) by time
        split_idx = int(len(df) * 0.7)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_test = df.iloc[split_idx:].reset_index(drop=True)
    except FileNotFoundError:
        # Generate synthetic data if real data not available
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        df_train = pd.DataFrame({
        'date': dates,
            'close': np.exp(np.cumsum(np.random.normal(0.001, 0.02, 500))) * 10000,
            'volume': np.random.randint(1e6, 1e7, 500),
            'fng_value': np.random.randint(0, 100, 500)
        })
        # Create test data with different random seed
        df_test = pd.DataFrame({
            'close': np.exp(np.cumsum(np.random.normal(0.001, 0.02, 500))) * 10000,
            'volume': np.random.randint(1e6, 1e7, 500),
            'fng_value': np.random.randint(0, 100, 500)
        })
        print("Using synthetic data")
    
    # Add features to both datasets
    for df in [df_train, df_test]:
        df['price_change_pct'] = df['close'].pct_change()
        df['volatility'] = df['price_change_pct'].rolling(window=5).std()
        df.dropna(inplace=True)
    
    print(f"Training data: {len(df_train)} periods, Test data: {len(df_test)} periods")
    
    return df_train, df_test

def main():
    # Load and prepare data
    df = load_data()
    
    # Create training and test environments
    window_size = 10
    df_train, df_test = load_data()
    env = BitcoinTradingEnv(df_train, df_test, window_size=window_size, mode='train')
    test_env = BitcoinTradingEnv(df_train, df_test, window_size=window_size, mode='test')
    print("Environment created successfully! Action space:", env.action_space)
    
    hyperparams = {
        "hidden_dim": 256,    # Deeper network for price patterns
        "lr": 0.0001,         # More stable learning rate
        "buffer_size": 100000,
        "batch_size": 128,    # Larger batches for smoother updates
        "gamma": 0.95,       # Slightly shorter horizon
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,  # Maintain some exploration
        "epsilon_decay": 0.995,  # Slower epsilon decay
        "target_update": 200    # More stable target network
    }
    
    agent = DQNAgent(env, hyperparams)
    portfolio_values, loss_history, reward_history, _, _ = agent.train(50)
    
    # Final evaluation on test set
    test_portfolios, test_rewards = agent.evaluate(num_episodes=20, test_env=test_env)
    
    # Print final statistics
    print(f"\n{'='*40}\nFinal Results:")
    print(f"Training Avg Portfolio: ${np.mean(portfolio_values[-20:]):.2f}")
    print(f"Test Avg Portfolio: ${np.mean(test_portfolios):.2f}")
    print(f"Test Avg Reward: {np.mean(test_rewards):.2f}")
    
    # Save final portfolio values
    final_train_portfolio = env.net_worth
    final_test_portfolio = np.mean(test_portfolios)
    print(f"\nTraining complete! Final portfolio value: ${final_portfolio:.2f}")


if __name__ == "__main__":
    main()

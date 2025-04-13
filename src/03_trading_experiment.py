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


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


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

    def train(self, num_episodes: int):
        portfolio_values = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_profit = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.total_steps += 1
                episode_profit += reward
                
                # Store transition in replay buffer
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                # Update networks if we have enough samples
                if self.buffer.ready():
                    batch = self.buffer.sample(self.hyperparams["batch_size"])
                    self.update(batch)
                    self.update_epsilon()
                
                # Update target network periodically
                if self.total_steps % self.hyperparams["target_update"] == 0:
                    self.update_target_network()
            
            portfolio_values.append(self.env.net_worth)
            
            # Print progress
            if episode % 10 == 0:
                avg_profit = np.mean(portfolio_values[-10:])
                print(f"Episode: {episode}, "
                      f"Portfolio: ${self.env.net_worth:.2f}, "
                      f"Avg Profit: ${avg_profit:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        self.env.close()
        return portfolio_values


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
    def __init__(self, df, initial_balance=10000, window_size=10):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: OHLCV + technical indicators
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, 10),  # Adjust based on features
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.btc_held = 0
        self.current_step = self.window_size
        return self._next_observation()
        
    def _next_observation(self):
        """Get window of market observations"""
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'price_change_pct', 'volatility', 'fed_rate',
            'SP500', 'fng_value'
        ]
        obs = self.df.iloc[
            self.current_step-self.window_size:self.current_step
        ][features].values
        return obs
        
    def step(self, action):
        self.current_step += 1
        
        current_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Execute trade action
        if action == 1:  # Buy
            self.btc_held += self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            
        # Calculate reward
        self.net_worth = self.balance + self.btc_held * current_price
        reward = self.net_worth - prev_net_worth
        
        # Check if done
        done = self.net_worth <= 0 or self.current_step >= len(self.df)-1
        
        return self._next_observation(), reward, done, {}

def load_data():
    """Load and preprocess all market data"""
    from data_preparation import load_and_prepare_data
    return load_and_prepare_data(window_size=10)

def main():
    # Load and prepare data
    df = load_data()
    
    # Create trading environment
    env = BitcoinTradingEnv(df)
    
    hyperparams = {
        "hidden_dim": 128,  # Increased for more complex trading patterns
        "lr": 0.0005,       # Lower learning rate for stability
        "buffer_size": 100000,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.999,  # Slower decay
        "target_update": 100      # Less frequent updates
    }
    
    agent = DQNAgent(env, hyperparams)
    rewards = agent.train(190)  # Increased episodes for better training
    
    # Print final statistics
    print(f"Average reward over last 100 episodes: {sum(rewards[-100:]) / 100:.2f}")
    
    # Record and save video of the trained agent
    agent.record_video()


if __name__ == "__main__":
    main()

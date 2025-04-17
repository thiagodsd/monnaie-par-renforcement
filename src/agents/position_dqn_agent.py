"""
DQN agent optimized for position trading with price action analysis.
"""
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents.base import Agent
from models.replay_buffer import ReplayBuffer
from networks.price_action import PriceActionQNetwork
from utils.visualization import plot_training_metrics
from utils.data import get_device


class PositionDQNAgent(Agent):
    """DQN agent specialized for position trading with price action analysis"""
    def __init__(self, env, hyperparams: dict):
        super().__init__(env, hyperparams)
        # Get input shape from environment
        input_shape = env.observation_space.shape
        action_dim = env.action_space.n
        hidden_dim = hyperparams["hidden_dim"]
        
        # Get device (GPU if available, otherwise CPU)
        self.device = get_device()
        
        # Initialize Q-networks with the correct input shape
        self.q_network = PriceActionQNetwork(input_shape, hidden_dim, action_dim).to(self.device)
        self.target_network = PriceActionQNetwork(input_shape, hidden_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=hyperparams["lr"]
        )
        self.buffer = ReplayBuffer(hyperparams["buffer_size"])
        self.criterion = nn.MSELoss()
        
        self.epsilon_start = hyperparams["epsilon_start"]
        self.epsilon_end = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        
        # Position trading parameters
        self.position_cooldown = 0
        self.min_hold_period = hyperparams.get("min_hold_period", 5)
        
        # For visualization
        self.video_dir = "../data/videos"
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        self.train_trades = []

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
            
        # Convert state to tensor for neural network and move to device
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            
            # If selected BUY or SELL, set cooldown
            if action > 0:  # action is BUY or SELL
                self.position_cooldown = self.min_hold_period
                
            return action

    def update_epsilon(self):
        """Decay exploration rate over time"""
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )

    def update_target_network(self):
        """Periodically update target network to stabilize training"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self, batch):
        """Update Q-network using experience batch"""
        states, actions, rewards, next_states, dones = batch
        
        # Move all tensors to the device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get Q-values for current states and actions taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values (using Double DQN approach)
        with torch.no_grad():
            # Get actions from current policy network (not target network)
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            # Get target values from target network for these actions
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            # Compute the target using Bellman equation
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.hyperparams["gamma"] * next_q_values
        
        # Compute loss and update network
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes: int, eval_every: int = 20):
        """Train the agent for specified number of episodes"""
        portfolio_values = []
        cash_balance_history = []  # Track cash balance
        asset_value_history = []   # Track asset value
        eval_portfolios = []
        loss_history = []
        reward_history = []
        eval_rewards = []
        self.train_trades = []
        
        # Create tqdm progress bar for episodes
        progress_bar = tqdm(range(num_episodes), desc="Training", unit="episode")
        
        for episode in progress_bar:
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
                
                # Extract info dict for tracking portfolio components
                if len(step_result) == 5:  # new gym api
                    next_state, reward, done, truncated, info = step_result
                    done = done or truncated
                else:  # old gym api
                    next_state, reward, done, info = step_result
                    
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
            
            # Record portfolio values and breakdown at end of episode
            portfolio_values.append(self.env.net_worth)
            cash_balance_history.append(self.env.cash_balance)
            asset_value_history.append(self.env.asset_value)
            
            if episode == num_episodes - 1:
                self.train_trades = self.env.trades.copy()
            
            if (episode + 1) % eval_every == 0:
                eval_portfolio, eval_reward = self.evaluate()
                eval_portfolios.append(eval_portfolio)
                eval_rewards.append(eval_reward)
                print(f"\nEvaluation after {episode+1} episodes:")
                print(f"  Avg Portfolio: ${np.mean(eval_portfolio):.2f}")
                print(f"  Avg Reward: {np.mean(eval_reward):.2f}\n")
            
            # Update progress bar with portfolio info
            avg_profit = np.mean(portfolio_values[-10:] if len(portfolio_values) >= 10 else portfolio_values)
            progress_bar.set_postfix({
                'Portfolio': f"${self.env.net_worth:.2f}",
                'Cash': f"${self.env.cash_balance:.2f}",
                'Assets': f"${self.env.asset_value:.2f}",
                'Avg Profit': f"${avg_profit:.2f}",
                'Epsilon': f"{self.epsilon:.4f}"
            })
            
            # Still keep some periodic console output for reference
            if episode % 10 == 0 and episode > 0:
                avg_profit = np.mean(portfolio_values[-10:] if len(portfolio_values) >= 10 else portfolio_values)
                print(f"Episode: {episode}, "
                      f"Portfolio: ${self.env.net_worth:.2f} (Cash: ${self.env.cash_balance:.2f}, "
                      f"Assets: ${self.env.asset_value:.2f}), "
                      f"Avg Profit: ${avg_profit:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        self.env.close()
        self.save_training_plots(loss_history, portfolio_values, cash_balance_history, asset_value_history)
        return portfolio_values, loss_history, reward_history, eval_portfolios, eval_rewards

    def evaluate(self, num_episodes: int = 10):
        """Evaluate agent performance without exploration"""
        original_epsilon = self.epsilon
        original_cooldown = self.position_cooldown
        self.epsilon = 0  # disable exploration
        self.position_cooldown = 0  # reset cooldown
        eval_portfolios = list()
        eval_rewards = list()
        
        # Create progress bar for evaluation
        eval_progress = tqdm(range(num_episodes), desc="Evaluating", unit="episode", leave=False)
        
        for _ in eval_progress:
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
            
            current_portfolio = self.env.net_worth
            current_reward = np.mean(episode_rewards)
            eval_portfolios.append(current_portfolio)
            eval_rewards.append(current_reward)
            
            # Update progress bar
            eval_progress.set_postfix({
                'Portfolio': f"${current_portfolio:.2f}",
                'Reward': f"{current_reward:.2f}"
            })
        
        self.epsilon = original_epsilon  # Restore original epsilon
        self.position_cooldown = original_cooldown  # Restore original cooldown
        return eval_portfolios, eval_rewards
    
    def save_training_plots(self, loss_history, portfolio_values, cash_balance=None, asset_value=None):
        """Save training metrics plots"""
        # Create output directory if it doesn't exist
        plots_dir = Path("../data/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the enhanced plot_training_metrics function with portfolio breakdown
        plot_training_metrics(loss_history, portfolio_values, cash_balance, asset_value)
        
        # Plot portfolio value history
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values)
        plt.title('Portfolio Value History')
        plt.xlabel('Episodes')
        plt.ylabel('Portfolio Value ($)')
        plt.savefig(plots_dir / 'portfolio_history.png')
        plt.close()
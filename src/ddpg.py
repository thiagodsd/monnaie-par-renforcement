import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Actor network for DDPG - outputs continuous actions"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(Actor, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, output_size)
        self.tanh = nn.Tanh()  # Action range [-1, 1]
    
    def forward(self, x):
        features = self.features(x)
        return self.tanh(self.output(features))

class Critic(nn.Module):
    """Critic network for DDPG - estimates Q-values for state-action pairs"""
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int]):
        super(Critic, self).__init__()
        
        # First process state
        self.state_fc = nn.Linear(state_size, hidden_sizes[0])
        
        # Then combine with action
        self.fc_layers = []
        prev_size = hidden_sizes[0] + action_size
        for hidden_size in hidden_sizes[1:]:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            self.fc_layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.fc_layers.append(nn.Linear(prev_size, 1))
        self.fc = nn.Sequential(*self.fc_layers)
    
    def forward(self, state, action):
        state_features = torch.relu(self.state_fc(state))
        x = torch.cat([state_features, action], dim=1)
        return self.fc(x)

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, action_size: int, mu: float = 0.0, theta: float = 0.15, 
                 sigma: float = 0.2):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_size) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_size)
        self.state = self.state + dx
        return self.state

class TradingEnvironmentContinuous:
    """Trading environment with continuous action space for DDPG"""
    def __init__(self, data: pd.DataFrame, window_size: int = 21, initial_balance: float = 10000.0):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.buy_fee = 0.02
        self.sell_fee = 0.05
        self.avg_buy_price = 0.0
        self.returns_window = 20
        self.risk_free_rate = 0.02 / 252
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.avg_buy_price = 0.0
        self.done = False
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step < self.window_size:
            return np.zeros((self.window_size, 13))
        
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Extract features
        features = np.zeros((self.window_size, 13))
        features[:, 0] = window_data['open'].values
        features[:, 1] = window_data['high'].values
        features[:, 2] = window_data['low'].values
        features[:, 3] = window_data['close'].values
        features[:, 4] = window_data.get('ema_9', window_data['sma_7']).values
        features[:, 5] = window_data.get('ema_21', window_data['sma_21']).values
        features[:, 6] = window_data['rsi'].values
        features[:, 7] = window_data['volatility'].values
        features[:, 8] = window_data['volume'].values
        features[:, 9] = window_data['sp500_change'].values
        features[:, 10] = window_data['djia_change'].values
        features[:, 11] = window_data['fng_value'].values
        
        # Calculate position PnL for the 12th feature
        current_price = window_data['close'].iloc[-1]
        if self.crypto_held > 0 and self.avg_buy_price > 0:
            position_pnl = ((current_price - self.avg_buy_price) / self.avg_buy_price) * 100
        else:
            position_pnl = -2.0
        
        features[:, 12] = position_pnl
        
        # Normalize
        price_min = features[:, :4].min()
        price_max = features[:, :4].max()
        if price_max > price_min:
            features[:, :6] = (features[:, :6] - price_min) / (price_max - price_min)
        
        features[:, 6] = features[:, 6] / 100.0  # RSI
        features[:, 7] = features[:, 7] / (features[:, 3].max() + 1e-8)  # ATR
        
        vol_min = features[:, 8].min()
        vol_max = features[:, 8].max()
        if vol_max > vol_min:
            features[:, 8] = (features[:, 8] - vol_min) / (vol_max - vol_min)
        
        features[:, 9:11] = np.clip(features[:, 9:11], -0.1, 0.1)
        features[:, 9:11] = (features[:, 9:11] + 0.1) / 0.2
        features[:, 11] = features[:, 11] / 100.0  # FNG
        
        features[:, 12] = np.clip(features[:, 12], -50, 50) / 100.0 + 0.5
        
        return features
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute continuous action
        action: float in range [-1, 1]
        - Negative values: sell (percentage of holdings)
        - Positive values: buy (percentage of available cash)
        - Near zero: hold
        """
        if self.done:
            raise ValueError("Episode is done")
        
        current_price = self.data.iloc[self.current_step]['close']
        previous_portfolio_value = self.portfolio_value
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Execute action with deadzone for small actions
        if abs(action) < 0.05:  # Hold if action is too small
            pass
        elif action > 0:  # Buy
            # Buy with percentage of available cash
            max_buy_amount = self.cash / (current_price * (1 + self.buy_fee))
            buy_amount = max_buy_amount * action
            if buy_amount > 0:
                cost = buy_amount * current_price * (1 + self.buy_fee)
                if self.crypto_held > 0:
                    total_value = self.crypto_held * self.avg_buy_price + buy_amount * current_price
                    self.avg_buy_price = total_value / (self.crypto_held + buy_amount)
                else:
                    self.avg_buy_price = current_price
                self.cash -= cost
                self.crypto_held += buy_amount
        else:  # Sell (action < 0)
            # Sell with percentage of holdings
            sell_amount = self.crypto_held * abs(action)
            if sell_amount > 0:
                self.cash += sell_amount * current_price * (1 - self.sell_fee)
                self.crypto_held -= sell_amount
                if self.crypto_held < 1e-8:  # Handle floating point precision
                    self.crypto_held = 0.0
                    self.avg_buy_price = 0.0
        
        # Update portfolio value
        self.portfolio_value = self.cash + self.crypto_held * current_price
        
        # Calculate return and update history
        current_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        self.returns.append(current_return)
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate Sharpe ratio based reward
        if len(self.returns) >= self.returns_window:
            recent_returns = self.returns[-self.returns_window:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 0:
                sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
                reward = sharpe_ratio * 0.01
            else:
                reward = (mean_return - self.risk_free_rate) * 0.1
        else:
            reward = current_return
        
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        next_observation = self._get_observation() if not self.done else np.zeros((self.window_size, 13))
        
        return next_observation, reward, self.done, {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'crypto_held': self.crypto_held,
            'action': action
        }

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(done).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

def soft_update(target_network, source_network, tau):
    """Soft update of target network parameters"""
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def train_ddpg(env, episodes=5000, batch_size=64, buffer_size=100000,
               learning_rate_actor=0.0001, learning_rate_critic=0.001,
               gamma=0.99, tau=0.001, train_freq=1, min_buffer_size=10000,
               hidden_sizes_actor=[400, 300], hidden_sizes_critic=[400, 300]):
    
    input_size = env.window_size * 13
    action_size = 1  # Single continuous action
    
    # Create networks
    actor = Actor(input_size, hidden_sizes_actor, action_size).to(device)
    actor_target = Actor(input_size, hidden_sizes_actor, action_size).to(device)
    actor_target.load_state_dict(actor.state_dict())
    
    critic = Critic(input_size, action_size, hidden_sizes_critic).to(device)
    critic_target = Critic(input_size, action_size, hidden_sizes_critic).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate_critic)
    
    # Replay buffer and noise
    replay_buffer = ReplayBuffer(buffer_size)
    noise = OrnsteinUhlenbeckNoise(action_size)
    
    episode_rewards = []
    episode_portfolio_values = []
    
    for episode in tqdm(range(episodes), desc="Training DDPG"):
        state = env.reset()
        noise.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Select action with exploration noise
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
                action = actor(state_tensor).cpu().numpy()[0]
                action_noise = noise.sample()
                action = np.clip(action + action_noise, -1.0, 1.0)
            
            next_state, reward, done, info = env.step(action[0])
            replay_buffer.push(state.flatten(), action, reward, next_state.flatten(), done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Train
            if len(replay_buffer) >= min_buffer_size and episode_steps % train_freq == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Update critic
                with torch.no_grad():
                    next_actions = actor_target(next_states)
                    target_q_values = critic_target(next_states, next_actions)
                    target_values = rewards.unsqueeze(1) + gamma * target_q_values * (1 - dones.unsqueeze(1))
                
                current_q_values = critic(states, actions)
                critic_loss = nn.MSELoss()(current_q_values, target_values)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Update actor
                actor_loss = -critic(states, actor(states)).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Soft update target networks
                soft_update(actor_target, actor, tau)
                soft_update(critic_target, critic, tau)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(info['portfolio_value'])
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, "
                  f"Avg Portfolio: ${avg_portfolio:.2f}")
    
    return actor, critic, episode_rewards, episode_portfolio_values

def validate_model(actor, env):
    actor.eval()
    state = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = [env.portfolio_value]
    
    with torch.no_grad():
        while True:
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().numpy()[0, 0]
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            actions_taken.append(action)
            portfolio_values.append(info['portfolio_value'])
            
            state = next_state
            
            if done:
                break
    
    return {
        'total_reward': total_reward,
        'final_portfolio_value': portfolio_values[-1],
        'initial_portfolio_value': portfolio_values[0],
        'return_pct': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100,
        'num_trades': sum(1 for a in actions_taken if abs(a) > 0.05),
        'actions': actions_taken,
        'portfolio_values': portfolio_values
    }

def main():
    # Load data
    data_path = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/04_feature/analytical_base_table_01.parquet"
    df = pd.read_parquet(data_path)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split data
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:].copy()
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    
    # Create environments
    train_env = TradingEnvironmentContinuous(train_data)
    val_env = TradingEnvironmentContinuous(val_data)
    
    # Train model
    print("\nTraining DDPG...")
    actor, critic, train_rewards, train_portfolios = train_ddpg(train_env)
    
    # Save models
    model_dir = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/06_models/ddpg"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'ddpg_trading_model.pth')
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'hidden_sizes_actor': [400, 300],
        'hidden_sizes_critic': [400, 300],
        'input_size': train_env.window_size * 13,
        'action_size': 1
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Validate model
    print("\nValidating model...")
    val_results = validate_model(actor, val_env)
    print(f"Validation Return: {val_results['return_pct']:.2f}%")
    print(f"Number of trades: {val_results['num_trades']}")
    print(f"Final portfolio value: ${val_results['final_portfolio_value']:.2f}")
    
    # Create output dataframe with predictions
    actor.eval()
    output_data = []
    
    # Process validation data
    val_env_output = TradingEnvironmentContinuous(val_data)
    state = val_env_output.reset()
    step = 0
    
    with torch.no_grad():
        while step < len(val_data) - val_env_output.window_size:
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().numpy()[0, 0]
            
            # Store prediction
            current_idx = val_env_output.current_step
            if current_idx < len(val_data):
                output_data.append({
                    'date': val_data.iloc[current_idx]['date'],
                    'state': state.flatten().tolist(),
                    'action': action
                })
            
            next_state, _, done, _ = val_env_output.step(action)
            state = next_state
            step += 1
            
            if done:
                break
    
    # Save output data
    output_df = pd.DataFrame(output_data)
    output_dir = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/07_model_output/ddpg"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.parquet")
    output_df.to_parquet(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Also save analytical base table with predictions merged
    val_data_with_predictions = val_data.iloc[val_env_output.window_size:val_env_output.window_size + len(output_df)].copy()
    val_data_with_predictions['action'] = output_df['action'].values
    val_data_with_predictions['state'] = output_df['state'].values
    
    output_abt_path = os.path.join(output_dir, "analytical_base_table_with_predictions.parquet")
    val_data_with_predictions.to_parquet(output_abt_path, index=False)
    print(f"Analytical base table with predictions saved to: {output_abt_path}")

if __name__ == "__main__":
    main()
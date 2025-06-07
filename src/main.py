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

class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super(DQN, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TradingEnvironment:
    def __init__(self, data: pd.DataFrame, window_size: int = 21, initial_balance: float = 10000.0):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.buy_fee = 0.02
        self.sell_fee = 0.05
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.done = False
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step < self.window_size:
            return np.zeros((self.window_size, 12))
        
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Extract features
        features = np.zeros((self.window_size, 12))
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
        
        # Normalize
        price_min = features[:, :4].min()
        price_max = features[:, :4].max()
        if price_max > price_min:
            features[:, :6] = (features[:, :6] - price_min) / (price_max - price_min)
        
        features[:, 6] = features[:, 6] / 100.0  # RSI
        features[:, 7] = features[:, 7] / (features[:, 3].max() + 1e-8)  # ATR as % of close
        
        vol_min = features[:, 8].min()
        vol_max = features[:, 8].max()
        if vol_max > vol_min:
            features[:, 8] = (features[:, 8] - vol_min) / (vol_max - vol_min)
        
        features[:, 9:11] = np.clip(features[:, 9:11], -0.1, 0.1)
        features[:, 9:11] = (features[:, 9:11] + 0.1) / 0.2
        features[:, 11] = features[:, 11] / 100.0  # FNG
        
        return features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise ValueError("Episode is done")
        
        current_price = self.data.iloc[self.current_step]['close']
        previous_portfolio_value = self.portfolio_value
        
        # Execute action
        if action == 0:  # HOLD
            pass
        elif action in [1, 2, 3, 4]:  # BUY
            percentage = [0.25, 0.5, 0.75, 1.0][action - 1]
            max_buyable = self.cash / (current_price * (1 + self.buy_fee))
            buy_amount = max_buyable * percentage
            if buy_amount > 0:
                cost = buy_amount * current_price * (1 + self.buy_fee)
                self.cash -= cost
                self.crypto_held += buy_amount
        elif action in [5, 6, 7, 8]:  # SELL
            percentage = [0.25, 0.5, 0.75, 1.0][action - 5]
            sell_amount = self.crypto_held * percentage
            if sell_amount > 0:
                revenue = sell_amount * current_price * (1 - self.sell_fee)
                self.cash += revenue
                self.crypto_held -= sell_amount
        
        # Update portfolio value
        self.portfolio_value = self.cash + self.crypto_held * current_price
        
        # Calculate reward
        reward = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        next_observation = self._get_observation() if not self.done else np.zeros((self.window_size, 12))
        
        return next_observation, reward, self.done, {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'crypto_held': self.crypto_held
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
            torch.LongTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(done).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_dqn(env, episodes=1000, batch_size=32, buffer_size=100000, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=50000,
              learning_rate=0.001, gamma=0.99, target_update_freq=1000,
              train_freq=4, min_buffer_size=10000, hidden_sizes=[512, 256, 128]):
    
    input_size = env.window_size * 12
    output_size = 9
    
    q_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
    total_steps = 0
    
    episode_rewards = []
    episode_portfolio_values = []
    
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(state.flatten()).unsqueeze(0).to(device))
                    action = q_values.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state.flatten(), action, reward, next_state.flatten(), done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Decay epsilon
            if epsilon > epsilon_end:
                epsilon -= epsilon_decay
            
            # Train
            if total_steps % train_freq == 0 and len(replay_buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update target network
            if total_steps % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(info['portfolio_value'])
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, "
                  f"Avg Portfolio: ${avg_portfolio:.2f}, Epsilon: {epsilon:.3f}")
    
    return q_network, episode_rewards, episode_portfolio_values

def validate_model(model, env):
    model.eval()
    state = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = [env.portfolio_value]
    
    with torch.no_grad():
        while True:
            q_values = model(torch.FloatTensor(state.flatten()).unsqueeze(0).to(device))
            action = q_values.argmax().item()
            
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
        'num_trades': sum(1 for a in actions_taken if a != 0),
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
    train_env = TradingEnvironment(train_data)
    val_env = TradingEnvironment(val_data)
    
    # Train model
    print("\nTraining DQN...")
    model, train_rewards, train_portfolios = train_dqn(train_env, episodes=1000)
    
    # Save model
    model_path = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/06_models/dqn_trading_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_sizes': [512, 256, 128],
        'input_size': train_env.window_size * 12,
        'output_size': 9
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Validate model
    print("\nValidating model...")
    val_results = validate_model(model, val_env)
    print(f"Validation Return: {val_results['return_pct']:.2f}%")
    print(f"Number of trades: {val_results['num_trades']}")
    print(f"Final portfolio value: ${val_results['final_portfolio_value']:.2f}")
    
    # Create output dataframe with predictions
    model.eval()
    output_data = []
    
    # Process validation data
    val_env_output = TradingEnvironment(val_data)
    state = val_env_output.reset()
    step = 0
    
    with torch.no_grad():
        while step < len(val_data) - val_env_output.window_size:
            q_values = model(torch.FloatTensor(state.flatten()).unsqueeze(0).to(device))
            action = q_values.argmax().item()
            
            # Store prediction
            current_idx = val_env_output.current_step
            if current_idx < len(val_data):
                output_data.append({
                    'date': val_data.iloc[current_idx]['date'],
                    'state': state.flatten().tolist(),
                    'action': action,
                    'q_values': q_values.cpu().numpy().flatten().tolist()
                })
            
            next_state, _, done, _ = val_env_output.step(action)
            state = next_state
            step += 1
            
            if done:
                break
    
    # Save output data
    output_df = pd.DataFrame(output_data)
    output_path = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/07_model_output/predictions.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Also save analytical base table with predictions merged
    val_data_with_predictions = val_data.iloc[val_env_output.window_size:val_env_output.window_size + len(output_df)].copy()
    val_data_with_predictions['action'] = output_df['action'].values
    val_data_with_predictions['state'] = output_df['state'].values
    
    output_abt_path = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/07_model_output/analytical_base_table_with_predictions.parquet"
    val_data_with_predictions.to_parquet(output_abt_path, index=False)
    print(f"Analytical base table with predictions saved to: {output_abt_path}")

if __name__ == "__main__":
    main()
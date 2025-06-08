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
import logging
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
log_dir = "../data/logs/dqn_lstm"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
    ]
)
logger = logging.getLogger(__name__)

class DQN_LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(DQN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layers after LSTM
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class TradingEnvironmentLSTM:
    def __init__(self, data: pd.DataFrame, window_size: int = 21, initial_balance: float = 10000.0):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.buy_fee = 0.02
        self.sell_fee = 0.05
        self.avg_buy_price = 0.0
        self.total_invested = 0.0
        self.returns_window = 20
        self.risk_free_rate = 0.02 / 252
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size  # Start after we have enough history
        self.cash = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.avg_buy_price = 0.0
        self.total_invested = 0.0
        self.done = False
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        return self._get_observation()
    
    def _get_observation(self):
        # Get the last window_size steps of data
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        if end_idx > len(self.data):
            # Return zeros if we don't have enough data
            sequence = np.zeros((self.window_size, 15))  # 15 features including position info
            return sequence
        
        # Extract window of market data
        window_data = self.data.iloc[start_idx:end_idx]
        
        # If we don't have enough history, pad with first available data
        if len(window_data) < self.window_size:
            padding_size = self.window_size - len(window_data)
            if len(window_data) > 0:
                padding_data = window_data.iloc[[0] * padding_size]
                window_data = pd.concat([padding_data, window_data], ignore_index=True)
            else:
                return np.zeros((self.window_size, 15))
        
        # Create sequence array
        sequence = np.zeros((self.window_size, 15))
        
        for i, (_, row) in enumerate(window_data.iterrows()):
            # Market features (12 features)
            sequence[i, 0] = row['open']
            sequence[i, 1] = row['high'] 
            sequence[i, 2] = row['low']
            sequence[i, 3] = row['close']
            sequence[i, 4] = row.get('ema_9', row.get('sma_7', row['close']))
            sequence[i, 5] = row.get('ema_21', row.get('sma_21', row['close']))
            sequence[i, 6] = row['rsi']
            sequence[i, 7] = row['volatility']
            sequence[i, 8] = row['volume']
            sequence[i, 9] = row['sp500_change']
            sequence[i, 10] = row['djia_change']
            sequence[i, 11] = row['fng_value']
            
            # Position information (3 features) - same for all timesteps in window
            current_price = self.data.iloc[self.current_step - 1]['close'] if self.current_step > 0 else row['close']
            
            # Position PnL percentage
            if self.crypto_held > 0 and self.avg_buy_price > 0:
                position_pnl = ((current_price - self.avg_buy_price) / self.avg_buy_price) * 100
            else:
                position_pnl = 0.0
            sequence[i, 12] = position_pnl
            
            # Cash and crypto position as percentages of initial balance
            sequence[i, 13] = (self.cash / self.initial_balance) * 100
            
            current_position_value = self.crypto_held * current_price
            sequence[i, 14] = (current_position_value / self.initial_balance) * 100
        
        # Normalize the sequence
        sequence = self._normalize_sequence(sequence)
        
        return sequence
    
    def _normalize_sequence(self, sequence):
        # Normalize each feature across the time dimension
        normalized = sequence.copy()
        
        # Price features (0-5): normalize relative to close price
        for i in range(4):  # OHLC
            if sequence[:, 3].max() > 0:  # Close prices
                normalized[:, i] = sequence[:, i] / sequence[:, 3]
        
        # EMAs (4-5): normalize relative to close
        for i in range(4, 6):
            if sequence[:, 3].max() > 0:
                normalized[:, i] = sequence[:, i] / sequence[:, 3]
        
        # RSI (6): already 0-100, normalize to 0-1
        normalized[:, 6] = sequence[:, 6] / 100.0
        
        # Volatility (7): normalize by close price
        if sequence[:, 3].max() > 0:
            normalized[:, 7] = sequence[:, 7] / sequence[:, 3]
        
        # Volume (8): log normalize
        vol_data = sequence[:, 8]
        if vol_data.max() > 0:
            log_vol = np.log1p(vol_data)
            if log_vol.max() > log_vol.min():
                normalized[:, 8] = (log_vol - log_vol.min()) / (log_vol.max() - log_vol.min())
        
        # Market indices (9-10): clip to reasonable range
        normalized[:, 9:11] = np.clip(sequence[:, 9:11], -0.1, 0.1) / 0.2 + 0.5
        
        # Fear & Greed (11): 0-100 to 0-1
        normalized[:, 11] = sequence[:, 11] / 100.0
        
        # Position PnL (12): clip and normalize
        normalized[:, 12] = np.clip(sequence[:, 12], -50, 50) / 100.0 + 0.5
        
        # Cash and position percentages (13-14): normalize to 0-1
        normalized[:, 13] = np.clip(sequence[:, 13], 0, 200) / 200.0
        normalized[:, 14] = np.clip(sequence[:, 14], 0, 200) / 200.0
        
        return normalized
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise ValueError("Episode is done")
        
        current_price = self.data.iloc[self.current_step]['close']
        previous_portfolio_value = self.portfolio_value
        
        # Execute action
        if action == 1:  # BUY 25%
            buy_amount = self.cash / (current_price * (1 + self.buy_fee)) * 0.25
            if buy_amount > 0:
                cost = buy_amount * current_price * (1 + self.buy_fee)
                # Update average buy price
                if self.crypto_held > 0:
                    total_value = self.crypto_held * self.avg_buy_price + buy_amount * current_price
                    self.avg_buy_price = total_value / (self.crypto_held + buy_amount)
                else:
                    self.avg_buy_price = current_price
                
                self.total_invested += cost
                self.cash -= cost
                self.crypto_held += buy_amount
                
        elif action == 2:  # SELL 25%
            sell_amount = self.crypto_held * 0.25
            if sell_amount > 0:
                self.cash += sell_amount * current_price * (1 - self.sell_fee)
                
                # Reduce total invested proportionally
                if self.crypto_held > 0:
                    proportion_sold = sell_amount / self.crypto_held
                    self.total_invested *= (1 - proportion_sold)
                
                self.crypto_held -= sell_amount
                if self.crypto_held == 0:
                    self.avg_buy_price = 0.0
                    self.total_invested = 0.0
        
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
        
        # Add penalties for invalid actions
        min_buy_amount = 100
        if action == 1 and self.cash < min_buy_amount:
            reward = -0.1
        if action == 2 and self.crypto_held < 1e-8:
            reward = -0.1
        
        # Clip reward
        reward = np.clip(reward, -1.0, 1.0)
        
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        next_observation = self._get_observation() if not self.done else np.zeros((self.window_size, 15))
        
        return next_observation, reward, self.done, {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'crypto_held': self.crypto_held
        }

class ReplayBufferLSTM:
    def __init__(self, capacity: int, sequence_length: int):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
    
    def push(self, state, action, reward, next_state, done):
        # Ensure states are numpy arrays with correct shape
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        
        # Ensure correct shape (sequence_length, features)
        assert state.shape == (self.sequence_length, 15), f"State shape {state.shape} != ({self.sequence_length}, 15)"
        assert next_state.shape == (self.sequence_length, 15), f"Next state shape {next_state.shape} != ({self.sequence_length}, 15)"
        
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

def train_dqn_lstm(env, episodes=1000, batch_size=32, buffer_size=100000,
                   epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=None,
                   learning_rate=0.0001, gamma=0.99, target_update_freq=2000,
                   train_freq=4, min_buffer_size=10000, 
                   hidden_size=128, num_layers=2, dropout=0.2):
    
    # Log training configuration
    logger.info("="*60)
    logger.info("STARTING DQN-LSTM TRAINING")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Hidden size: {hidden_size}")
    logger.info(f"LSTM layers: {num_layers}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Window size: {env.window_size}")
    logger.info("="*60)
    
    input_size = 15  # 12 market features + 3 position features
    output_size = 3
    
    q_network = DQN_LSTM(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    target_network = DQN_LSTM(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBufferLSTM(buffer_size, env.window_size)
    
    # Calculate epsilon decay steps
    if epsilon_decay_steps is None:
        estimated_steps_per_episode = len(env.data) - env.window_size
        epsilon_decay_steps = int(episodes * estimated_steps_per_episode * 0.8)
    
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
    total_steps = 0
    
    episode_rewards = []
    episode_portfolio_values = []
    episode_metrics = []
    
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        step_history = []
        
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    # Add batch dimension: (1, sequence_length, features)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Store step information for logging
            actual_price = env.data.iloc[env.current_step - 1]['close']
            current_rsi = state[-1, 6] * 100  # Last timestep RSI
            current_pnl = (state[-1, 12] - 0.5) * 100  # Last timestep PnL
            cash_pct = state[-1, 13] * 200  # Last timestep cash %
            position_pct = state[-1, 14] * 200  # Last timestep position %
            
            step_info = {
                'step': episode_steps,
                'action': action,
                'reward': reward,
                'portfolio': info['portfolio_value'],
                'cash': info['cash'],
                'crypto': info['crypto_held'],
                'price': actual_price,
                'rsi': current_rsi,
                'pnl': current_pnl,
                'cash_pct': cash_pct,
                'position_pct': position_pct
            }
            step_history.append(step_info)
            
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
                
                if total_steps % 1000 == 0:
                    logger.info(f"Step {total_steps}: Loss = {loss.item():.6f}, Epsilon = {epsilon:.4f}")
            
            # Update target network
            if total_steps % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
                logger.info(f"Step {total_steps}: Target network updated")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(info['portfolio_value'])
        
        # Collect detailed episode metrics
        portfolio_return = (info['portfolio_value'] - env.initial_balance) / env.initial_balance * 100
        action_counts = [sum(1 for s in step_history if s['action'] == i) for i in range(3)]
        
        episode_metric = {
            'episode': episode + 1,
            'total_reward': episode_reward,
            'portfolio_value': info['portfolio_value'],
            'portfolio_return_pct': portfolio_return,
            'cash': info['cash'],
            'crypto_held': info['crypto_held'],
            'total_invested': env.total_invested,
            'avg_buy_price': env.avg_buy_price,
            'steps': episode_steps,
            'epsilon': epsilon,
            'total_steps': total_steps,
            'action_hold': action_counts[0],
            'action_buy': action_counts[1],
            'action_sell': action_counts[2],
            'hold_pct': action_counts[0] / len(step_history) * 100 if step_history else 0,
            'buy_pct': action_counts[1] / len(step_history) * 100 if step_history else 0,
            'sell_pct': action_counts[2] / len(step_history) * 100 if step_history else 0,
        }
        
        # Add moving averages
        if len(episode_rewards) >= 10:
            episode_metric['reward_ma_10'] = np.mean(episode_rewards[-10:])
            episode_metric['portfolio_ma_10'] = np.mean(episode_portfolio_values[-10:])
        if len(episode_rewards) >= 50:
            episode_metric['reward_ma_50'] = np.mean(episode_rewards[-50:])
            episode_metric['portfolio_ma_50'] = np.mean(episode_portfolio_values[-50:])
        if len(episode_rewards) >= 100:
            episode_metric['reward_ma_100'] = np.mean(episode_rewards[-100:])
            episode_metric['portfolio_ma_100'] = np.mean(episode_portfolio_values[-100:])
            
        episode_metrics.append(episode_metric)
        
        # Log every episode
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, "
                   f"Portfolio = ${info['portfolio_value']:.2f}, "
                   f"Return = {portfolio_return:.2f}%, "
                   f"Cash = ${info['cash']:.2f}, "
                   f"Crypto = {info['crypto_held']:.6f}, "
                   f"Steps = {episode_steps}")
        
        # Print to console every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, "
                  f"Avg Portfolio: ${avg_portfolio:.2f}, Epsilon: {epsilon:.3f}")
            
            logger.info(f"Episode {episode + 1} Summary: "
                       f"Avg Reward (100 eps) = {avg_reward:.4f}, "
                       f"Avg Portfolio (100 eps) = ${avg_portfolio:.2f}, "
                       f"Epsilon = {epsilon:.3f}")
            
            # Log detailed step history every 50 episodes
            if (episode + 1) % 50 == 0:
                logger.info(f"\nDetailed steps for Episode {episode + 1}:")
                logger.info("Step | Action | Price    | RSI   | PnL%   | Cash%  | Pos%   | Reward    | Portfolio")
                logger.info("-" * 100)
                
                # Log first 10 and last 10 steps
                if len(step_history) <= 20:
                    steps_to_log = step_history
                else:
                    steps_to_log = step_history[:10] + [{'step': '...', 'action': '...', 'price': '...', 
                                                         'rsi': '...', 'pnl': '...', 'cash_pct': '...', 
                                                         'position_pct': '...', 'reward': '...', 'portfolio': '...'}] + step_history[-10:]
                
                action_names = {0: "HOLD", 1: "BUY ", 2: "SELL"}
                for step in steps_to_log:
                    if step['step'] == '...':
                        logger.info("  ...    ...      ...       ...     ...     ...      ...      ...         ...")
                    else:
                        logger.info(f"{step['step']:4d} | {action_names.get(step['action'], str(step['action']))} | "
                                   f"{step['price']:8.2f} | {step['rsi']:5.1f} | {step['pnl']:6.2f} | "
                                   f"{step['cash_pct']:6.1f} | {step['position_pct']:6.1f} | "
                                   f"{step['reward']:9.4f} | ${step['portfolio']:10.2f}")
                
                # Log action distribution
                logger.info(f"\nAction distribution for Episode {episode + 1}:")
                logger.info(f"  HOLD: {action_counts[0]:4d} ({action_counts[0]/len(step_history)*100:5.1f}%)")
                logger.info(f"  BUY:  {action_counts[1]:4d} ({action_counts[1]/len(step_history)*100:5.1f}%)")
                logger.info(f"  SELL: {action_counts[2]:4d} ({action_counts[2]/len(step_history)*100:5.1f}%)")
                logger.info("-" * 100)
    
    # Save episode metrics
    metrics_df = pd.DataFrame(episode_metrics)
    metrics_dir = "../data/07_model_output/dqn_lstm"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "training_metrics.parquet")
    metrics_df.to_parquet(metrics_path, index=False)
    logger.info(f"Training metrics saved to: {metrics_path}")
    
    return q_network, episode_rewards, episode_portfolio_values, episode_metrics

def validate_model(model, env):
    model.eval()
    state = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = [env.portfolio_value]
    
    with torch.no_grad():
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
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
    logger.info("="*60)
    logger.info("STARTING DQN-LSTM TRADING EXPERIMENT")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)
    
    # Load data
    data_path = "../data/04_feature/analytical_base_table_01.parquet"
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data from: {data_path}")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split data
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:].copy()
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    print(f"Log file: {log_file}")
    
    logger.info(f"Data split: {len(train_data)} training rows, {len(val_data)} validation rows")
    logger.info(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")
    logger.info(f"Validation period: {val_data['date'].min()} to {val_data['date'].max()}")
    
    # Create environments
    train_env = TradingEnvironmentLSTM(train_data, window_size=21)
    val_env = TradingEnvironmentLSTM(val_data, window_size=21)
    
    # Train model
    print("\nTraining DQN-LSTM...")
    model, train_rewards, train_portfolios, train_metrics = train_dqn_lstm(
        train_env,
        episodes=1000,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    # Save model
    model_dir = "../data/06_models/dqn_lstm"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dqn_lstm_trading_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'input_size': 15,
        'output_size': 3,
        'window_size': 21
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Print training metrics info
    metrics_output_path = "../data/07_model_output/dqn_lstm/training_metrics.parquet"
    print(f"Training metrics saved to: {metrics_output_path}")
    print(f"Total training episodes: {len(train_metrics)}")
    
    # Print training summary
    final_portfolio = train_metrics[-1]['portfolio_value']
    final_return = train_metrics[-1]['portfolio_return_pct']
    best_episode = max(train_metrics, key=lambda x: x['portfolio_return_pct'])
    worst_episode = min(train_metrics, key=lambda x: x['portfolio_return_pct'])
    
    print(f"\nTraining Summary:")
    print(f"  Final Portfolio Value: ${final_portfolio:.2f}")
    print(f"  Final Return: {final_return:.2f}%")
    print(f"  Best Episode: {best_episode['episode']} (Return: {best_episode['portfolio_return_pct']:.2f}%)")
    print(f"  Worst Episode: {worst_episode['episode']} (Return: {worst_episode['portfolio_return_pct']:.2f}%)")
    
    # Log training summary
    logger.info("="*60)
    logger.info("TRAINING COMPLETED - SUMMARY")
    logger.info("="*60)
    logger.info(f"Total episodes: {len(train_metrics)}")
    logger.info(f"Final portfolio value: ${final_portfolio:.2f}")
    logger.info(f"Final return: {final_return:.2f}%")
    logger.info(f"Best episode: {best_episode['episode']} (Return: {best_episode['portfolio_return_pct']:.2f}%)")
    logger.info(f"Worst episode: {worst_episode['episode']} (Return: {worst_episode['portfolio_return_pct']:.2f}%)")
    logger.info(f"Training metrics saved to: {metrics_output_path}")
    logger.info("="*60)
    
    # Validate model
    print("\nValidating model...")
    logger.info("="*60)
    logger.info("STARTING VALIDATION")
    logger.info("="*60)
    
    val_results = validate_model(model, val_env)
    
    print(f"Validation Return: {val_results['return_pct']:.2f}%")
    print(f"Number of trades: {val_results['num_trades']}")
    print(f"Final portfolio value: ${val_results['final_portfolio_value']:.2f}")
    
    # Log detailed validation results
    logger.info("Validation Results:")
    logger.info(f"  - Initial Portfolio: ${val_results['initial_portfolio_value']:.2f}")
    logger.info(f"  - Final Portfolio: ${val_results['final_portfolio_value']:.2f}")
    logger.info(f"  - Total Return: {val_results['return_pct']:.2f}%")
    logger.info(f"  - Total Reward: {val_results['total_reward']:.4f}")
    logger.info(f"  - Number of Trades: {val_results['num_trades']}")
    
    # Log action distribution
    actions = val_results['actions']
    action_counts = [actions.count(0), actions.count(1), actions.count(2)]
    logger.info("  - Action Distribution:")
    logger.info(f"    - HOLD: {action_counts[0]} ({action_counts[0]/len(actions)*100:.1f}%)")
    logger.info(f"    - BUY:  {action_counts[1]} ({action_counts[1]/len(actions)*100:.1f}%)")
    logger.info(f"    - SELL: {action_counts[2]} ({action_counts[2]/len(actions)*100:.1f}%)")
    
    # Create output dataframe with predictions
    model.eval()
    output_data = []
    
    # Process validation data
    val_env_output = TradingEnvironmentLSTM(val_data, window_size=21)
    state = val_env_output.reset()
    step = 0
    
    with torch.no_grad():
        while step < len(val_data) - 21:  # 21 days needed for window
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = q_values.argmax().item()
            
            # Store prediction
            current_idx = val_env_output.current_step
            if current_idx < len(val_data):
                output_data.append({
                    'date': val_data.iloc[current_idx]['date'],
                    'state': state.flatten().tolist(),  # Flatten sequence for storage
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
    output_dir = "../data/07_model_output/dqn_lstm"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.parquet")
    output_df.to_parquet(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Also save analytical base table with predictions merged
    val_data_with_predictions = val_data.iloc[21:21 + len(output_df)].copy()
    val_data_with_predictions['action'] = output_df['action'].values
    val_data_with_predictions['state'] = output_df['state'].values
    
    output_abt_path = os.path.join(output_dir, "analytical_base_table_with_predictions.parquet")
    val_data_with_predictions.to_parquet(output_abt_path, index=False)
    print(f"Analytical base table with predictions saved to: {output_abt_path}")
    
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info(f"Full training log saved to: {log_file}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
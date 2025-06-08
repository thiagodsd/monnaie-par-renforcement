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
log_dir = "../data/logs/dqn"
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
        self.avg_buy_price = 0.0  # Track average buy price for PnL calculation
        self.returns_window = 20  # Window for calculating Sharpe ratio
        self.risk_free_rate = 0.02 / 252  # 2% annual risk-free rate, daily
        self.reset()
        
    def reset(self):
        self.current_step = 7  # Start from day 7 to have weekly history
        self.cash = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.avg_buy_price = 0.0
        self.done = False
        self.portfolio_values = [self.initial_balance]  # Track portfolio values for Sharpe
        self.returns = []  # Track returns for Sharpe calculation
        return self._get_observation()
    
    def _get_observation(self):
        # Need at least 7 days of history for weekly comparison
        if self.current_step < 7:
            return np.zeros(26)  # 13 features * 2 (current + ratio)
        
        # Get current data (t)
        current_data = self.data.iloc[self.current_step]
        # Get data from 7 days ago (t-7)
        week_ago_data = self.data.iloc[self.current_step - 7]
        
        # Extract current features
        features = np.zeros(26)
        
        # Current values (t) - first 13 features
        features[0] = current_data['open']
        features[1] = current_data['high']
        features[2] = current_data['low']
        features[3] = current_data['close']
        features[4] = current_data.get('ema_9', current_data.get('sma_7', current_data['close']))
        features[5] = current_data.get('ema_21', current_data.get('sma_21', current_data['close']))
        features[6] = current_data['rsi']
        features[7] = current_data['volatility']
        features[8] = current_data['volume']
        features[9] = current_data['sp500_change']
        features[10] = current_data['djia_change']
        features[11] = current_data['fng_value']
        
        # Position PnL
        if self.crypto_held > 0 and self.avg_buy_price > 0:
            position_pnl = ((current_data['close'] - self.avg_buy_price) / self.avg_buy_price) * 100
        else:
            position_pnl = -2.0
        features[12] = position_pnl
        
        # Weekly ratios (t / t-7) - next 13 features
        # For prices and technical indicators, use ratio
        features[13] = current_data['open'] / (week_ago_data['open'] + 1e-8)  # Avoid division by zero
        features[14] = current_data['high'] / (week_ago_data['high'] + 1e-8)
        features[15] = current_data['low'] / (week_ago_data['low'] + 1e-8)
        features[16] = current_data['close'] / (week_ago_data['close'] + 1e-8)
        features[17] = features[4] / (week_ago_data.get('ema_9', week_ago_data.get('sma_7', week_ago_data['close'])) + 1e-8)
        features[18] = features[5] / (week_ago_data.get('ema_21', week_ago_data.get('sma_21', week_ago_data['close'])) + 1e-8)
        
        # For RSI, use difference since it's already a percentage
        features[19] = current_data['rsi'] - week_ago_data['rsi']
        
        # For volatility and volume, use ratio
        features[20] = current_data['volatility'] / (week_ago_data['volatility'] + 1e-8)
        features[21] = current_data['volume'] / (week_ago_data['volume'] + 1e-8)
        
        # Market indices already represent daily changes, so take difference
        features[22] = current_data['sp500_change'] - week_ago_data['sp500_change']
        features[23] = current_data['djia_change'] - week_ago_data['djia_change']
        
        # Fear & Greed difference
        features[24] = current_data['fng_value'] - week_ago_data['fng_value']
        
        # Keep current position PnL (no historical comparison needed)
        features[25] = position_pnl
        
        # Normalize features
        # Current prices (normalized together)
        price_features = features[0:6]
        price_min, price_max = price_features.min(), price_features.max()
        if price_max > price_min:
            features[0:6] = (features[0:6] - price_min) / (price_max - price_min)
        
        # Current indicators
        features[6] = features[6] / 100.0  # RSI
        features[7] = features[7] / (features[3] + 1e-8)  # ATR as % of close
        features[8] = features[8] / (features[8] + 1e-8) if features[8] > 0 else 0  # Volume self-normalized
        features[9:11] = np.clip(features[9:11], -0.1, 0.1) / 0.2 + 0.5  # Market indices
        features[11] = features[11] / 100.0  # F&G
        features[12] = np.clip(features[12], -50, 50) / 100.0 + 0.5  # PnL
        
        # Normalize ratios (center around 1.0)
        features[13:19] = np.clip(features[13:19], 0.5, 2.0) - 1.0  # Map [0.5, 2.0] to [-0.5, 1.0]
        features[19] = np.clip(features[19], -50, 50) / 100.0  # RSI diff
        features[20:22] = np.clip(features[20:22], 0.5, 2.0) - 1.0  # Volatility and volume ratios
        features[22:25] = np.clip(features[22:25], -0.2, 0.2) / 0.4 + 0.5  # Market diffs
        features[25] = np.clip(features[25], -50, 50) / 100.0 + 0.5  # PnL
        
        return features
    
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
                
                self.cash -= cost
                self.crypto_held += buy_amount
                
        elif action == 2:  # SELL 25%
            sell_amount = self.crypto_held * 0.25
            if sell_amount > 0:
                self.cash += sell_amount * current_price * (1 - self.sell_fee)
                self.crypto_held -= sell_amount
                # Reset avg_buy_price if all crypto is sold
                if self.crypto_held == 0:
                    self.avg_buy_price = 0.0
        
        # Update portfolio value
        self.portfolio_value = self.cash + self.crypto_held * current_price
        
        # Calculate return and update history
        current_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        self.returns.append(current_return)
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate Sharpe ratio based reward
        if len(self.returns) >= self.returns_window:
            # Use recent returns for Sharpe calculation
            recent_returns = self.returns[-self.returns_window:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 0:
                # Sharpe ratio calculation
                sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
                # Scale Sharpe ratio to reasonable reward range
                reward = sharpe_ratio * 0.01
            else:
                # If no volatility, reward is just the excess return
                reward = (mean_return - self.risk_free_rate) * 0.1
        else:
            # Not enough data for Sharpe, use simple return
            reward = current_return
        
        # Add small penalty for invalid actions
        if action == 2 and self.crypto_held == 0:
            reward = -0.01
        
        # Clip reward to prevent explosion
        original_reward = reward
        reward = np.clip(reward, -1.0, 1.0)
        
        # Log if reward was clipped (only occasionally to avoid spam)
        if original_reward != reward and hasattr(self, '_clip_warning_count'):
            if self._clip_warning_count < 10:
                logger.warning(f"Reward clipped: {original_reward:.4f} -> {reward:.4f}")
                self._clip_warning_count += 1
        elif not hasattr(self, '_clip_warning_count'):
            self._clip_warning_count = 0
        
        # Move to next step
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        next_observation = self._get_observation() if not self.done else np.zeros(26)
        
        return next_observation, reward, self.done, {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'crypto_held': self.crypto_held
        }

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Ensure states are numpy arrays with correct shape
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        
        # Ensure correct shape (26,)
        assert state.shape == (26,), f"State shape {state.shape} != (26,)"
        assert next_state.shape == (26,), f"Next state shape {next_state.shape} != (26,)"
        
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
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=None,
              learning_rate=0.0001, gamma=0.99, target_update_freq=2000,
              train_freq=4, min_buffer_size=10000, hidden_sizes=[128, 64, 32]):
    
    # Log training configuration
    logger.info("="*60)
    logger.info("STARTING DQN TRAINING")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Gamma: {gamma}")
    logger.info(f"Hidden sizes: {hidden_sizes}")
    logger.info(f"Initial epsilon: {epsilon_start}")
    logger.info(f"Final epsilon: {epsilon_end}")
    logger.info(f"Target update frequency: {target_update_freq}")
    logger.info("="*60)
    
    input_size = 26  # 13 current features + 13 weekly ratios
    output_size = 3
    
    q_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Calculate epsilon decay steps based on episodes if not provided
    if epsilon_decay_steps is None:
        # Estimate steps per episode (roughly length of data minus 7 for weekly history)
        estimated_steps_per_episode = len(env.data) - 7
        # Use 80% of total steps for exploration
        epsilon_decay_steps = int(episodes * estimated_steps_per_episode * 0.8)
    
    epsilon = epsilon_start
    epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
    total_steps = 0
    
    episode_rewards = []
    episode_portfolio_values = []
    
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        step_history = []  # Track all steps for detailed logging
        
        while True:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q_values.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Store step information for detailed logging
            # Extract key state features from the flattened observation
            # Get actual price from the data (before normalization)
            actual_price = env.data.iloc[env.current_step - 1]['close']
            current_rsi = state[6] * 100  # RSI (denormalized from position 6)
            current_pnl = (state[12] - 0.5) * 100  # Position PnL (denormalized from position 12)
            weekly_price_change = (state[16] + 1.0 - 1.0) * 100  # Weekly close price ratio converted to %
            
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
                'weekly_change': weekly_price_change
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
                
                # Log training loss occasionally
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
        
        # Log every episode to file
        portfolio_return = (info['portfolio_value'] - env.initial_balance) / env.initial_balance * 100
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
            
            # Also log summary stats
            logger.info(f"Episode {episode + 1} Summary: "
                       f"Avg Reward (100 eps) = {avg_reward:.4f}, "
                       f"Avg Portfolio (100 eps) = ${avg_portfolio:.2f}, "
                       f"Epsilon = {epsilon:.3f}")
            
            # Log detailed step history for this episode
            logger.info(f"\nDetailed steps for Episode {episode + 1}:")
            logger.info("Step | Action | Price    | RSI   | PnL    | WkChg  | Reward    | Portfolio   | Cash       | Crypto")
            logger.info("-" * 120)
            
            # Log first 20 and last 20 steps (or all if episode is shorter)
            if len(step_history) <= 40:
                steps_to_log = step_history
            else:
                steps_to_log = step_history[:20] + [{'step': '...', 'action': '...', 'price': '...', 'rsi': '...', 
                                                     'pnl': '...', 'weekly_change': '...', 'reward': '...', 'portfolio': '...', 
                                                     'cash': '...', 'crypto': '...'}] + step_history[-20:]
            
            action_names = {0: "HOLD", 1: "BUY ", 2: "SELL"}
            for step in steps_to_log:
                if step['step'] == '...':
                    logger.info("  ...    ...      ...       ...     ...       ...      ...         ...          ...         ...")
                else:
                    logger.info(f"{step['step']:4d} | {action_names.get(step['action'], str(step['action']))} | "
                               f"{step['price']:8.2f} | {step['rsi']:5.1f} | {step['pnl']:6.2f}% | {step['weekly_change']:6.2f}% | "
                               f"{step['reward']:9.4f} | ${step['portfolio']:10.2f} | "
                               f"${step['cash']:10.2f} | {step['crypto']:8.6f}")
            
            # Log action distribution for this episode
            action_counts = [sum(1 for s in step_history if s['action'] == i) for i in range(3)]
            logger.info(f"\nAction distribution for Episode {episode + 1}:")
            logger.info(f"  HOLD: {action_counts[0]:4d} ({action_counts[0]/len(step_history)*100:5.1f}%)")
            logger.info(f"  BUY:  {action_counts[1]:4d} ({action_counts[1]/len(step_history)*100:5.1f}%)")
            logger.info(f"  SELL: {action_counts[2]:4d} ({action_counts[2]/len(step_history)*100:5.1f}%)")
            logger.info("-" * 120)
    
    return q_network, episode_rewards, episode_portfolio_values

def validate_model(model, env):
    model.eval()
    state = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = [env.portfolio_value]
    
    with torch.no_grad():
        while True:
            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))
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
    logger.info("STARTING DQN TRADING EXPERIMENT")
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
    train_env = TradingEnvironment(train_data)
    val_env = TradingEnvironment(val_data)
    
    # Train model
    print("\nTraining DQN...")
    model, train_rewards, train_portfolios = train_dqn(train_env,)
    
    # Save model
    model_dir = "../data/06_models/dqn"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dqn_trading_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_sizes': [128, 64, 32],
        'input_size': 26,  # 13 current + 13 weekly features
        'output_size': 3
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
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
    val_env_output = TradingEnvironment(val_data)
    state = val_env_output.reset()
    step = 0
    
    with torch.no_grad():
        while step < len(val_data) - 7:  # 7 days needed for weekly comparison
            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))
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
    output_dir = "../data/07_model_output/dqn"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.parquet")
    output_df.to_parquet(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Also save analytical base table with predictions merged
    val_data_with_predictions = val_data.iloc[7:7 + len(output_df)].copy()
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
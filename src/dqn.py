import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Dict
from tqdm import tqdm
import logging
from datetime import datetime
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_dir: str = None):
    """Setup logging configuration"""
    if log_dir is None:
        log_dir = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/logs/dqn"
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dqn_trading_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

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
    def __init__(self, data: pd.DataFrame, window_size: int = 21, initial_balance: float = 10000.0, 
                 logger: logging.Logger = None):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.buy_fee = 0.02
        self.sell_fee = 0.05
        self.avg_buy_price = 0.0  # Track average buy price for PnL calculation
        self.returns_window = 20  # Window for calculating Sharpe ratio
        self.risk_free_rate = 0.02 / 252  # 2% annual risk-free rate, daily
        self.logger = logger
        self.trade_history = []  # Track all trades
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_balance
        self.crypto_held = 0.0
        self.portfolio_value = self.initial_balance
        self.avg_buy_price = 0.0
        self.done = False
        self.portfolio_values = [self.initial_balance]  # Track portfolio values for Sharpe
        self.returns = []  # Track returns for Sharpe calculation
        self.trade_history = []  # Reset trade history for new episode
        self.step_count = 0
        
        if self.logger:
            self.logger.info(f"Episode reset - Initial balance: ${self.initial_balance:.2f}")
        
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
            position_pnl = -2.0  # Default value when no position
        
        features[:, 12] = position_pnl  # Same PnL value for entire window
        
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
        
        # Normalize PnL feature (clip to reasonable range and normalize)
        features[:, 12] = np.clip(features[:, 12], -50, 50) / 100.0 + 0.5  # Range [-50%, +50%] -> [0, 1]
        
        return features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise ValueError("Episode is done")
        
        current_price = self.data.iloc[self.current_step]['close']
        previous_portfolio_value = self.portfolio_value
        previous_cash = self.cash
        previous_crypto = self.crypto_held
        
        action_name = ["HOLD", "BUY", "SELL"][action]
        trade_executed = False
        trade_details = {}
        
        # Execute action
        if action == 1:  # BUY 25%
            buy_amount = self.cash / (current_price * (1 + self.buy_fee)) * 0.25
            if buy_amount > 0:
                cost = buy_amount * current_price * (1 + self.buy_fee)
                fees = buy_amount * current_price * self.buy_fee
                
                # Update average buy price
                if self.crypto_held > 0:
                    total_value = self.crypto_held * self.avg_buy_price + buy_amount * current_price
                    self.avg_buy_price = total_value / (self.crypto_held + buy_amount)
                else:
                    self.avg_buy_price = current_price
                
                self.cash -= cost
                self.crypto_held += buy_amount
                trade_executed = True
                
                trade_details = {
                    'action': 'BUY',
                    'amount': buy_amount,
                    'price': current_price,
                    'cost': cost,
                    'fees': fees,
                    'cash_before': previous_cash,
                    'cash_after': self.cash,
                    'crypto_before': previous_crypto,
                    'crypto_after': self.crypto_held
                }
                
        elif action == 2:  # SELL 25%
            sell_amount = self.crypto_held * 0.25
            if sell_amount > 0:
                proceeds = sell_amount * current_price * (1 - self.sell_fee)
                fees = sell_amount * current_price * self.sell_fee
                
                self.cash += proceeds
                self.crypto_held -= sell_amount
                trade_executed = True
                
                # Reset avg_buy_price if all crypto is sold
                if self.crypto_held < 1e-8:
                    self.crypto_held = 0.0
                    self.avg_buy_price = 0.0
                
                trade_details = {
                    'action': 'SELL',
                    'amount': sell_amount,
                    'price': current_price,
                    'proceeds': proceeds,
                    'fees': fees,
                    'cash_before': previous_cash,
                    'cash_after': self.cash,
                    'crypto_before': previous_crypto,
                    'crypto_after': self.crypto_held
                }
        
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
        if action == 2 and previous_crypto == 0:
            reward = -0.01
        
        # Log trade details
        if trade_executed and self.logger:
            self.logger.info(
                f"Step {self.step_count}: {trade_details['action']} - "
                f"Amount: {trade_details['amount']:.6f}, Price: ${current_price:.2f}, "
                f"Cash: ${previous_cash:.2f} -> ${self.cash:.2f}, "
                f"Crypto: {previous_crypto:.6f} -> {self.crypto_held:.6f}, "
                f"Portfolio: ${previous_portfolio_value:.2f} -> ${self.portfolio_value:.2f}, "
                f"Return: {current_return:.4f}, Reward: {reward:.4f}"
            )
        
        # Store trade history
        if trade_executed:
            trade_record = {
                'step': self.step_count,
                'date': self.data.iloc[self.current_step]['date'] if 'date' in self.data.columns else self.current_step,
                'price': current_price,
                'portfolio_before': previous_portfolio_value,
                'portfolio_after': self.portfolio_value,
                'return': current_return,
                'reward': reward,
                **trade_details
            }
            self.trade_history.append(trade_record)
        
        # Move to next step
        self.step_count += 1
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        next_observation = self._get_observation() if not self.done else np.zeros((self.window_size, 13))
        
        return next_observation, reward, self.done, {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'crypto_held': self.crypto_held,
            'action': action_name,
            'trade_executed': trade_executed
        }
    
    def get_episode_summary(self) -> Dict:
        """Get summary statistics for the completed episode"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'final_portfolio': self.portfolio_value,
                'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance * 100,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_fees': 0
            }
        
        buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        total_fees = sum(t.get('fees', 0) for t in self.trade_history)
        
        return {
            'total_trades': len(self.trade_history),
            'final_portfolio': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_fees': total_fees,
            'final_cash': self.cash,
            'final_crypto': self.crypto_held,
            'avg_buy_price': self.avg_buy_price
        }
    
    def log_episode_summary(self):
        """Log episode summary in table format"""
        if not self.logger:
            return
            
        summary = self.get_episode_summary()
        
        # Create summary table
        summary_data = [
            ["Metric", "Value"],
            ["Final Portfolio", f"${summary['final_portfolio']:.2f}"],
            ["Total Return", f"{summary['total_return']:.2f}%"],
            ["Final Cash", f"${summary['final_cash']:.2f}"],
            ["Final Crypto", f"{summary['final_crypto']:.6f}"],
            ["Total Trades", summary['total_trades']],
            ["Buy Trades", summary['buy_trades']],
            ["Sell Trades", summary['sell_trades']],
            ["Total Fees", f"${summary['total_fees']:.2f}"],
            ["Avg Buy Price", f"${summary['avg_buy_price']:.2f}"]
        ]
        
        self.logger.info("\n" + "="*50)
        self.logger.info("EPISODE SUMMARY")
        self.logger.info("="*50)
        self.logger.info("\n" + tabulate(summary_data, headers="firstrow", tablefmt="grid"))
        
        # Log trade history table if there are trades
        if self.trade_history:
            trade_data = [["Step", "Action", "Amount", "Price", "Cash After", "Crypto After", "Portfolio", "Return%"]]
            for trade in self.trade_history[-10:]:  # Last 10 trades
                trade_data.append([
                    trade['step'],
                    trade['action'],
                    f"{trade['amount']:.6f}",
                    f"${trade['price']:.2f}",
                    f"${trade['cash_after']:.2f}",
                    f"{trade['crypto_after']:.6f}",
                    f"${trade['portfolio_after']:.2f}",
                    f"{trade['return']*100:.2f}%"
                ])
            
            self.logger.info("\nLAST 10 TRADES:")
            self.logger.info(tabulate(trade_data, headers="firstrow", tablefmt="grid"))
        
        self.logger.info("="*50 + "\n")

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
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=None,
              learning_rate=0.0001, gamma=0.99, target_update_freq=2000,
              train_freq=4, min_buffer_size=10000, hidden_sizes=[512, 256, 128],
              logger=None):
    
    input_size = env.window_size * 13
    output_size = 3
    
    q_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Calculate epsilon decay steps based on episodes if not provided
    if epsilon_decay_steps is None:
        # Estimate steps per episode (roughly length of data minus window size)
        estimated_steps_per_episode = len(env.data) - env.window_size
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
        
        # Log episode summary every 10 episodes or for first few episodes
        if logger and ((episode + 1) % 10 == 0 or episode < 5):
            env.log_episode_summary()
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            log_msg = (f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, "
                      f"Avg Portfolio: ${avg_portfolio:.2f}, Epsilon: {epsilon:.3f}")
            print(log_msg)
            if logger:
                logger.info(log_msg)
    
    return q_network, episode_rewards, episode_portfolio_values

def validate_model(model, env, logger=None):
    model.eval()
    state = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = [env.portfolio_value]
    
    if logger:
        logger.info("="*60)
        logger.info("STARTING VALIDATION")
        logger.info("="*60)
    
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
    
    # Log validation summary
    if logger:
        env.log_episode_summary()
        
        # Create action distribution table
        action_counts = [actions_taken.count(i) for i in range(3)]
        action_data = [
            ["Action", "Count", "Percentage"],
            ["HOLD", action_counts[0], f"{action_counts[0]/len(actions_taken)*100:.1f}%"],
            ["BUY", action_counts[1], f"{action_counts[1]/len(actions_taken)*100:.1f}%"],
            ["SELL", action_counts[2], f"{action_counts[2]/len(actions_taken)*100:.1f}%"]
        ]
        
        logger.info("ACTION DISTRIBUTION:")
        logger.info(tabulate(action_data, headers="firstrow", tablefmt="grid"))
        logger.info("="*60)
    
    return {
        'total_reward': total_reward,
        'final_portfolio_value': portfolio_values[-1],
        'initial_portfolio_value': portfolio_values[0],
        'return_pct': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100,
        'num_trades': sum(1 for a in actions_taken if a != 0),
        'actions': actions_taken,
        'portfolio_values': portfolio_values,
        'trade_history': env.trade_history
    }

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting DQN Trading Experiment")
    
    # Load data
    data_path = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/04_feature/analytical_base_table_01.parquet"
    df = pd.read_parquet(data_path)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split data
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:].copy()
    
    logger.info(f"Training data: {len(train_data)} rows")
    logger.info(f"Validation data: {len(val_data)} rows")
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    
    # Create environments
    train_env = TradingEnvironment(train_data, logger=logger)
    val_env = TradingEnvironment(val_data, logger=logger)
    
    # Train model
    logger.info("Starting DQN training...")
    print("\nTraining DQN...")
    model, train_rewards, train_portfolios = train_dqn(train_env, logger=logger)
    
    # Save model
    model_dir = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/06_models/dqn"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dqn_trading_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_sizes': [512, 256, 128],
        'input_size': train_env.window_size * 13,
        'output_size': 3
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Validate model
    logger.info("Starting model validation...")
    print("\nValidating model...")
    val_results = validate_model(model, val_env, logger=logger)
    
    validation_msg = (f"Validation Return: {val_results['return_pct']:.2f}%, "
                     f"Trades: {val_results['num_trades']}, "
                     f"Final Portfolio: ${val_results['final_portfolio_value']:.2f}")
    print(validation_msg)
    logger.info(validation_msg)
    
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
    output_dir = "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/data/07_model_output/dqn"
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
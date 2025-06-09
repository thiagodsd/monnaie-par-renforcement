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
        self.avg_buy_price = 0.0
        self.total_invested = 0.0
        self.returns_window = 20
        self.risk_free_rate = 0.02 / 252
        self.reset()
        
    def reset(self):
        self.current_step = 7  # precisa de 7 dias de histórico para comparação semanal
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
        if self.current_step < 7:
            return np.zeros(32)
        
        current_data = self.data.iloc[self.current_step]
        week_ago_data = self.data.iloc[self.current_step - 7]
        
        features = np.zeros(32)
        
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
        
        # calcula lucro/prejuízo não realizado
        if self.crypto_held > 0 and self.avg_buy_price > 0:
            position_pnl = ((current_data['close'] - self.avg_buy_price) / self.avg_buy_price) * 100
        else:
            position_pnl = 0.0
        features[12] = position_pnl
        
        # razões semanais para capturar momentum
        features[13] = current_data['open'] / (week_ago_data['open'] + 1e-8)
        features[14] = current_data['high'] / (week_ago_data['high'] + 1e-8)
        features[15] = current_data['low'] / (week_ago_data['low'] + 1e-8)
        features[16] = current_data['close'] / (week_ago_data['close'] + 1e-8)
        features[17] = features[4] / (week_ago_data.get('ema_9', week_ago_data.get('sma_7', week_ago_data['close'])) + 1e-8)
        features[18] = features[5] / (week_ago_data.get('ema_21', week_ago_data.get('sma_21', week_ago_data['close'])) + 1e-8)
        
        features[19] = current_data['rsi'] - week_ago_data['rsi']
        
        features[20] = current_data['volatility'] / (week_ago_data['volatility'] + 1e-8)
        features[21] = current_data['volume'] / (week_ago_data['volume'] + 1e-8)
        
        features[22] = current_data['sp500_change'] - week_ago_data['sp500_change']
        features[23] = current_data['djia_change'] - week_ago_data['djia_change']
        
        features[24] = current_data['fng_value'] - week_ago_data['fng_value']
        
        features[25] = position_pnl
        
        # indicadores de viabilidade de ação
        min_buy_amount = 100
        can_buy = 1.0 if self.cash >= min_buy_amount else 0.0
        features[26] = can_buy
        
        can_sell = 1.0 if self.crypto_held > 1e-8 else 0.0
        features[27] = can_sell
        
        current_position_value = self.crypto_held * current_data['close']
        position_size_pct = (current_position_value / self.initial_balance) * 100
        features[28] = position_size_pct
        
        cash_pct = (self.cash / self.initial_balance) * 100
        features[29] = cash_pct
        
        total_invested_pct = (self.total_invested / self.initial_balance) * 100
        features[30] = total_invested_pct
        
        unrealized_pnl_dollars = current_position_value - self.total_invested
        features[31] = unrealized_pnl_dollars
        
        # normaliza preços juntos para preservar relações relativas
        price_features = features[0:6]
        price_min, price_max = price_features.min(), price_features.max()
        if price_max > price_min:
            features[0:6] = (features[0:6] - price_min) / (price_max - price_min)
        
        features[6] = features[6] / 100.0
        features[7] = features[7] / (features[3] + 1e-8)
        features[8] = features[8] / (features[8] + 1e-8) if features[8] > 0 else 0
        features[9:11] = np.clip(features[9:11], -0.1, 0.1) / 0.2 + 0.5
        features[11] = features[11] / 100.0
        features[12] = np.clip(features[12], -50, 50) / 100.0 + 0.5
        
        features[13:19] = np.clip(features[13:19], 0.5, 2.0) - 1.0
        features[19] = np.clip(features[19], -50, 50) / 100.0
        features[20:22] = np.clip(features[20:22], 0.5, 2.0) - 1.0
        features[22:25] = np.clip(features[22:25], -0.2, 0.2) / 0.4 + 0.5
        features[25] = np.clip(features[25], -50, 50) / 100.0 + 0.5
        
        
        features[28] = np.clip(features[28], 0, 200) / 200.0
        features[29] = np.clip(features[29], 0, 200) / 200.0
        features[30] = np.clip(features[30], 0, 200) / 200.0
        features[31] = np.clip(features[31], -5000, 5000) / 10000.0 + 0.5
        
        return features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise ValueError("Episode is done")
        
        current_price = self.data.iloc[self.current_step]['close']
        previous_portfolio_value = self.portfolio_value
        
        if action == 1: # compra 25% do dinheiro disponível
            buy_amount = self.cash / (current_price * (1 + self.buy_fee)) * 0.25
            if buy_amount > 0:
                cost = buy_amount * current_price * (1 + self.buy_fee)
                # atualiza preço médio de compra ponderado
                if self.crypto_held > 0:
                    total_value = self.crypto_held * self.avg_buy_price + buy_amount * current_price
                    self.avg_buy_price = total_value / (self.crypto_held + buy_amount)
                else:
                    self.avg_buy_price = current_price
                
                self.total_invested += cost
                
                self.cash -= cost
                self.crypto_held += buy_amount
                
        elif action == 2: # vende 25% das posições
            sell_amount = self.crypto_held * 0.25
            if sell_amount > 0:
                self.cash += sell_amount * current_price * (1 - self.sell_fee)
                
                # reduz valor investido proporcionalmente
                if self.crypto_held > 0:
                    proportion_sold = sell_amount / self.crypto_held
                    self.total_invested *= (1 - proportion_sold)
                
                self.crypto_held -= sell_amount
                if self.crypto_held == 0:
                    self.avg_buy_price = 0.0
                    self.total_invested = 0.0
        
        self.portfolio_value = self.cash + self.crypto_held * current_price
        
        current_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        self.returns.append(current_return)
        self.portfolio_values.append(self.portfolio_value)
        
        # recompensa baseada em retornos ajustados ao risco (índice sharpe)
        if len(self.returns) >= self.returns_window:
            recent_returns = self.returns[-self.returns_window:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 0:
                sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
                reward = sharpe_ratio * 0.01  # escala para aprendizado estável
            else:
                reward = (mean_return - self.risk_free_rate) * 0.1
        else:
            # usa retorno simples quando não há histórico suficiente
            reward = current_return
        
        # penaliza ações inválidas para guiar exploração
        min_buy_amount = 100
        
        if action == 1 and self.cash < min_buy_amount:
            reward = -0.1
            
        if action == 2 and self.crypto_held < 1e-8:
            reward = -0.1
        
        # limita recompensa para evitar explosão de gradiente
        original_reward = reward
        reward = np.clip(reward, -1.0, 1.0)
        
        # registra eventos de clipping (limitado para evitar spam)
        if original_reward != reward and hasattr(self, '_clip_warning_count'):
            if self._clip_warning_count < 10:
                logger.warning(f"Reward clipped: {original_reward:.4f} -> {reward:.4f}")
                self._clip_warning_count += 1
        elif not hasattr(self, '_clip_warning_count'):
            self._clip_warning_count = 0
        
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        
        next_observation = self._get_observation() if not self.done else np.zeros(32)
        
        return next_observation, reward, self.done, {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'crypto_held': self.crypto_held
        }


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        assert state.shape == (32,), f"State shape {state.shape} != (32,)"
        assert next_state.shape == (32,), f"Next state shape {next_state.shape} != (32,)"
        
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


def train_dqn(
        env, 
        episodes=5000, 
        batch_size=32, 
        buffer_size=100000, 
        epsilon_start=1.0,
        epsilon_end=0.01, 
        epsilon_decay_steps=None,
        learning_rate=0.0001, 
        gamma=0.99, 
        target_update_freq=2000,
        train_freq=4,
        min_buffer_size=10000,
        hidden_sizes=[128, 64, 32]
    ):
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
    
    input_size = 32
    output_size = 3
    
    q_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network = DQN(input_size, hidden_sizes, output_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    if epsilon_decay_steps is None:
        # calcula automaticamente período de exploração
        estimated_steps_per_episode = len(env.data) - 7
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
            # exploração epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q_values.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            actual_price = env.data.iloc[env.current_step - 1]['close']
            current_rsi = state[6] * 100
            current_pnl = (state[12] - 0.5) * 100
            weekly_price_change = (state[16] + 1.0 - 1.0) * 100
            can_buy = state[26]
            can_sell = state[27]
            position_size_pct = state[28] * 200
            cash_pct = state[29] * 200
            total_invested_pct = state[30] * 200
            unrealized_pnl_dollars = (state[31] - 0.5) * 10000
            
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
                'weekly_change': weekly_price_change,
                'can_buy': can_buy,
                'can_sell': can_sell,
                'position_size_pct': position_size_pct,
                'cash_pct': cash_pct,
                'total_invested_pct': total_invested_pct,
                'unrealized_pnl_dollars': unrealized_pnl_dollars
            }
            step_history.append(step_info)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if epsilon > epsilon_end:
                epsilon -= epsilon_decay
            
            # treina com batch do buffer de replay
            if total_steps % train_freq == 0 and len(replay_buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
                
                # calcula td target
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if total_steps % 1000 == 0:
                    logger.info(f"Step {total_steps}: Loss = {loss.item():.6f}, Epsilon = {epsilon:.4f}")
            
            # sincronização periódica da rede alvo para estabilidade
            if total_steps % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
                logger.info(f"Step {total_steps}: Target network updated")
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(info['portfolio_value'])
        
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
        
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, "
                   f"Portfolio = ${info['portfolio_value']:.2f}, "
                   f"Return = {portfolio_return:.2f}%, "
                   f"Cash = ${info['cash']:.2f}, "
                   f"Crypto = {info['crypto_held']:.6f}, "
                   f"Steps = {episode_steps}")
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_portfolio = np.mean(episode_portfolio_values[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.4f}, "
                  f"Avg Portfolio: ${avg_portfolio:.2f}, Epsilon: {epsilon:.3f}")
            
            logger.info(f"Episode {episode + 1} Summary: "
                       f"Avg Reward (100 eps) = {avg_reward:.4f}, "
                       f"Avg Portfolio (100 eps) = ${avg_portfolio:.2f}, "
                       f"Epsilon = {epsilon:.3f}")
            
            logger.info(f"\nDetailed steps for Episode {episode + 1}:")
            logger.info("Step | Action | Price    | RSI   | PnL%   | WkChg% | PosSize% | Cash%  | Invested% | UnrlzdPnL$ | Reward    | Portfolio")
            logger.info("-" * 140)
            
            if len(step_history) <= 40:
                steps_to_log = step_history
            else:
                steps_to_log = step_history[:20] + [{'step': '...', 'action': '...', 'price': '...', 'rsi': '...', 
                                                     'pnl': '...', 'weekly_change': '...', 'position_size_pct': '...', 
                                                     'cash_pct': '...', 'total_invested_pct': '...', 'unrealized_pnl_dollars': '...', 
                                                     'reward': '...', 'portfolio': '...'}] + step_history[-20:]
            
            action_names = {0: "HOLD", 1: "BUY ", 2: "SELL"}
            for step in steps_to_log:
                if step['step'] == '...':
                    logger.info("  ...    ...      ...       ...     ...     ...     ...       ...      ...        ...         ...         ...")
                else:
                    logger.info(f"{step['step']:4d} | {action_names.get(step['action'], str(step['action']))} | "
                               f"{step['price']:8.2f} | {step['rsi']:5.1f} | {step['pnl']:6.2f} | {step['weekly_change']:6.2f} | "
                               f"{step['position_size_pct']:8.1f} | {step['cash_pct']:6.1f} | {step['total_invested_pct']:9.1f} | "
                               f"{step['unrealized_pnl_dollars']:10.2f} | {step['reward']:9.4f} | ${step['portfolio']:10.2f}")
            
            action_counts = [sum(1 for s in step_history if s['action'] == i) for i in range(3)]
            logger.info(f"\nAction distribution for Episode {episode + 1}:")
            logger.info(f"  HOLD: {action_counts[0]:4d} ({action_counts[0]/len(step_history)*100:5.1f}%)")
            logger.info(f"  BUY:  {action_counts[1]:4d} ({action_counts[1]/len(step_history)*100:5.1f}%)")
            logger.info(f"  SELL: {action_counts[2]:4d} ({action_counts[2]/len(step_history)*100:5.1f}%)")
            logger.info("-" * 120)
    
    metrics_df = pd.DataFrame(episode_metrics)
    metrics_dir = "../data/07_model_output/dqn"
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
    
    data_path = "../data/04_feature/analytical_base_table_01.parquet"
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data from: {data_path}")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size].copy()
    val_data = df.iloc[train_size:].copy()
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    print(f"Log file: {log_file}")
    
    logger.info(f"Data split: {len(train_data)} training rows, {len(val_data)} validation rows")
    logger.info(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")
    logger.info(f"Validation period: {val_data['date'].min()} to {val_data['date'].max()}")
    
    train_env = TradingEnvironment(train_data)
    val_env = TradingEnvironment(val_data)
    
    print("\nTraining DQN...")
    model, train_rewards, train_portfolios, train_metrics = train_dqn(train_env,)
    
    model_dir = "../data/06_models/dqn"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dqn_trading_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_sizes': [128, 64, 32],
        'input_size': 32,  # 13 current + 13 weekly + 2 binary controls + 4 position features
        'output_size': 3
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
  
    metrics_output_path = "../data/07_model_output/dqn/training_metrics.parquet"
    print(f"Training metrics saved to: {metrics_output_path}")
    print(f"Total training episodes: {len(train_metrics)}")
    
    final_portfolio = train_metrics[-1]['portfolio_value']
    final_return = train_metrics[-1]['portfolio_return_pct']
    best_episode = max(train_metrics, key=lambda x: x['portfolio_return_pct'])
    worst_episode = min(train_metrics, key=lambda x: x['portfolio_return_pct'])
    
    print("\nTraining Summary:")
    print(f"  Final Portfolio Value: ${final_portfolio:.2f}")
    print(f"  Final Return: {final_return:.2f}%")
    print(f"  Best Episode: {best_episode['episode']} (Return: {best_episode['portfolio_return_pct']:.2f}%)")
    print(f"  Worst Episode: {worst_episode['episode']} (Return: {worst_episode['portfolio_return_pct']:.2f}%)")
    
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
    
    print("\nValidating model...")
    logger.info("="*60)
    logger.info("STARTING VALIDATION")
    logger.info("="*60)
    
    val_results = validate_model(model, val_env)
    
    print(f"Validation Return: {val_results['return_pct']:.2f}%")
    print(f"Number of trades: {val_results['num_trades']}")
    print(f"Final portfolio value: ${val_results['final_portfolio_value']:.2f}")
    
    logger.info("Validation Results:")
    logger.info(f"  - Initial Portfolio: ${val_results['initial_portfolio_value']:.2f}")
    logger.info(f"  - Final Portfolio: ${val_results['final_portfolio_value']:.2f}")
    logger.info(f"  - Total Return: {val_results['return_pct']:.2f}%")
    logger.info(f"  - Total Reward: {val_results['total_reward']:.4f}")
    logger.info(f"  - Number of Trades: {val_results['num_trades']}")
    
    actions = val_results['actions']
    action_counts = [actions.count(0), actions.count(1), actions.count(2)]
    logger.info("  - Action Distribution:")
    logger.info(f"    - HOLD: {action_counts[0]} ({action_counts[0]/len(actions)*100:.1f}%)")
    logger.info(f"    - BUY:  {action_counts[1]} ({action_counts[1]/len(actions)*100:.1f}%)")
    logger.info(f"    - SELL: {action_counts[2]} ({action_counts[2]/len(actions)*100:.1f}%)")
    
    model.eval()
    output_data = []
    
    val_env_output = TradingEnvironment(val_data)
    state = val_env_output.reset()
    step = 0
    
    with torch.no_grad():
        while step < len(val_data) - 7:
            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))
            action = q_values.argmax().item()
            
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
    
    output_df = pd.DataFrame(output_data)
    output_dir = "../data/07_model_output/dqn"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.parquet")
    output_df.to_parquet(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
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
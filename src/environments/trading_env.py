"""
Trading environment optimized for price action position trading using only OHLC data.
"""
import numpy as np
import gymnasium as gym
import pandas as pd
import ta


class PriceActionTradingEnv(gym.Env):
    """Trading environment optimized for price action position trading"""
    def __init__(self, df, initial_balance=1000, window_size=20):
        super().__init__()
        self.df = self.add_technical_indicators(df)
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        
        # Action space: [HOLD, BUY, SELL]
        self.action_space = gym.spaces.Discrete(3)
        
        # Define features per time step: OHLC + technical indicators
        self.features_per_step = 8  # OHLC (4) + EMAs (2) + RSI + ATR
        
        # Observation space has shape (window_size, features_per_step)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(window_size, self.features_per_step),
            dtype=np.float32
        )
        
        # Trading history for visualization
        self.trades = []
        self.position_size = 0.4  # Use 40% of balance per position
        self.trade_fee = 0.001    # 0.1% trading fee
        self.holding_position = False  # Track if we're in a position
        
        self.reset()
    
    def add_technical_indicators(self, df):
        """Add technical indicators for price action analysis"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # EMA - Exponential Moving Averages
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        
        # RSI - Relative Strength Index
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR - Average True Range for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Fill NaN values that may have been created
        df = df.fillna(method='bfill').fillna(0)
        
        return df
        
    def reset(self):
        # Cash balance (actual money not invested)
        self.cash_balance = self.initial_balance
        # Asset holdings
        self.btc_held = 0
        self.asset_value = 0
        # Total portfolio value (cash + assets)
        self.net_worth = self.initial_balance
        
        self.current_step = self.window_size
        self.trades = []  # Reset trade history
        self.holding_position = False
        self.last_action_step = 0
        
        # Handle both old and new Gym API
        try:
            return self._next_observation(), {}  # New Gym API (obs, info)
        except Exception:
            return self._next_observation()  # Old Gym API (just obs)
        
    def _next_observation(self):
        """
        Get a window of price data observations with proper normalization
        Returns a 2D array of shape (window_size, features_per_step)
        """
        # Get the window of data (last window_size bars)
        window_slice = self.df.iloc[self.current_step - self.window_size:self.current_step]
        
        # Initialize the observation matrix
        obs = np.zeros((self.window_size, self.features_per_step), dtype=np.float32)
        
        # Calculate normalization factors for this window
        price_min = window_slice[['open', 'high', 'low', 'close']].min().min()
        price_max = window_slice[['open', 'high', 'low', 'close']].max().max()
        price_range = max(price_max - price_min, 1e-5)  # Avoid division by zero
        
        # For each time step in the window
        for i in range(self.window_size):
            price_data = window_slice.iloc[i]
            
            # Normalize OHLC values to [0,1]
            obs[i, 0] = (price_data['open'] - price_min) / price_range
            obs[i, 1] = (price_data['high'] - price_min) / price_range
            obs[i, 2] = (price_data['low'] - price_min) / price_range
            obs[i, 3] = (price_data['close'] - price_min) / price_range
            
            # Normalize EMAs
            obs[i, 4] = (price_data['ema_9'] - price_min) / price_range
            obs[i, 5] = (price_data['ema_21'] - price_min) / price_range
            
            # RSI is already in [0,100] range, normalize to [0,1]
            obs[i, 6] = price_data['rsi'] / 100.0
            
            # Normalize ATR (as percentage of close price)
            obs[i, 7] = min(price_data['atr'] / price_data['close'], 1.0) if price_data['close'] > 0 else 0
        
        return obs
        
    def step(self, action):
        self.current_step += 1
        
        current_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Update current asset value
        self.asset_value = self.btc_held * current_price
        
        # Apply position trading logic with fees and larger position sizes
        steps_since_last_action = self.current_step - self.last_action_step
                
        if action == 1 and not self.holding_position and self.cash_balance > 0:  # BUY
            # Calculate how much cash to use for this position
            position_cash = self.cash_balance * self.position_size
            # Deduct trade fee
            cost = position_cash * (1 - self.trade_fee)
            # Calculate BTC amount bought
            btc_bought = cost / current_price
            
            # Update account state
            self.btc_held += btc_bought
            self.cash_balance -= position_cash
            self.asset_value = self.btc_held * current_price
            self.holding_position = True
            self.last_action_step = self.current_step
            
            # Record buy trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'buy',
                'amount': btc_bought,
                'value': position_cash
            })
            
        elif action == 2 and self.holding_position and self.btc_held > 0:  # SELL
            # Sell entire position
            btc_sold = self.btc_held
            # Calculate total proceeds including fee
            proceeds = btc_sold * current_price * (1 - self.trade_fee)
            
            # Update account state
            self.cash_balance += proceeds
            self.btc_held = 0
            self.asset_value = 0
            self.holding_position = False
            self.last_action_step = self.current_step
            
            # Record sell trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'sell',
                'amount': btc_sold,
                'value': proceeds
            })
        
        # Calculate current portfolio value
        self.net_worth = self.cash_balance + self.asset_value
        
        # Enhanced reward function for position trading
        if prev_net_worth == 0:
            reward = 0
        else:
            returns = (self.net_worth - prev_net_worth) / prev_net_worth
            
            # Scale returns into a reward
            reward = returns * 100  # Base reward is percent return
            
            # Penalize frequent trading
            if action > 0 and steps_since_last_action < 5:  # If trade and last trade was recent
                reward -= 0.5  # Penalty for frequent trading
                
            # Reward for successful position trades
            if action == 2 and returns > 0:  # Selling at a profit
                reward *= 1.5  # Bonus for profitable exits
                
            # Reduce excessive trading frequency by small penalty for any action
            if action > 0:
                reward -= 0.1  # Small cost for any trade
        
        # Check if done
        done = self.net_worth <= 0 or self.current_step >= len(self.df)-1
        
        # Add account info to info dict for logging
        info = {
            "cash_balance": self.cash_balance,
            "asset_value": self.asset_value,
            "net_worth": self.net_worth,
            "btc_held": self.btc_held,
            "price": current_price
        }
        
        # Gym API can be inconsistent, so handle both old and new versions
        try:
            return self._next_observation(), reward, done, False, info  # New Gym API (done, truncated, info)
        except Exception:
            return self._next_observation(), reward, done, info  # Old Gym API (done, info)
    
    def close(self):
        """Clean up resources"""
        pass
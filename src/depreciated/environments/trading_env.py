"""
Trading environment optimized for price action position trading with advanced technical indicators.
"""
import numpy as np
import gymnasium as gym
import pandas as pd
import ta
import scipy.stats
from sklearn.linear_model import LinearRegression


class PriceActionTradingEnv(gym.Env):
    """Trading environment optimized for price action position trading"""
    def __init__(self, df, initial_balance=1000, window_size=20):
        super().__init__()
        self.df = self.add_technical_indicators(df)
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        
        # Action space: [HOLD, BUY_25PCT, BUY_50PCT, BUY_75PCT, BUY_100PCT, SELL_25PCT, SELL_50PCT, SELL_75PCT, SELL_100PCT]
        self.action_space = gym.spaces.Discrete(9)
        
        # Define features per time step: OHLC + basic technical indicators
        self.features_per_step = 10  # OHLC (4) + EMAs (3) + RSI + ATR + above_ema
        
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
        """Add basic technical indicators for trading analysis"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # EMA - Exponential Moving Averages
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        # 7-week EMA (assuming daily data, 7 weeks = ~35 trading days)
        df['ema_7w'] = ta.trend.ema_indicator(df['close'], window=35)
        df['above_7w_ema'] = (df['close'] > df['ema_7w']).astype(int)
        
        # RSI - Relative Strength Index
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR - Average True Range for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # ATR Percent (ATR as percentage of close price)
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Fill NaN values that may have been created
        df = df.bfill().fillna(0)
        
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
            
            feature_idx = 0
            
            # SECTION 1: BASIC PRICE DATA
            # Normalize OHLC values to [0,1]
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['open'] - price_min) / price_range
                feature_idx += 1
            
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['high'] - price_min) / price_range
                feature_idx += 1
            
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['low'] - price_min) / price_range
                feature_idx += 1
            
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['close'] - price_min) / price_range
                feature_idx += 1
            
            # SECTION 2: BASIC TECHNICAL INDICATORS
            
            # Normalize EMAs
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['ema_9'] - price_min) / price_range
                feature_idx += 1
            
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['ema_21'] - price_min) / price_range
                feature_idx += 1
            
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['ema_50'] - price_min) / price_range
                feature_idx += 1
            
            # RSI is already in [0,100] range, normalize to [0,1]
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = price_data['rsi'] / 100.0
                feature_idx += 1
            
            # ATR (as percentage of close price)
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = min(price_data['atr_percent'] / 10.0, 1.0)  # Cap at 10% for normalization
                feature_idx += 1
            
            # Above/Below weekly EMA (trend direction)
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = price_data['above_7w_ema']
                feature_idx += 1
            
        return obs
        
    def step(self, action):
        self.current_step += 1
        
        current_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Update current asset value
        self.asset_value = self.btc_held * current_price
        
        # Apply position trading logic with fees and variable position sizes
        steps_since_last_action = self.current_step - self.last_action_step
        
        # BUY actions (1-4) with different position sizes
        if 1 <= action <= 4 and self.cash_balance > 0:  # BUY with different percentages
            # Calculate position size based on action
            position_size_pct = {
                1: 0.25,  # 25% of available cash
                2: 0.50,  # 50% of available cash
                3: 0.75,  # 75% of available cash
                4: 1.00,  # 100% of available cash
            }[action]
            
            # Calculate how much cash to use for this position
            position_cash = self.cash_balance * position_size_pct
            
            # Deduct trade fee
            cost = position_cash * (1 - self.trade_fee)
            
            # Calculate BTC amount bought
            btc_bought = cost / current_price
            
            # Update account state
            self.btc_held += btc_bought
            self.cash_balance -= position_cash
            self.asset_value = self.btc_held * current_price
            self.holding_position = True if self.btc_held > 0 else False
            self.last_action_step = self.current_step
            
            # Record buy trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'buy',
                'pct': position_size_pct * 100,
                'amount': btc_bought,
                'value': position_cash
            })
            
        # SELL actions (5-8) with different position sizes
        elif 5 <= action <= 8 and self.btc_held > 0:  # SELL with different percentages
            # Calculate position size based on action
            position_size_pct = {
                5: 0.25,  # Sell 25% of holdings
                6: 0.50,  # Sell 50% of holdings
                7: 0.75,  # Sell 75% of holdings
                8: 1.00,  # Sell 100% of holdings (all)
            }[action]
            
            # Calculate amount to sell
            btc_sold = self.btc_held * position_size_pct
            
            # Calculate total proceeds including fee
            proceeds = btc_sold * current_price * (1 - self.trade_fee)
            
            # Update account state
            self.cash_balance += proceeds
            self.btc_held -= btc_sold
            self.asset_value = self.btc_held * current_price
            self.holding_position = True if self.btc_held > 0 else False
            self.last_action_step = self.current_step
            
            # Record sell trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'type': 'sell',
                'pct': position_size_pct * 100,
                'amount': btc_sold,
                'value': proceeds
            })
        
        # Calculate current portfolio value
        self.net_worth = self.cash_balance + self.asset_value
        
        # Simplified reward function focusing on returns and trading discipline
        if prev_net_worth == 0:
            reward = 0
        else:
            # Base reward is percent return
            returns = (self.net_worth - prev_net_worth) / prev_net_worth
            reward = returns * 100  # Scale to percentage points for clearer rewards
            
            # Get current technical indicator values
            current_data = self.df.iloc[self.current_step]
            
            # --- Trading discipline rewards ---
            
            # Encourage trend following using the 7-week EMA
            above_ema = current_data['above_7w_ema'] == 1
            
            # Action-specific trend reward
            if 1 <= action <= 4:  # BUY actions
                # Bonus for buying in uptrends (price above 7-week EMA)
                if above_ema:
                    reward += 0.2
                # Penalty for buying in downtrends
                else:
                    reward -= 0.2
                    
            elif 5 <= action <= 8:  # SELL actions
                # Bonus for selling in downtrends (price below 7-week EMA)
                if not above_ema:
                    reward += 0.2
                # Penalty for selling in uptrends
                else:
                    reward -= 0.2
            
            # Bonus for profitable exits
            if 5 <= action <= 8 and returns > 0:  # SELL actions with profit
                # Larger bonus for larger position sizes
                sell_bonus = {5: 0.2, 6: 0.3, 7: 0.4, 8: 0.5}
                reward += sell_bonus[action]
            
            # Penalize frequent trading
            if action > 0 and steps_since_last_action < 5:
                reward -= 0.5
            
            # Scale transaction costs based on position size
            if action > 0:
                # Base transaction cost + position size-based cost
                position_cost = {
                    1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20,  # BUY costs
                    5: 0.05, 6: 0.10, 7: 0.15, 8: 0.20   # SELL costs
                }
                reward -= position_cost[action]
        
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
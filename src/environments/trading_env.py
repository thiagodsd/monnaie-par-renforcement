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
        
        # Action space: [HOLD, BUY, SELL]
        self.action_space = gym.spaces.Discrete(3)
        
        # Define features per time step: OHLC + technical indicators
        self.features_per_step = 25  # OHLC (4) + Basic indicators (6) + Advanced indicators (15)
        
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
        """Add technical indicators for position trading analysis"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # --- Basic Indicators ---
        
        # EMA - Exponential Moving Averages
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
        
        # 7-week EMA (assuming daily data, 7 weeks = ~35 trading days)
        df['ema_7w'] = ta.trend.ema_indicator(df['close'], window=35)
        df['above_7w_ema'] = (df['close'] > df['ema_7w']).astype(int)
        
        # RSI - Relative Strength Index
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR - Average True Range for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # --- 1. Trendlines ---
        
        # ADX - Average Directional Index (14 and 50 periods)
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx_14'] = adx_indicator.adx()
        df['plus_di_14'] = adx_indicator.adx_pos()
        df['minus_di_14'] = adx_indicator.adx_neg()
        
        adx_50 = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=50)
        df['adx_50'] = adx_50.adx()
        
        # Initialize Linear Regression Slope columns with float data type
        df['lr_slope_50'] = 0.0
        df['lr_slope_50_norm'] = 0.0
        
        # Linear Regression Slope (50-period)
        if len(df) >= 50:  # Make sure we have enough data points
            try:
                for i in range(len(df) - 50):
                    window = df.iloc[i:i+50]
                    if len(window) == 50:  # Ensure we have a complete window
                        y = window['close'].values.reshape(-1, 1)
                        X = np.arange(len(y)).reshape(-1, 1)
                        model = LinearRegression().fit(X, y)
                        df.loc[df.index[i+49], 'lr_slope_50'] = model.coef_[0][0]
                        # Normalize slope by average price to get comparable values
                        if window['close'].mean() > 0:  # Avoid division by zero
                            df.loc[df.index[i+49], 'lr_slope_50_norm'] = model.coef_[0][0] / window['close'].mean()
            except:
                # If there's any issue with Linear Regression, we'll keep zeros
                pass
        
        # --- 2. Trend Channels ---
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        # Bollinger Bands Width as indicator of volatility/consolidation
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20, window_atr=10)
        df['kc_upper'] = keltner.keltner_channel_hband()
        df['kc_lower'] = keltner.keltner_channel_lband()
        df['kc_middle'] = keltner.keltner_channel_mband()
        # Keltner Channel Position
        df['kc_position'] = (df['close'] - df['kc_middle']) / ((df['kc_upper'] - df['kc_lower']) / 2)
        
        # --- 3. Pullbacks ---
        
        # Initialize Fibonacci columns
        df['at_fib_38'] = 0
        df['at_fib_50'] = 0
        df['at_fib_62'] = 0
        df['at_fib_level'] = 0
        
        # Fibonacci Retracement Level Check
        # We need a rolling window approach for this
        window_size = 50  # Look back 50 candles for swing highs/lows
        
        # Make sure we have enough data points
        if len(df) > window_size:
            for i in range(window_size, len(df)):
                try:
                    window = df.iloc[i-window_size:i]
                    swing_high = window['high'].max()
                    swing_low = window['low'].min()
                    
                    # Calculate Fibonacci levels
                    range_size = swing_high - swing_low
                    if range_size > 0:  # Avoid division by zero
                        fib_38 = swing_high - 0.382 * range_size
                        fib_50 = swing_high - 0.5 * range_size
                        fib_62 = swing_high - 0.618 * range_size
                        
                        # Check if current price is near any Fib level
                        current_price = df.iloc[i]['close']
                        tolerance = 0.015 * range_size  # 1.5% tolerance
                        
                        df.loc[df.index[i], 'at_fib_38'] = 1 if abs(current_price - fib_38) < tolerance else 0
                        df.loc[df.index[i], 'at_fib_50'] = 1 if abs(current_price - fib_50) < tolerance else 0
                        df.loc[df.index[i], 'at_fib_62'] = 1 if abs(current_price - fib_62) < tolerance else 0
                        
                        # Combined Fibonacci indicator (whether we're at any key level)
                        df.loc[df.index[i], 'at_fib_level'] = 1 if (df.loc[df.index[i], 'at_fib_38'] == 1 or 
                                                                    df.loc[df.index[i], 'at_fib_50'] == 1 or 
                                                                    df.loc[df.index[i], 'at_fib_62'] == 1) else 0
                except:
                    # Skip if any error occurs
                    pass
        
        # --- 4. Trading Ranges ---
        
        # ATR Percent (ATR as percentage of close price)
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Initialize Bandwidth Ratio column with float data type
        df['bandwidth_ratio'] = 0.0
        
        # Bandwidth Ratio (20-day range to 100-day range)
        if len(df) > 100:  # Make sure we have enough data points
            try:
                for i in range(100, len(df)):
                    short_window = df.iloc[i-20:i]
                    long_window = df.iloc[i-100:i]
                    short_range = short_window['high'].max() - short_window['low'].min()
                    long_range = long_window['high'].max() - long_window['low'].min()
                    df.loc[df.index[i], 'bandwidth_ratio'] = short_range / long_range if long_range > 0 else 0
            except:
                # Skip if any error occurs
                pass
        
        # --- 5. Breakouts ---
        
        # Volume Ratio (Current volume to 20-day average)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Initialize Price Range Position column with float data type
        df['price_range_position'] = 0.0
        
        # Price Distance from Range
        if len(df) > 20:  # Make sure we have enough data points
            try:
                for i in range(20, len(df)):
                    window = df.iloc[i-20:i]
                    high = window['high'].max()
                    low = window['low'].min()
                    mid = (high + low) / 2
                    range_size = high - low
                    df.loc[df.index[i], 'price_range_position'] = (df.iloc[i]['close'] - mid) / (range_size / 2) if range_size > 0 else 0
            except:
                # Skip if any error occurs
                pass
        
        # --- 6. Magnets (Price Attraction Levels) ---
        
        # Distance to Major MAs (50 and 200-day)
        df['dist_to_ma50'] = (df['close'] - df['ema_50']) / df['close'] * 100
        df['dist_to_ma200'] = (df['close'] - df['ema_200']) / df['close'] * 100
        
        # VWAP Deviation
        # First calculate VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['atr']
        
        # --- 7. Trend Reversals ---
        
        # Elder Force Index (13-period)
        df['force_index'] = (df['close'] - df['close'].shift(1)) * df['volume']
        df['force_index_ema13'] = ta.trend.ema_indicator(df['force_index'], window=13)
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # MACD Histogram Divergence (detect divergence between MACD histogram and price)
        # Initialize the divergence columns
        df['macd_bearish_div'] = 0
        df['macd_bullish_div'] = 0
        
        lookback = 14
        # Ensure we don't go out of bounds by stopping 1 candle before the end
        for i in range(lookback, len(df) - 1):
            # Check if we have a local price high (price peak)
            if df.iloc[i]['close'] > df.iloc[i-1]['close'] and df.iloc[i]['close'] > df.iloc[i+1]['close']:
                try:
                    # Check if MACD histogram is lower than previous peak
                    window = df.iloc[i-lookback:i+1]
                    # Use skipna=True to avoid warnings and check if there are non-NA values
                    if not window['macd_hist'].isna().all():
                        hist_max_idx = window['macd_hist'].dropna().idxmax() if not window['macd_hist'].dropna().empty else None
                        price_max_idx = window['close'].dropna().idxmax() if not window['close'].dropna().empty else None
                        
                        if hist_max_idx is not None and price_max_idx is not None and hist_max_idx < price_max_idx:
                            df.loc[df.index[i], 'macd_bearish_div'] = 1
                except:
                    # Skip if any error occurs
                    pass
            
            # Check if we have a local price low (price trough)
            if df.iloc[i]['close'] < df.iloc[i-1]['close'] and df.iloc[i]['close'] < df.iloc[i+1]['close']:
                try:
                    # Check if MACD histogram is higher than previous trough
                    window = df.iloc[i-lookback:i+1]
                    # Use skipna=True to avoid warnings and check if there are non-NA values
                    if not window['macd_hist'].isna().all():
                        hist_min_idx = window['macd_hist'].dropna().idxmin() if not window['macd_hist'].dropna().empty else None
                        price_min_idx = window['close'].dropna().idxmin() if not window['close'].dropna().empty else None
                        
                        if hist_min_idx is not None and price_min_idx is not None and hist_min_idx < price_min_idx:
                            df.loc[df.index[i], 'macd_bullish_div'] = 1
                except:
                    # Skip if any error occurs
                    pass
                    
        # --- 8. Minor Reversal Failures ---
        
        # False Breakout Detection
        # Initialize false breakout column
        df['false_breakout'] = 0
        
        # Make sure we have enough data points and don't go out of bounds
        if len(df) > 6:  # Need at least 5 periods + current + next
            for i in range(5, len(df) - 1):
                try:
                    # Previous high
                    prev_high = df.iloc[i-5:i]['high'].max()
                    atr_value = df.iloc[i]['atr']
                    
                    # Check if we break above previous high
                    if df.iloc[i]['high'] > prev_high + 0.5 * atr_value:
                        # Check if we close back below the breakout level
                        if df.iloc[i+1]['close'] < prev_high:
                            df.loc[df.index[i+1], 'false_breakout'] = 1
                except:
                    # Skip if any error occurs
                    pass
        
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
            # Normalize OHLC values to [0,1] (make sure we don't exceed feature size)
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
            
            # SECTION 2: BASIC TECHNICAL INDICATORS (make sure we don't exceed feature size)
            
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
            
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (price_data['ema_7w'] - price_min) / price_range
                feature_idx += 1
            
            # RSI is already in [0,100] range, normalize to [0,1]
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = price_data['rsi'] / 100.0
                feature_idx += 1
            
            # ATR (as percentage of close price)
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = min(price_data['atr_percent'] / 10.0, 1.0)  # Cap at 10% for normalization
                feature_idx += 1
            
            # SECTION 3: ADVANCED INDICATORS - TRENDLINES (make sure we don't exceed feature size)
            
            # ADX (Average Directional Index)
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = price_data['adx_14'] / 100.0  # ADX is in [0,100]
                feature_idx += 1
            
            # Directional indicators
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = price_data['plus_di_14'] / 100.0
                feature_idx += 1
            
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = price_data['minus_di_14'] / 100.0
                feature_idx += 1
            
            # Linear regression slope (normalized)
            if feature_idx < self.features_per_step:
                # We clip to [-1, 1] and then normalize to [0, 1]
                if 'lr_slope_50_norm' in price_data:
                    slope_norm = np.clip(price_data['lr_slope_50_norm'] * 100, -1, 1)
                    obs[i, feature_idx] = (slope_norm + 1) / 2  # Map from [-1,1] to [0,1]
                else:
                    obs[i, feature_idx] = 0.5  # Default to neutral if not available
                feature_idx += 1
            
            # SECTION 4: TREND CHANNELS (make sure we don't exceed feature size)
            
            # Bollinger Bands Width
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = min(price_data['bb_width'] / 0.2, 1.0)  # Normalize with 0.2 as reference
                feature_idx += 1
            
            # Keltner Channel Position
            if feature_idx < self.features_per_step:
                # Already in [-1, 1], normalize to [0, 1]
                obs[i, feature_idx] = (np.clip(price_data['kc_position'], -1, 1) + 1) / 2
                feature_idx += 1
            
            # SECTION 5: FIBONACCI LEVELS (make sure we don't exceed feature size)
            
            # At key Fibonacci level
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = price_data['at_fib_level']
                feature_idx += 1
            
            # SECTION 6: BREAKOUTS (make sure we don't exceed feature size)
            
            # Volume ratio (normalized)
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = min(price_data['volume_ratio'] / 3.0, 1.0)  # Cap at 3x normal volume
                feature_idx += 1
            
            # Price position in trading range
            if feature_idx < self.features_per_step:
                obs[i, feature_idx] = (np.clip(price_data['price_range_position'], -1, 1) + 1) / 2
                feature_idx += 1
            
            # SECTION 7: PRICE MAGNETS (make sure we don't exceed feature size)
            
            # Distance to 50MA (normalized)
            if feature_idx < self.features_per_step:
                dist_ma50 = price_data['dist_to_ma50']
                obs[i, feature_idx] = (np.clip(dist_ma50 / 10, -1, 1) + 1) / 2  # ±10% range
                feature_idx += 1
            
            # VWAP deviation
            if feature_idx < self.features_per_step:
                vwap_dev = price_data['vwap_deviation']
                obs[i, feature_idx] = (np.clip(vwap_dev, -2, 2) + 2) / 4  # ±2 ATR range
                feature_idx += 1
            
            # SECTION 8: REVERSALS (make sure we don't exceed feature size)
            
            # Elder Force Index (13-period)
            if feature_idx < self.features_per_step:
                # Normalize using the max absolute value in the window
                force_idx_max = window_slice['force_index_ema13'].abs().max()
                if force_idx_max > 0:
                    obs[i, feature_idx] = (np.clip(price_data['force_index_ema13'] / force_idx_max, -1, 1) + 1) / 2
                else:
                    obs[i, feature_idx] = 0.5
                feature_idx += 1
            
            # MACD Histogram 
            if feature_idx < self.features_per_step:
                # Normalize using the max absolute value in the window
                macd_hist_max = window_slice['macd_hist'].abs().max()
                if macd_hist_max > 0:
                    obs[i, feature_idx] = (np.clip(price_data['macd_hist'] / macd_hist_max, -1, 1) + 1) / 2
                else:
                    obs[i, feature_idx] = 0.5
                feature_idx += 1
            
            # MACD Bullish Divergence (make sure we don't exceed feature size)
            if feature_idx < self.features_per_step:
                if 'macd_bullish_div' in price_data:
                    obs[i, feature_idx] = price_data['macd_bullish_div']
                else:
                    obs[i, feature_idx] = 0
                feature_idx += 1
            
            # SECTION 9: FALSE BREAKOUTS (make sure we don't exceed feature size)
            
            # False breakout detection
            if feature_idx < self.features_per_step:
                if 'false_breakout' in price_data:
                    obs[i, feature_idx] = price_data['false_breakout']
                else:
                    obs[i, feature_idx] = 0
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
        
        # Enhanced reward function for position trading with technical indicator influence
        if prev_net_worth == 0:
            reward = 0
        else:
            # Base reward is percent return
            returns = (self.net_worth - prev_net_worth) / prev_net_worth
            reward = returns * 100
            
            # Get current technical indicator values for reward adjustments
            current_data = self.df.iloc[self.current_step]
            
            # --- Trend-based rewards ---
            
            # ADX trend strength
            adx_value = current_data['adx_14']
            trend_strength = adx_value / 100.0  # Normalize to [0,1]
            
            # Trend direction
            trend_direction = 1 if current_data['plus_di_14'] > current_data['minus_di_14'] else -1
            
            # --- Action-specific rewards ---
            
            if action == 1:  # BUY
                # Reward buying in strong uptrends
                if trend_direction > 0 and adx_value > 25:
                    reward *= (1.0 + 0.5 * trend_strength)
                    
                # Penalty for buying against the trend
                if trend_direction < 0 and adx_value > 25:
                    reward *= (1.0 - 0.3 * trend_strength)
                    
                # Bonus for buying at Fibonacci support levels
                if current_data['at_fib_level'] == 1:
                    reward += 0.5
                    
                # Bonus for buying at false breakout points (potential reversal)
                if 'false_breakout' in current_data and current_data['false_breakout'] == 1:
                    reward += 0.7
                    
                # Bonus for buying with bullish MACD divergence
                if 'macd_bullish_div' in current_data and current_data['macd_bullish_div'] == 1:
                    reward += 0.8
                
            elif action == 2:  # SELL
                # Reward selling in strong downtrends
                if trend_direction < 0 and adx_value > 25:
                    reward *= (1.0 + 0.5 * trend_strength)
                    
                # Penalty for selling against the trend
                if trend_direction > 0 and adx_value > 25:
                    reward *= (1.0 - 0.3 * trend_strength)
                
                # Bigger bonus for profitable exits
                if returns > 0:
                    reward *= 1.5
                    
                # Bonus for selling at price resistance levels
                if current_data['price_range_position'] > 0.8:  # Near the top of range
                    reward += 0.4
                    
                # Bonus for selling with bearish MACD divergence
                if 'macd_bearish_div' in current_data and current_data['macd_bearish_div'] == 1:
                    reward += 0.8
            
            # --- Trading discipline rewards ---
            
            # Penalize frequent trading
            if action > 0 and steps_since_last_action < 5:
                reward -= 0.5
                
            # Reward for respecting volatility
            if action > 0:  # Any trade
                # ATR percentage indicates volatility
                atr_percent = current_data['atr_percent']
                
                # If volatility is high (>3%), reduce trading frequency penalty
                if atr_percent > 3.0:
                    reward += 0.2
                    
                # If volatility is very low (<1%), increase trading frequency penalty
                if atr_percent < 1.0:
                    reward -= 0.3
                
            # Base transaction cost
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
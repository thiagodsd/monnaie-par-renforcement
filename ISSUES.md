(monnaie) dusoudeth@u22:~/Documentos/github/monnaie-par-renforcement/src$ python main.py 
Starting Position Trading Experiment with Price Action Analysis
Using GPU: NVIDIA GeForce GTX 1660 Ti
Using device: cuda
GPU Memory: 6.22 GB
Loaded data with shape (1863, 20)
Split data into training ((1490, 20)) and validation ((373, 20))
Using OHLC data: open, high, low, close, volume
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:89: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '-79.35405455150809' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  df.loc[df.index[i+49], 'lr_slope_50'] = model.coef_[0][0]
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:92: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '-0.010615387454529075' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  df.loc[df.index[i+49], 'lr_slope_50_norm'] = model.coef_[0][0] / window['close'].mean()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:174: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '0.26663948952991445' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  df.loc[df.index[i], 'bandwidth_ratio'] = short_range / long_range if long_range > 0 else 0
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:196: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '-1.1740858985526308' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
  df.loc[df.index[i], 'price_range_position'] = (df.iloc[i]['close'] - mid) / (range_size / 2) if range_size > 0 else 0
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:251: FutureWarning: The behavior of Series.idxmin with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_min_idx = window['macd_hist'].idxmin()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:237: FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_max_idx = window['macd_hist'].idxmax()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:251: FutureWarning: The behavior of Series.idxmin with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_min_idx = window['macd_hist'].idxmin()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:237: FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_max_idx = window['macd_hist'].idxmax()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:251: FutureWarning: The behavior of Series.idxmin with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_min_idx = window['macd_hist'].idxmin()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:237: FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_max_idx = window['macd_hist'].idxmax()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:251: FutureWarning: The behavior of Series.idxmin with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_min_idx = window['macd_hist'].idxmin()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:237: FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_max_idx = window['macd_hist'].idxmax()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:251: FutureWarning: The behavior of Series.idxmin with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_min_idx = window['macd_hist'].idxmin()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:237: FutureWarning: The behavior of Series.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError
  hist_max_idx = window['macd_hist'].idxmax()
/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py:284: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df = df.fillna(method='bfill').fillna(0)
Traceback (most recent call last):
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py", line 304, in reset
    return self._next_observation(), {}  # New Gym API (obs, info)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py", line 458, in _next_observation
    obs[i, feature_idx] = price_data['above_7w_ema']
    ~~~^^^^^^^^^^^^^^^^
IndexError: index 25 is out of bounds for axis 1 with size 25

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/main.py", line 90, in <module>
    main()
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/main.py", line 86, in main
    run_position_trading_experiment()
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/main.py", line 25, in run_position_trading_experiment
    env = PriceActionTradingEnv(
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py", line 40, in __init__
    self.reset()
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py", line 306, in reset
    return self._next_observation()  # Old Gym API (just obs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dusoudeth/Documentos/github/monnaie-par-renforcement/src/environments/trading_env.py", line 458, in _next_observation
    obs[i, feature_idx] = price_data['above_7w_ema']
    ~~~^^^^^^^^^^^^^^^^
IndexError: index 25 is out of bounds for axis 1 with size 25
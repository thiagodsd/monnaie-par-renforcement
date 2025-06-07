"""
Evaluation utilities for assessing agent performance.
"""
from typing import Tuple, List, Dict
import pandas as pd
from tqdm import tqdm

from environments.trading_env import PriceActionTradingEnv
from agents.base import Agent


def run_out_of_time_validation(agent: Agent, val_df: pd.DataFrame, 
                             window_size: int, initial_balance: float = 1000) -> Tuple[float, List[Dict]]:
    """Run the trained agent on out-of-time validation data"""
    val_env = PriceActionTradingEnv(val_df, initial_balance=initial_balance, window_size=window_size)
    reset_result = val_env.reset()
    
    # Handle both gym API versions
    if isinstance(reset_result, tuple):
        state, _ = reset_result  # New Gym API
    else:
        state = reset_result  # Old Gym API
        
    done = False
    step_count = 0
    total_steps = len(val_df) - window_size  # Estimate of total steps
    
    print("\nRunning out-of-time validation...")
    # Create a progress bar for validation steps
    progress_bar = tqdm(total=total_steps, desc="Validation", unit="step")
    
    while not done:
        # Always use exploitation (no exploration)
        action = agent.act(state, exploration=False)
        step_result = val_env.step(action)
        
        # Handle both gym API versions
        if len(step_result) == 5:  # New Gym API
            next_state, reward, done, truncated, info = step_result
            done = done or truncated
        else:  # Old Gym API
            next_state, reward, done, info = step_result
            
        state = next_state
        step_count += 1
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            'Portfolio': f"${val_env.net_worth:.2f}",
            'Cash': f"${val_env.cash_balance:.2f}",
            'Assets': f"${val_env.asset_value:.2f}"
        })
    
    # Close the progress bar
    progress_bar.close()
    
    # Calculate final profit
    profit = val_env.net_worth - val_env.initial_balance
    roi = (profit / val_env.initial_balance) * 100
    
    print(f"Validation Results:")
    print(f"  Final Portfolio: ${val_env.net_worth:.2f}")
    print(f"  Profit: ${profit:.2f} (ROI: {roi:.2f}%)")
    print(f"  Total Trades: {len(val_env.trades)}")
    
    return val_env.net_worth, val_env.trades
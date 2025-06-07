"""
Visualization utilities for displaying trading results.
"""
from typing import List, Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def plot_training_metrics(loss_history: List[float], portfolio_values: List[float], 
                          cash_balance: List[float] = None, asset_value: List[float] = None,
                          window_size: int = 10) -> None:
    """
    Plot training metrics in a 1x2 grid:
    1. Average loss vs epochs (smoothed)
    2. Portfolio composition over episodes
    
    Args:
        loss_history: List of loss values from training
        portfolio_values: List of portfolio values at the end of each episode
        cash_balance: Optional list of cash balances over time
        asset_value: Optional list of asset values over time
        window_size: Size of the moving average window for smoothing
    """
    if cash_balance and asset_value:
        # If we have cash and asset breakdowns, create a 2x1 grid
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    else:
        # Otherwise use a 1x2 grid
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Calculate moving averages for smoothing
    def moving_average(data, window):
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')
    
    # Plot 1: Average Loss vs Epochs
    if len(loss_history) > window_size:
        # Apply smoothing if we have enough data
        smooth_loss = moving_average(loss_history, window_size)
        epochs = range(window_size-1, len(loss_history))
        ax1.plot(epochs, smooth_loss, 'b-', linewidth=2)
    else:
        # Plot raw data if we don't have enough for smoothing
        epochs = range(len(loss_history))
        ax1.plot(epochs, loss_history, 'b-', linewidth=2)
    
    ax1.set_title(f'Average Loss vs Epochs (MA{window_size})')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Portfolio Composition (if we have the breakdown data)
    if cash_balance and asset_value:
        episodes = range(len(portfolio_values))
        ax2.stackplot(episodes, [cash_balance, asset_value], 
                     labels=['Cash', 'Assets'],
                     colors=['#3A7CA5', '#D9A5B3'], alpha=0.7)
        ax2.plot(episodes, portfolio_values, 'g-', linewidth=2, label='Total')
        ax2.set_title('Portfolio Composition vs Episodes')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Value ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
    else:
        # Simple portfolio value plot if no breakdown
        episodes = range(len(portfolio_values))
        ax2.plot(episodes, portfolio_values, 'g-', linewidth=2)
        ax2.set_title('Portfolio Value vs Episodes')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
    
    # Save figure
    plots_dir = Path("../data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plots_dir / 'training_metrics.png', dpi=300)
    plt.show()


def plot_trading_results(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       train_trades: List[Dict], val_trades: List[Dict], 
                       train_profit: float, val_profit: float) -> None:
    """Plot training and validation results side by side with buy/sell indicators"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # Plot training data and trades
    ax1.plot(train_df['close'].values, color='blue', alpha=0.6)
    ax1.set_title(f'Training Set - Profit: ${train_profit:.2f}')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Price')
    
    # Plot buy/sell points
    for trade in train_trades:
        if trade['type'] == 'buy':
            ax1.scatter(trade['step'] - train_df.index[0], trade['price'], 
                       marker='^', color='green', s=100, alpha=0.7)
        else:  # sell
            ax1.scatter(trade['step'] - train_df.index[0], trade['price'], 
                       marker='v', color='red', s=100, alpha=0.7)
    
    # Plot validation data and trades
    ax2.plot(val_df['close'].values, color='purple', alpha=0.6)
    ax2.set_title(f'Out-of-Time Validation - Profit: ${val_profit:.2f}')
    ax2.set_xlabel('Time Steps')
    
    # Plot buy/sell points
    for trade in val_trades:
        if trade['type'] == 'buy':
            ax2.scatter(trade['step'] - val_df.index[0], trade['price'], 
                       marker='^', color='green', s=100, alpha=0.7)
        else:  # sell
            ax2.scatter(trade['step'] - val_df.index[0], trade['price'], 
                       marker='v', color='red', s=100, alpha=0.7)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Training Price'),
        Line2D([0], [0], color='purple', lw=2, label='Validation Price'),
        Line2D([0], [0], marker='^', color='green', markersize=10, linestyle='None', label='Buy'),
        Line2D([0], [0], marker='v', color='red', markersize=10, linestyle='None', label='Sell')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    plots_dir = Path("../data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / 'trading_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
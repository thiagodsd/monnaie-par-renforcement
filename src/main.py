"""
Main entry point for the reinforcement learning trading application.
"""
import os
import torch
from pathlib import Path
from networks.base import PolicyNetwork
from networks.price_action import PriceActionQNetwork
from agents.position_dqn_agent import PositionDQNAgent
from environments.trading_env import PriceActionTradingEnv
from utils.data import load_data, get_device
from utils.visualization import plot_trading_results, plot_training_metrics
from utils.evaluation import run_out_of_time_validation


def run_position_trading_experiment(
        num_episodes: int = 50
    ) -> None:
    """
    Run the position trading experiment with price action analysis
    """
    train_df, val_df = load_data()
    window_size = 7
    initial_balance = 1000
    env = PriceActionTradingEnv(
        train_df,
        initial_balance=initial_balance,
        window_size=window_size
    )
    print(f"Environment created successfully! Action space: {env.action_space}, Initial balance: ${initial_balance}")
    
    # Check if we're using GPU to optimize hyperparameters
    device = get_device()
    is_gpu = device.type == 'cuda'
    
    # Adjust hyperparameters based on whether we're using GPU
    hyperparams = {
        "hidden_dim"      : 512 if is_gpu else 256,       # Larger network on GPU
        "lr"              : 0.00025,
        "buffer_size"     : 200000 if is_gpu else 100000, # Larger buffer on GPU
        "batch_size"      : 128 if is_gpu else 64,        # Larger batches on GPU
        "gamma"           : 0.99,
        "epsilon_start"   : 1.0,
        "epsilon_min"     : 0.05,
        "epsilon_decay"   : 0.995,
        "target_update"   : 250 if is_gpu else 500,       # More frequent updates on GPU
        "min_hold_period" : 10
    }
    agent = PositionDQNAgent(env, hyperparams)
    portfolio_values, loss_history, reward_history, _, _ = agent.train(num_episodes)
    
    # Plot training metrics (loss vs epochs and balance vs episodes)
    plot_training_metrics(loss_history, portfolio_values)
    
    # train
    train_profit = env.net_worth - env.initial_balance
    train_roi = (train_profit / env.initial_balance) * 100
    print("\nTraining complete!")
    print(f"  Final portfolio: ${env.net_worth:.2f}")
    print(f"  Profit: ${train_profit:.2f} (ROI: {train_roi:.2f}%)")
    print(f"  Total trades: {len(agent.train_trades)}")
    # out-of-time
    val_portfolio, val_trades = run_out_of_time_validation(agent, val_df, window_size, initial_balance)
    val_profit = val_portfolio - initial_balance
    # plot
    plot_trading_results(train_df, val_df, agent.train_trades, val_trades, train_profit, val_profit)


def main():
    """Main entry point for the application"""
    print("Starting Position Trading Experiment with Price Action Analysis")
    
    # Check for GPU
    device = get_device()
    print(f"Using device: {device}")
    
    # Set up PyTorch to use GPU for optimal performance
    if device.type == 'cuda':
        # Enable TensorFloat-32 for Ampere GPUs (if available)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Print memory info
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    run_position_trading_experiment()


if __name__ == "__main__":
    main()
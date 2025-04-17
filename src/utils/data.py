"""
Data loading and preprocessing utilities for OHLC price data.
"""
from typing import Tuple, Union
import pandas as pd
import torch


def get_device() -> torch.device:
    """
    Determine the device to use for PyTorch operations.
    Returns a torch.device object that can be used to move tensors and models.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead")
    
    return device


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess market data, focusing on OHLC data"""
    df = pd.read_parquet("../data/04_feature/analytical_base_table_01.parquet")
    print(f"Loaded data with shape {df.shape}")
    
    # Make sure required OHLC columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'date']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Sort by date and reset index
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split into training and validation sets (80/20)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"Split data into training ({train_df.shape}) and validation ({val_df.shape})")
    print(f"Using OHLC data: {', '.join(required_columns[:-1])}")
    
    return train_df, val_df
"""
Neural network implementations for price action analysis in trading.
"""
import torch
from torch import nn


class PriceActionQNetwork(nn.Module):
    """Q-Network optimized for price action analysis in trading environments"""
    def __init__(self, input_shape, hidden_dim: int, action_dim: int):
        super().__init__()
        
        # Extract dimensions from input shape (window_size, features_per_step)
        window_size, features_per_step = input_shape
        
        # Feature extraction using 1D convolutions along the time dimension
        self.conv1 = nn.Conv1d(in_channels=features_per_step, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        
        # Calculate the flattened size after convolutions
        # Window size will be reduced by (kernel_size - 1) * number of layers
        conv_output_size = window_size - (3-1)*3  # 3 conv layers with kernel size 3
        
        # Fully connected layers for Q-value prediction
        self.fc1 = nn.Linear(128 * conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Regularization
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input: [batch_size, window_size, features] -> [batch_size, features, window_size]
        # This puts the features in the channel dimension for 1D convolutions
        x = x.permute(0, 2, 1)
        
        # Apply convolutional layers with batch normalization
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = torch.relu(self.fc2(x))
        
        # Output layer - no activation
        return self.fc3(x)
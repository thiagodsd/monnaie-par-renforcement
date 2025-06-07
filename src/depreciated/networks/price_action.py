"""
Neural network implementations for price action analysis in trading.
"""
import torch
from torch import nn


class PriceActionQNetwork(nn.Module):
    """Q-Network optimized for price action analysis in trading environments (Pure MLP)"""
    def __init__(self, input_shape, hidden_dim: int, action_dim: int):
        super().__init__()
        
        # Extract dimensions from input shape (window_size, features_per_step)
        window_size, features_per_step = input_shape
        
        # Calculate flattened input size
        input_size = window_size * features_per_step
        
        # Create a fully-connected MLP architecture
        self.fc1 = nn.Linear(input_size, hidden_dim * 2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer_norm3 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input: [batch_size, window_size, features] -> [batch_size, window_size * features]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = self.layer_norm3(x)
        x = torch.relu(x)
        
        # Output layer - no activation
        return self.fc4(x)
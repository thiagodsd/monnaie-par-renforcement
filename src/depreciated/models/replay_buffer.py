"""
Replay buffer for experience replay in reinforcement learning.
"""
import random
from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
    """
    Memory buffer for experience replay in reinforcement learning agents.
    Stores transitions (state, action, reward, next_state, done) for training.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = list()
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences from the buffer"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_np = np.array(states)  # converting to numpy arrays first for efficiency
        next_states_np = np.array(next_states)
        return (
            torch.FloatTensor(states_np),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states_np),
            torch.FloatTensor(dones),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def ready(self) -> bool:
        """Check if buffer has enough data to start training"""
        return len(self.buffer) >= self.capacity // 10
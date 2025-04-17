"""
Base environment classes for reinforcement learning.
"""
from abc import ABC, abstractmethod


class Environment(ABC):
    """Abstract base class for custom environments"""
    @abstractmethod
    def reset(self):
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action):
        """Execute action and return new state, reward, done flag"""
        pass
    
    @abstractmethod
    def close(self):
        """Close environment and release resources"""
        pass
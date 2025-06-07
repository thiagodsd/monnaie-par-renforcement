"""
Base agent classes for reinforcement learning.
"""
from abc import ABC, abstractmethod
import gymnasium as gym


class Agent(ABC):
    """Abstract base class for reinforcement learning agents"""
    def __init__(self, env: gym.Env, hyperparams: dict):
        self.env = env
        self.hyperparams = hyperparams
        
    @abstractmethod
    def act(self, state, exploration: bool = True):
        """Select an action based on the current state"""
        pass

    @abstractmethod
    def update(self, batch):
        """Update agent's policy using a batch of experiences"""
        pass

    @abstractmethod
    def train(self, num_episodes: int):
        """Train agent for specified number of episodes"""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10):
        """Evaluate agent's performance without exploration"""
        pass
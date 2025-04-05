from abc import ABC
from typing import List, Optional, Union
import gymnasium as gym
import torch
from torch import nn, optim


class PolicyNetwork(nn.Module):
    """
    `todo`
    """
    def __init__(self, state_dim:int, hidden_dim:int, action_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        `todo`
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)


class ReplayBuffer:
    """
    `todo`
    """
    def __init__(self, capacity:int):
        pass
        # initialize buffer


class Agent:
    """
    `todo`
    """
    def __init__(self, env:gym.Env, hyperparams:dict):
        self.env = env
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam()
        self.buffer = ReplayBuffer()

    def act(self, state, exploration:bool=True):
        """
        `todo`
        """
        # select action based on policy

    def update(self, batch):
        """
        `todo`
        """
        # update policy using batch of experiences

    def train(self, num_episodes:int):
        """
        `todo`
        """
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                if self.buffer.ready():
                    batch = self.buffer.sample()
                    self.update(batch)
            self.env.close()


def main():
    """
    `todo`
    """
    pass


if __name__ == "__main__":
    main()
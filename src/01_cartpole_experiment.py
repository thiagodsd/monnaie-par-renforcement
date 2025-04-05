from abc import ABC
from typing import List, Optional, Union
import gymnasium as gym
import torch
from torch import nn, optim


class PolicyNetwork(nn.Module):
    """
    `todo`
    """
    def __init__(self, state_dim:int, action_dim:int, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class ReplayBuffer:
    """
    `todo`
    """
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return zip(*self.buffer)

    def ready(self):
        return len(self.buffer) >= self.capacity


class Agent:
    """
    `todo`
    """
    def __init__(self, env:gym.Env, hyperparams:dict):
        self.env = env
        self.hyperparams = hyperparams
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hyperparams['lr'])
        self.buffer = ReplayBuffer(hyperparams['buffer_capacity'])

    def act(self, state, exploration:bool=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        
        # Calculate discounted returns
        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.hyperparams['gamma'] * R
            discounted_returns.insert(0, R)
        discounted_returns = torch.FloatTensor(discounted_returns)
        
        # Normalize returns
        discounted_returns = (discounted_returns - discounted_returns.mean()) / \
                           (discounted_returns.std() + 1e-9)
        
        # Calculate policy loss
        action_probs = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * discounted_returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    env = gym.make('CartPole-v1')
    hyperparams = {
        'gamma': 0.99,
        'lr': 0.01,
        'buffer_capacity': 1000,
        'batch_size': 32
    }
    
    agent = Agent(env, hyperparams)
    agent.train(num_episodes=1000)
    
    env.close()


if __name__ == "__main__":
    main()

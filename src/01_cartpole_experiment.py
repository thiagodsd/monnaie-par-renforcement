from abc import ABC
from typing import List, Optional, Union
import gymnasium as gym
import torch
import random
from torch import nn, optim


class PolicyNetwork(nn.Module):
    """Policy network that maps states to action probabilities"""
    def __init__(self, state_dim:int, hidden_dim:int, action_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def ready(self):
        """Check if buffer has enough samples"""
        return len(self.buffer) >= self.capacity


class Agent:
    """RL agent that learns using policy gradients with experience replay"""
    def __init__(self, env:gym.Env, hyperparams:dict):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.policy = PolicyNetwork(state_dim, hyperparams['hidden_dim'], action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hyperparams['lr'])
        self.buffer = ReplayBuffer(hyperparams['buffer_size'])
        self.hyperparams = hyperparams

    def act(self, state, exploration:bool=True):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, batch):
        """Update policy using a batch of experiences"""
        states, actions, rewards, next_states, dones = batch
        
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        for r in reversed(rewards.numpy()):
            running_reward = r + self.hyperparams['gamma'] * running_reward
            discounted_rewards.insert(0, running_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                           (discounted_rewards.std() + 1e-7)
        
        # Calculate loss
        probs = self.policy(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * discounted_rewards).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes:int):
        """Train the agent for given number of episodes"""
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                if self.buffer.ready():
                    batch = self.buffer.sample(self.hyperparams['batch_size'])
                    self.update(batch)
            self.env.close()


def main():
    """Main training function"""
    env = gym.make('CartPole-v1')
    hyperparams = {
        'hidden_dim': 64,
        'lr': 0.01,
        'buffer_size': 1000,
        'gamma': 0.99,
        'batch_size': 32
    }
    
    agent = Agent(env, hyperparams)
    agent.train(num_episodes=1000)
    
    env.close()


if __name__ == "__main__":
    main()

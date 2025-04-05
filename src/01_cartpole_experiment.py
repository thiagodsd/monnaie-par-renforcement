import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Simple MLP policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
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

def train(env_name='CartPole-v1', num_episodes=1000, gamma=0.99, lr=0.01):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            action_probs = policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_log_probs.append(dist.log_prob(action))
            
            state = next_state
        
        # Calculate discounted returns
        discounted_returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)
        
        # Normalize returns
        discounted_returns = torch.FloatTensor(discounted_returns)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(episode_log_probs, discounted_returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {sum(episode_rewards)}')

if __name__ == '__main__':
    train()

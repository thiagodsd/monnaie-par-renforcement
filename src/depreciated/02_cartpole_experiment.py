import random
from typing import List, Tuple
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
import imageio
import os
from PIL import Image
from tqdm import tqdm


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def ready(self) -> bool:
        return len(self.buffer) >= self.capacity // 10


class DQNAgent:
    def __init__(self, env: gym.Env, hyperparams: dict):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dim = hyperparams["hidden_dim"]
        
        # Q networks
        self.q_network = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_network = QNetwork(state_dim, hidden_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=hyperparams["lr"]
        )
        self.buffer = ReplayBuffer(hyperparams["buffer_size"])
        self.hyperparams = hyperparams
        self.criterion = nn.MSELoss()
        
        # Epsilon for exploration
        self.epsilon_start = hyperparams["epsilon_start"]
        self.epsilon_end = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.epsilon = self.epsilon_start
        self.total_steps = 0
        
        # Create output directory for videos
        self.video_dir = "../data/videos"
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def act(self, state, exploration: bool = True) -> int:
        if exploration and random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        
        # Exploit: best action according to Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update_epsilon(self):
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.hyperparams["gamma"] * max_next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, num_episodes: int):
        all_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action = self.act(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.total_steps += 1
                episode_reward += reward
                
                # Store transition in replay buffer
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                # Update networks if we have enough samples
                if self.buffer.ready():
                    batch = self.buffer.sample(self.hyperparams["batch_size"])
                    self.update(batch)
                    self.update_epsilon()
                
                # Update target network periodically
                if self.total_steps % self.hyperparams["target_update"] == 0:
                    self.update_target_network()
            
            all_rewards.append(episode_reward)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards[-10:]))
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        self.env.close()
        return all_rewards


    def record_video(self, filename="cartpole_solution.gif", num_episodes=1):
        """Record a video of the agent solving the environment"""
        # Create a new environment with render mode for recording
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        
        frames = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            
            print(f"Recording episode {episode+1}/{num_episodes}...")
            while not (done or truncated):
                # Use greedy policy (no exploration)
                action = self.act(state, exploration=False)
                state, _, done, truncated, _ = env.step(action)
                
                # Render and capture frame
                frame = env.render()
                frames.append(Image.fromarray(frame))
        
        env.close()
        
        # Save as GIF
        output_path = os.path.join(self.video_dir, filename)
        print(f"Saving video to {output_path}")
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Video saved successfully!")
        return output_path


def main():
    env = gym.make("CartPole-v1")
    hyperparams = {
        "hidden_dim": 64,
        "lr": 0.001,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "target_update": 10  # Update target network every 10 steps
    }
    
    agent = DQNAgent(env, hyperparams)
    rewards = agent.train(190)  # Increased episodes for better training
    
    # Print final statistics
    print(f"Average reward over last 100 episodes: {sum(rewards[-100:]) / 100:.2f}")
    
    # Record and save video of the trained agent
    agent.record_video()


if __name__ == "__main__":
    main()
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import play_with_trained_agent

class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

class ValueNet(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size=4, action_size=2):
        self.gamma = 0.98
        self.lr_pi = 0.00002
        self.lr_v = 0.00005
        self.state_size = state_size
        self.action_size = action_size
        self.pi = PolicyNet(state_size, action_size)
        self.v = ValueNet(state_size)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.pi.eval()
        
        with torch.no_grad():
            probs = self.pi(state_tensor)
        self.pi.train()
        
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        
        return action
    
    def update(self, state, action, reward, next_state, terminated):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
        
        self.pi.train()
        self.v.train()
        
        with torch.no_grad():
            target_q_value = self.v(next_state_tensor)
        
        # Calculate loss for value network
        target = reward + self.gamma * target_q_value * (1 - terminated)
        v_s = self.v(state_tensor)
        loss_v = F.mse_loss(v_s, target)
        
        # Calculate loss for policy network
        delta = (target - v_s).detach()
        action_prob = self.pi(state_tensor)
        dist = torch.distributions.Categorical(probs=action_prob)
        log_prob = dist.log_prob(torch.tensor(action, dtype=torch.int64))
        loss_pi = -log_prob * delta
        
        # Update networks
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()
        
        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

def main():
    # Initialize environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agent
    agent = Agent(state_size, action_size)

    # Train agent
    start = time.time()
    episodes = 2000
    reward_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        reward_sum = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            reward_sum += reward
            agent.update(state, action, reward, next_state, terminated)
            if terminated or reward_sum > 999:
                agent.update(state, action, reward, next_state, terminated)
                break
            state = next_state
            
        reward_history.append(reward_sum)
        print(f"Episode {episode+1} reward: {reward_sum}")

    compute_time = time.time() - start
    print(f"Training complete in {compute_time:.2f} seconds")
    env.close()

    # # Plot results
    # plt.figure(figsize=(10, 5))
    # plt.plot(reward_history)
    # plt.title('Reward History')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.grid(True)
    # plt.show()

    play_with_trained_agent(agent)

if __name__ == '__main__':
    main()
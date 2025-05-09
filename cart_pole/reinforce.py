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

class Agent:
    def __init__(self, state_size=4, action_size=2):
        self.gamma = 0.98
        self.lr = 0.0002
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.pi = PolicyNet(state_size, action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)
    
    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        self.pi.eval()
        with torch.no_grad():
            probs = self.pi(state_tensor)
        self.pi.train()
        
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        
        return action

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
    
    def update(self):
        if not self.memory:
            return
        
        self.optimizer.zero_grad()
        
        states = [data[0] for data in self.memory]
        actions = [data[1] for data in self.memory]
        rewards = [data[2] for data in self.memory]
        
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns
        if len(returns_tensor) > 1:
            eps = np.finfo(np.float32).eps.item()
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + eps)
        
        loss = 0
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            G = returns_tensor[i]
            
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            action_probs = self.pi(state_tensor)
            dist = torch.distributions.Categorical(probs=action_probs)
            log_prob = dist.log_prob(torch.tensor(action))
            
            loss += -log_prob * G
        
        loss.backward()
        self.optimizer.step()
        
        self.memory = []

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
            agent.add(state, action, reward)
            if terminated or reward_sum >= 1000:
                break
            state = next_state
            
        agent.update()
        reward_history.append(reward_sum)
        print(f"Episode {episode + 1} score: {reward_sum}")
        
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
    
    # Play with the trained agent
    play_with_trained_agent(agent)

if __name__ == '__main__':
    main()
import gymnasium as gym
import time

def play_with_trained_agent(agent, num_episodes=5, upper_limit=1000):
    print("Playing with the trained agent...")
    
    env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        reward_sum = 0
        
        while True:
            env.render()
            action = agent.get_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            reward_sum += reward
            if terminated or reward_sum >= upper_limit:
                env.render()
                break
            state = next_state
            time.sleep(0.01)
        
        print(f"Episode {episode + 1} score: {reward_sum}")
    
    env.close()
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO, DQN
import time
from custom_network import CustomActorCriticPolicy
from dataset import CustomDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

SMALLEST_MAX_DATAGRAM_SIZE = 1200

class Env(gym.Env):
    def __init__(self, dataset: CustomDataset):
        #self.action_space = spaces.Box(low=0, high=1_000_000, shape=(1,), dtype=np.float32)
        self.observation_shape = 54
        self.dataset = dataset
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_shape,), dtype=np.float32)
        self.current_step = 0
        self.cw = self.dataset.original_y[self.current_step].item() // SMALLEST_MAX_DATAGRAM_SIZE
        self.action_mapping = {
            0: lambda x: x / 3,
            1: lambda x: x / 2,
            2: lambda x: x - 10,
            3: lambda x: x - 7,
            4: lambda x: x - 3,
            5: lambda x: x,
            6: lambda x: x + 3,
            7: lambda x: x + 7,
            8: lambda x: x + 10,
            9: lambda x: x * 2,
            10: lambda x: x * 3
        }
        self.max_steps = len(dataset) - 1

    def perform_action(self, action, value):
        return self.action_mapping[action](value)
    
    def step(self, action):
        ts = self.dataset.get_timestamps(self.current_step)

        self.cw = self.perform_action(action.item(), self.cw)
               
        cw_value = self.dataset.original_y[self.current_step].item() // SMALLEST_MAX_DATAGRAM_SIZE
        #cw_value = self.dataset.y[self.current_step].item()

        cw_difference = abs(self.cw - cw_value)
        #print(cw_difference)
        
        if cw_difference < 10:
            reward = 1
        else:
            reward = 1 / cw_difference

        terminated = False
        truncated = False
        info = {}

        self.current_step += 1

        if self.current_step > self.max_steps:
            truncated = True
            self.current_step = 0
            self.cw = self.dataset.original_y[self.current_step].item()

        if self.dataset.get_timestamps(self.current_step) - ts > 2:
            terminated = True

        observation = self.dataset.data[self.current_step]

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        #self.dataset.shuffle_data()
        observation = self.dataset.data[self.current_step]
        self.cw = self.dataset.original_y[self.current_step].item() // SMALLEST_MAX_DATAGRAM_SIZE
        return observation, {}
    
    def get_current_step(self):
        return self.current_step
    
    
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(script_dir, "..", "stats_cubic_aioquic.csv"))
    dataset = CustomDataset(csv_file=csv_path, columns_to_remove=["timestamp"])
    env = Env(dataset)

    model = DQN("MlpPolicy", env, exploration_fraction=0.2)
    print(model.policy)
    model = model.learn(total_timesteps=2000, progress_bar=True)
    #print("done!")

    total_timesteps = 50
    episode_rewards = []
    best_reward = float('-inf')

    progress_bar = tqdm(total=total_timesteps, desc=f"Evaluating", position=0)
    
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.zip")

    for i in range(total_timesteps):
        start = time.time()
        obs, _ = env.reset()
        episode_length = 0
        episode_reward = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_length += 1
            episode_reward += reward
        
        episode_rewards.append(episode_reward)

        progress_bar.set_description(f"Episode {i}: Total Reward = {episode_reward}: Episode length = {episode_length} steps")
        progress_bar.update(1)

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.legend()
    
    plots_dir = os.path.normpath(os.path.join(script_dir, "..", "plots"))
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "cubic_env_train.png"))
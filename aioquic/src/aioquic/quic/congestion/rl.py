import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import os, sys, time
import zmq
from stable_baselines3.common.buffers import ReplayBuffer

def get_original_username():
    # Get the original username using the SUDO_USER environment variable
    return os.getenv("SUDO_USER") or os.getenv("USER")

user = get_original_username()

if user == None:
    directory_path = f"/ETTUS-data-collection/data_collection"
else:
    directory_path = f"/home/{user}/ETTUS-data-collection/data_collection"


directory = os.path.abspath(directory_path)
sys.path.append(directory)

from parsing import parse_L1, parse_MAC, parse_bw
from live_processing import clean_L1_list, clean_MAC_list

K_MINIMUM_WINDOW = 2

max_datagram_size = 1500

#subscriber_addr = "tcp://127.0.0.1:1234"
#context = zmq.Context()
#subscriber = context.socket(zmq.SUB)
#subscriber.bind(subscriber_addr)
#subscriber.setsockopt(zmq.SUBSCRIBE, b'')
#print("Socket SUB is open")
#try:
#    observation = subscriber.recv(flags=zmq.NOBLOCK).decode()
#except zmq.error.Again:
#    print("No message available")
#    pass

class Env(gym.Env):
    def __init__(self):
        self.observation_shape = 3
        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_shape,), dtype=np.float32)
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
        self.congestion_window = 0
        self.second_last_obs = None
        self.last_obs = None
        self.last_cw = None 
        self.replay_buffer = ReplayBuffer(buffer_size=10000, observation_space=self.observation_space, action_space=self.action_space, device='auto')

    def reset(self, seed = None):
        return self.last_obs, {}

    def step(self, action):
        # Prendi osservazione
        #observation = subscriber.recv().decode()
        #print(type(self.last_obs))
        #print(type(self.second_last_obs))
        self.congestion_window = self.second_last_obs[1]

        #Applica azione
        self.congestion_window = self.perform_action(action, self.congestion_window)

        #Riprendi la nuova osservazione
        #observation = subscriber.recv().decode()
        observation = self.last_obs
        self.congestion_window = self.last_obs[1]
        self.bw = self.last_obs[0]
        self.rtt = self.last_obs[2]

        #Calcolo reward
        reward = self.get_reward()
        self.replay_buffer.add(self.second_last_obs, self.last_obs, action, reward, True, {})
        #Ritorna cose di gym
        return observation, reward, True, True, {}

    def perform_action(self, action, value):
        self.last_cw = value
        return self.action_mapping[action](value)
    
    def get_reward(self):
        return self.bw // self.rtt

    def normalize_obs(self, obs):
        return obs
    
    def normalize_reward(self, reward):
        return reward

class RL_agent():
    def __init__(self) -> None:
        self.env = Env()
        self.path = "/home/cristiano/ETTUS-data-collection/aioquic/src/aioquic/quic/congestion/model.zip"
        self.action = None

        if os.path.isfile(self.path):
            print("Loading...")
            start = time.time()
            self.model = DQN.load(self.path)
            self.model.set_env(self.env)
            self.model.policy.optimizer.batch_size = 1
            print(f"Time to load: {time.time() - start}")

        else:
            self.model = DQN("MlpPolicy", self.env, verbose=1, batch_size=1)
            print("Creating new model")
        #self.model = DQN("MlpPolicy", self.env, verbose=1, batch_size=1)

    def get_new_cw(self, obs, cw):
        self.env.second_last_obs = np.array(obs)
        action = self.model.predict(np.array(obs))[0]
        action_key = int(action)
        self.action = action
        return self.env.perform_action(action_key, cw)
    
    def predict_and_learn(self, obs, cw):
        self.env.last_obs = np.array(obs)
        self.env.congestion_window = cw
        self.bw = self.env.last_obs[0]
        self.rtt = self.env.last_obs[2]
        reward = self.bw // self.rtt
        print(reward)
        infos = [{"TimeLimit.truncated": False}]
        self.model.replay_buffer.add(self.env.second_last_obs, self.env.last_obs, self.action, reward, True, infos)
        #self.model.learn(total_timesteps=10, log_interval=1)
        if self.model.replay_buffer.size() > 2:
            #batch = self.model.replay_buffer.sample(self.model.batch_size, env=self.env)
            print("Training")
            self.model.train(10)
        self.model.save(self.path)
        #print("Saving...")
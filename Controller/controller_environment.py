import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

from RL.dataset import CustomDataset
from Controller.config import CONTROLLER_OBSERVATION_COLS, REWARD_WEIGHTS, MAX_STEPS_PER_EPISODE
from Controller.tcp_adapter import TCPAdapter
from Controller.fc_adapter import FCAdapter

class ControllerEnv(gym.Env):
    """
    Adaptive Controller Environment.
    Decides between TCP (Action 0) and Fountain Codes (Action 1) based on channel conditions.
    """
    def __init__(self, dataset_path: str = "stats_cubic_aioquic.csv", is_eval: bool = False):
        super().__init__()
        
        # We reuse the CustomDataset logic to fetch normalized channel conditions
        if is_eval:
            train_dataset = CustomDataset(
                csv_file="stats_cubic_aioquic.csv", 
                columns_to_remove=["timestamp"],
                fc_cols=CONTROLLER_OBSERVATION_COLS
            )
            self.dataset = CustomDataset(
                csv_file=dataset_path, 
                columns_to_remove=["timestamp"],
                fc_cols=CONTROLLER_OBSERVATION_COLS,
                scaler=train_dataset.scaler
            )
        else:
            self.dataset = CustomDataset(
                csv_file=dataset_path, 
                columns_to_remove=["timestamp"],
                fc_cols=CONTROLLER_OBSERVATION_COLS
            )
            
        # Initialize adapters internally, avoiding global variables
        self.tcp_adapter = TCPAdapter(csv_path=dataset_path)
        self.fc_adapter = FCAdapter(csv_path=dataset_path)
        # Sync the dataset to ensure FC Env uses the exact same shuffled rows
        self.fc_adapter.sync_dataset(self.dataset)

            
        # Action space: 0 -> TCP, 1 -> FC
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 
        # Base features (len(CONTROLLER_OBSERVATION_COLS)) + 
        # 4 extra features (prev_mode, time_since_switch, prev_reward, rolling_reward)
        self.obs_dim = len(CONTROLLER_OBSERVATION_COLS) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.max_steps = min(MAX_STEPS_PER_EPISODE, len(self.dataset))
        
        # State tracking variables
        self.current_step = 0
        self.prev_mode = 0.0 # Default to TCP initially
        self.time_since_switch = 0.0
        self.prev_reward = 0.0
        self.reward_history = []
        
    def _get_obs(self) -> np.ndarray:
        # Get base channel features
        base_obs = self.dataset.data[self.current_step].numpy()
        
        # Calculate rolling reward
        rolling_reward = np.mean(self.reward_history[-10:]) if self.reward_history else 0.0
        
        # Extra state features
        extra_obs = np.array([
            self.prev_mode,
            self.time_since_switch / self.max_steps, # Normalized time since switch
            self.prev_reward,
            rolling_reward
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, extra_obs])

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.dataset.shuffle_data()
        self.current_step = 0
        self.prev_mode = 0.0
        self.time_since_switch = 0.0
        self.prev_reward = 0.0
        self.reward_history = []
        
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= self.max_steps:
            return self._get_obs(), 0.0, True, False, {}

        # 1. Check for switch
        switched = (action != int(self.prev_mode))
        if switched:
            self.time_since_switch = 0.0
        else:
            self.time_since_switch += 1.0
            
        # 2. Run the chosen mode
        info = {"action": action, "switched": switched}
        
        if action == 0:
            # TCP Mode
            latency, packet_loss, rtt, overhead, delivery_success_rate = self.tcp_adapter.run_tcp(self.current_step)
            
            # Compute TCP Reward
            reward = (
                REWARD_WEIGHTS["delivery_success"] * delivery_success_rate -
                REWARD_WEIGHTS["latency"] * (latency / 100.0) - # Normalize latency for reward
                REWARD_WEIGHTS["packet_loss"] * packet_loss -
                REWARD_WEIGHTS["overhead"] * (overhead - 1.0)
            )
            
            info.update({
                "latency": latency,
                "packet_loss": packet_loss,
                "overhead": overhead,
                "success_rate": delivery_success_rate,
                "mode": "TCP"
            })
            
        else:
            # FC Mode
            decode_success_rate, latency, packet_loss, overhead, fc_reward, redundancy = self.fc_adapter.run_fc(self.current_step)
            
            # Compute FC Reward (incorporating the base FC reward + our weights)
            reward = (
                REWARD_WEIGHTS["decode_success"] * decode_success_rate -
                REWARD_WEIGHTS["latency"] * (latency / 100.0) -
                REWARD_WEIGHTS["packet_loss"] * packet_loss -
                REWARD_WEIGHTS["overhead"] * (overhead - 1.0)
            )
            # Add the underlying FC reward as a baseline
            reward += (fc_reward * 0.1) 
            
            info.update({
                "latency": latency,
                "packet_loss": packet_loss,
                "overhead": overhead,
                "success_rate": decode_success_rate,
                "mode": "FC"
            })
            
        # 3. Apply Switching Penalty
        if switched:
            reward -= REWARD_WEIGHTS["switch_penalty"]
            
        # Clip reward for stability
        reward = float(np.clip(reward, -10.0, 10.0))
        
        # 4. Update tracking variables
        self.prev_mode = float(action)
        self.prev_reward = reward
        self.reward_history.append(reward)
        self.current_step += 1
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        next_obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        
        return next_obs, reward, terminated, truncated, info

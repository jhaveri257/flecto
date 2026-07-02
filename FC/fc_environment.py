import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

from RL.dataset import CustomDataset
from FC.fountain_code import FountainCode, simulate_erasure_channel, bler_to_loss_rate

FC_OBSERVATION_COLS = [
    "BLER_UL",
    "BLER_DL",
    "MCS_UL",
    "MCS_DL",
    "smoothed_rtt",
    "rtt_mean_deviation",
    "bw_avg",
    "ulsch_errors",
]

ACTION_MAP = {
    0:  (64,  1.1),
    1:  (64,  1.3),
    2:  (64,  1.5),
    3:  (128, 1.1),
    4:  (128, 1.3),
    5:  (128, 1.5),
    6:  (128, 1.8),
    7:  (256, 1.2),
    8:  (256, 1.5),
    9:  (256, 1.8),
    10: (256, 2.0),
}

class FCEnv(gym.Env):
    """
    Fountain Code Parameter Optimization Environment.
    
    The RL agent observes 8 channel features and selects one of 11 discrete
    actions that map to (k, redundancy_ratio). The environment simulates
    the Fountain Code transmission over a channel with the observed BLER,
    and returns a reward based on decode success and overhead efficiency.
    """
    def __init__(self, dataset_path: str, is_eval: bool = False):
        super().__init__()
        
        # To avoid training-serving skew, we must use the training scaler during evaluation
        if is_eval:
            train_dataset = CustomDataset(
                csv_file="stats_cubic_aioquic.csv", 
                columns_to_remove=["timestamp"],
                fc_cols=FC_OBSERVATION_COLS
            )
            self.dataset = CustomDataset(
                csv_file=dataset_path, 
                columns_to_remove=["timestamp"],
                fc_cols=FC_OBSERVATION_COLS,
                scaler=train_dataset.scaler
            )
        else:
            self.dataset = CustomDataset(
                csv_file=dataset_path, 
                columns_to_remove=["timestamp"],
                fc_cols=FC_OBSERVATION_COLS
            )
            
        # Action space: 11 discrete choices
        self.action_space = spaces.Discrete(11)
        
        # Observation space: 8 normalized features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(FC_OBSERVATION_COLS),), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = min(500, len(self.dataset))
        
        # For repeatable FC generation within an episode
        self.fc_seed_base = 42
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # In a real training scenario, we shuffle the dataset per episode.
        self.dataset.shuffle_data()
        self.current_step = 0
        
        obs = self.dataset.data[self.current_step].numpy()
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_step >= self.max_steps:
            return self.dataset.data[-1].numpy(), 0.0, True, False, {}

        # 1. Get channel state and true BLER
        obs = self.dataset.data[self.current_step].numpy()
        raw_bler = self.dataset.raw_bler[self.current_step]
        loss_rate = bler_to_loss_rate(raw_bler)
        
        # 2. Decode action
        k, redundancy_ratio = ACTION_MAP[action]
        
        # 3. Simulate FC transmission
        fc = FountainCode(k=k, seed=self.fc_seed_base + self.current_step)
        
        # Use dummy source data since we only care if it decodes
        source_data = np.zeros(k, dtype=np.uint8)
        encoded_packets = fc.encode(source_data, redundancy_ratio=redundancy_ratio)
        
        # Simulate packet erasure channel
        rng = np.random.default_rng(self.fc_seed_base + self.current_step)
        received_packets = simulate_erasure_channel(encoded_packets, loss_rate=loss_rate, rng=rng)
        
        # Try to decode
        decoded = fc.decode(received_packets, k=k)
        decode_success = decoded is not None
        
        # 4. Compute reward
        if decode_success:
            # Overhead penalty: proportional to wasted redundancy.
            # We allow a safe margin (e.g., 10%) before penalizing.
            wasted_ratio = max(0.0, redundancy_ratio - (1.0 + loss_rate) - 0.1)
            reward = max(0.1, 1.0 - (wasted_ratio * 2.0))
        else:
            # Catastrophic failure penalty
            reward = -50.0
            
        reward = float(np.clip(reward, -50.0, 1.0))
        
        # 5. Advance step
        
        # Extract correctly shuffled raw metrics for tracking
        raw_bw = self.dataset.raw_bw[self.current_step]
        raw_rtt = self.dataset.raw_rtt[self.current_step]
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "decode_success": decode_success,
            "redundancy_ratio": redundancy_ratio,
            "k": k,
            "loss_rate": loss_rate,
            "n_encoded": len(encoded_packets),
            "n_received": len(received_packets),
            "reward": reward,
            "raw_bler": raw_bler,
            "raw_bw": raw_bw,
            "raw_rtt": raw_rtt,
        }
        
        next_obs = self.dataset.data[self.current_step].numpy() if not terminated else obs
        
        return next_obs, reward, terminated, truncated, info

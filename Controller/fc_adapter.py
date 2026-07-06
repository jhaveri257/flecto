import os
from typing import Tuple
import numpy as np

from stable_baselines3 import PPO
from FC.fc_environment import FCEnv

class FCAdapter:
    """
    Adapter for running Fountain Code mode.
    Uses the existing, pre-trained FC PPO model to decide redundancy,
    treating the FC implementation as a black box.
    """
    def __init__(self, model_path: str = "FC/models/fc_ppo_agent_v2.zip", csv_path: str = "stats_cubic_aioquic.csv"):
        self.csv_path = csv_path
        self.env = FCEnv(csv_path, is_eval=True)
        
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
        else:
            print(f"Warning: FC model not found at {model_path}. Will use random actions.")
            self.model = None

    def sync_dataset(self, dataset):
        """
        Synchronizes the internal FC environment's dataset with the controller's dataset.
        This ensures both environments see the exact same channel conditions 
        even after shuffling (Issue 3 fix).
        """
        self.env.dataset = dataset

    def run_fc(self, step_idx: int) -> Tuple[float, float, float, float, float, float]:
        """
        Simulates an FC transmission step using the pre-trained FC PPO agent.
        
        Args:
            step_idx: Current step index. The environment is manually stepped to this point.
            
        Returns:
            Tuple containing:
            - Decode Success Rate (float)
            - Latency (float)
            - Packet Loss (float)
            - Overhead (float)
            - Reward (float)
            - Redundancy (float)
        """
        # Proper synchronization mechanism (Issue 2 fix)
        # We must align FCEnv's internal step with the Controller's step 
        # since the Controller might have skipped FC steps while using TCP.
        self.env.current_step = step_idx
        
        # Observation builder (Issue 3 fix)
        # FCEnv does not have a separate _get_obs(). It builds observations
        # exactly like this in its reset() and step() methods.
        obs = self.env.dataset.data[step_idx].numpy()
        
        if self.model:
            action, _states = self.model.predict(obs, deterministic=True)
        else:
            action = self.env.action_space.sample()
            
        # Convert action to a Python integer to avoid "unhashable type: 'numpy.ndarray'" in ACTION_MAP
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif isinstance(action, np.generic):
            action = int(action)
        else:
            action = int(action)
            
        # Step the environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract metrics from info
        decode_success_rate = 1.0 if info.get("decode_success", False) else 0.0
        
        # Latency for FC:
        # FCEnv does not expose a separate latency metric, only raw_rtt.
        # Fountain Codes eliminate retransmissions but do not reduce the physical 
        # propagation delay of the network. Therefore, we use the measured RTT directly
        # rather than applying arbitrary multipliers.
        rtt = info.get("raw_rtt", 20.0)
        latency = float(rtt)
        
        packet_loss = info.get("loss_rate", 0.0)
        redundancy = info.get("redundancy_ratio", 1.0)
        overhead = redundancy  # Overhead is directly proportional to redundancy in FC
        fc_reward = reward
        
        return decode_success_rate, latency, packet_loss, overhead, fc_reward, redundancy

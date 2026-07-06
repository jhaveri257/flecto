import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class TCPAdapter:
    """
    Adapter for running TCP mode.
    Currently reuses static CSV datasets (e.g., stats_cubic_aioquic.csv) to simulate TCP performance.
    Designed to be easily replaced with live aioquic execution later.
    """
    def __init__(self, csv_path: str = "stats_cubic_aioquic.csv"):
        self.csv_path = csv_path
        # Load the dataset for simulation
        try:
            self.df = pd.read_csv(self.csv_path)
            self.max_idx = len(self.df) - 1
        except FileNotFoundError:
            print(f"Warning: {csv_path} not found. TCP Adapter will return dummy data.")
            self.df = None
            self.max_idx = 0
            
    def run_tcp(self, step_idx: int) -> Tuple[float, float, float, float, float]:
        """
        Simulates a TCP transmission step based on current channel conditions.
        
        Args:
            step_idx: The current step index to fetch simulated performance metrics.
            
        Returns:
            Tuple containing:
            - Latency (float)
            - Packet Loss (float)
            - RTT (float)
            - Overhead (float)
            - Delivery Success Rate (float)
        """
        if self.df is None or step_idx > self.max_idx:
            # Fallback dummy data if dataset missing or out of bounds
            return 50.0, 0.05, 20.0, 1.05, 0.95
            
        row = self.df.iloc[step_idx]
        
        # Extract metrics (approximations based on available columns)
        # Using raw_rtt and smoothed_rtt as latency and RTT indicators
        rtt = float(row.get('raw_rtt', 20.0))
        latency = float(row.get('smoothed_rtt', rtt)) 
        
        # Approximate packet loss from ulsch_errors or BLER
        packet_loss = float(row.get('BLER_DL', 0.0)) / 100.0
        
        # Improved Delivery Success Rate:
        # TCP retransmits lost packets. Ultimate failure only occurs if all retries fail.
        # Assuming a timeout threshold equivalent to ~3 retries for a simplified model.
        delivery_success_rate = 1.0 - (packet_loss ** 3)
        delivery_success_rate = max(0.0, min(1.0, delivery_success_rate))
        
        # Improved Overhead:
        # Base TCP/IP header overhead is approx 2.7% (40 bytes / 1500 bytes MTU).
        # Retransmissions due to packet loss increase overhead.
        # mathematically: expected transmissions per packet = 1 / (1 - packet_loss)
        base_overhead = 1.027
        overhead = base_overhead / max(0.01, (1.0 - packet_loss))
        
        return latency, packet_loss, rtt, overhead, delivery_success_rate

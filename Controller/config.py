"""
Configuration parameters for the Adaptive Controller.
Includes reward weights, state features, and hyperparameters.
"""

# State features that the controller will observe
CONTROLLER_OBSERVATION_COLS = [
    "BLER_UL",
    "BLER_DL",
    "MCS_UL",
    "MCS_DL",
    "smoothed_rtt",
    "rtt_mean_deviation",
    "bw_avg",
    "ulsch_errors",
    # Additionally, the environment will append:
    # - Previous Mode (0 for TCP, 1 for FC)
    # - Time Since Last Switch
    # - Previous Reward
    # - Rolling Reward Average
]

# Reward Weights
REWARD_WEIGHTS = {
    "decode_success": 1.0,     # Weight for successful FC decode
    "delivery_success": 1.0,   # Weight for successful TCP delivery
    
    # Penalize delay while keeping successful delivery the primary objective.
    "latency": 0.5,
    
    # Penalize unreliable links.
    "packet_loss": 0.5,
    
    # Fountain Codes intentionally add redundancy.
    # Keep this penalty small so PPO does not avoid FC simply because of overhead.
    "overhead": 0.2,
    
    # Increase switching penalty to discourage oscillating
    # TCP -> FC -> TCP -> FC every few timesteps.
    "switch_penalty": 0.3,
}

# General Configuration
MAX_STEPS_PER_EPISODE = 500
TRAINING_TIMESTEPS = 30000

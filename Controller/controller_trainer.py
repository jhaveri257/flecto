import os
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import CheckpointCallback

from Controller.controller_environment import ControllerEnv
from RL.custom_network import CustomActorCriticPolicy
from Controller.config import TRAINING_TIMESTEPS

def train_controller_agent(csv_path: str = "stats_cubic_aioquic.csv", models_dir: str = "Controller/models", logs_dir: str = "Controller/logs"):
    """
    Trains the PPO-based Adaptive Controller to switch between TCP and FC.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create the environment and wrap with Monitor for logging
    env = ControllerEnv(csv_path)
    env = Monitor(env, logs_dir)
    
    # Check if a model already exists to resume training
    model_path = os.path.join(models_dir, "controller_ppo_agent.zip")
    if os.path.exists(model_path):
        print(f"Resuming training from {model_path}")
        model = PPO.load(model_path, env=env)
        model.ent_coef = 0.01
    else:
        print("Starting PPO Adaptive Controller training from scratch")
        model = PPO(
            CustomActorCriticPolicy, 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01
        )
    
    # Checkpoint callback: save every 5000 steps
    checkpoint_cb = CheckpointCallback(
        save_freq=5000,
        save_path=os.path.join(models_dir, "checkpoints"),
        name_prefix="controller_ppo"
    )
    
    print(f"Training Controller agent for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=checkpoint_cb)
    
    # Save the final trained model
    final_path = os.path.join(models_dir, "controller_ppo_agent")
    model.save(final_path)
    print(f"Model saved to {final_path}.zip")
    
    # Plot training curve
    results = load_results(logs_dir)
    if len(results) > 0:
        x, y = ts2xy(results, 'timesteps')
        
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, alpha=0.3, label='Episode Reward')
        
        # Moving average
        window = min(10, len(y))
        if window > 0:
            y_smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smoothed = x[window-1:]
            plt.plot(x_smoothed, y_smoothed, label=f'Moving Avg ({window} ep)', color='blue', linewidth=2)
            
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Adaptive Controller PPO Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(models_dir, "training_curve.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Training curve saved to {plot_path}")
    else:
        print("No results logged to plot.")

if __name__ == "__main__":
    train_controller_agent()

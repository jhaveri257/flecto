"""
Federated/config.py
====================
Single source of truth for every configurable parameter in the
Federated Learning module.

Design principle: No magic numbers anywhere else in the codebase.
Every constant lives here with a full comment explaining its purpose.

Settings are organized into seven logical sections:

    REPRODUCIBILITY  — global RNG seed and determinism settings
    DATASET          — MNIST paths, partitioning strategy
    MODEL            — MLP architecture dimensions
    TRAINING         — SGD hyperparameters, local epochs
    CLIENT           — FL topology (number of clients, participation rate)
    COMMUNICATION    — Top-K sparsification, error feedback, transport bridge
    OUTPUT           — Paths for results, logs, plots; verbosity

Backward compatibility: every constant retains its original flat name at
module level so any code using ``from Federated.config import TOP_K_RATIO``
continues to work without modification.
"""

import random
import numpy as np
import torch

# ===========================================================================
# SECTION 1 — REPRODUCIBILITY
# ===========================================================================

# Master random seed used by all submodules.
# Pass this seed to torch, numpy, and random for fully deterministic runs.
SEED: int = 42

# Also aliased as PARTITION_SEED for the data partitioner (backward-compat)
PARTITION_SEED: int = SEED


def set_seed(seed: int = SEED) -> None:
    """
    Initialize all global RNG sources for deterministic execution.

    Call this function at the very start of any training or evaluation
    script before constructing datasets, models, or data loaders.

    Sets seeds for:
    - Python's built-in ``random`` module
    - NumPy's global RNG (``numpy.random``)
    - PyTorch CPU RNG (``torch.manual_seed``)
    - PyTorch CUDA RNG (``torch.cuda.manual_seed_all``) if available
    - CuDNN deterministic mode (disables non-deterministic CUDA kernels)

    Args:
        seed: Integer seed value. Defaults to the module-level ``SEED``
              constant (42).

    Returns:
        None

    Note:
        CuDNN deterministic mode may reduce GPU throughput slightly.
        This is acceptable for research reproducibility.

    Example:
        >>> from Federated.config import set_seed
        >>> set_seed()          # uses default SEED=42
        >>> set_seed(seed=7)    # uses a custom seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic CUDA kernels (may slow GPU training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# SECTION 2 — DATASET
# ===========================================================================

# MNIST input dimensionality: 28 x 28 = 784 flattened pixels.
INPUT_DIM: int = 784

# Number of output classes (MNIST digits 0–9).
NUM_CLASSES: int = 10

# Fraction of MNIST training data used per experiment run.
# 1.0 = use the full 60,000-sample training set.
# Reduce (e.g., 0.1) for rapid smoke-test runs.
DATASET_FRACTION: float = 1.0

# Local filesystem path for the downloaded MNIST cache.
MNIST_DATA_DIR: str = "Federated/results/data"

# Data partitioning strategy for distributing MNIST across FL clients.
#   "iid"     — each client receives a uniform random shard (unrealistic baseline).
#   "non_iid" — each client receives data from N_CLASSES_PER_CLIENT classes only
#               (McMahan et al. 2017 pathological Non-IID).
PARTITION_MODE: str = "non_iid"

# Number of distinct label classes assigned to each client in Non-IID mode.
# McMahan et al. 2017 uses 2; larger values make the distribution closer to IID.
N_CLASSES_PER_CLIENT: int = 2


# ===========================================================================
# SECTION 3 — MODEL
# ===========================================================================

# MLP architecture: Input (784) -> 128 -> 64 -> Output (10)
# Fully connected layers only. No CNN.
# Objective is communication efficiency, not state-of-the-art image accuracy.

# Width of the first hidden layer.
HIDDEN_1: int = 128

# Width of the second hidden layer.
HIDDEN_2: int = 64

# Dropout probability applied after each hidden layer.
# Helps regularize small client datasets in Non-IID settings.
DROPOUT_RATE: float = 0.1


# ===========================================================================
# SECTION 4 — TRAINING
# ===========================================================================

# Number of local SGD epochs each client performs per communication round.
# McMahan et al. 2017 use E=5; we default to 3 for faster experiments.
LOCAL_EPOCHS: int = 3

# Local mini-batch size for SGD.
BATCH_SIZE: int = 32

# SGD learning rate for local training.
LEARNING_RATE: float = 0.01

# SGD momentum coefficient. Standard value for FL baselines.
MOMENTUM: float = 0.9

# L2 weight decay (regularization). Prevents overfitting on small local shards.
WEIGHT_DECAY: float = 1e-4

# Maximum gradient L2 norm for gradient clipping during local training.
# Prevents exploding gradients in heterogeneous Non-IID client settings.
GRAD_CLIP_NORM: float = 1.0


# ===========================================================================
# SECTION 5 — CLIENT (FL TOPOLOGY)
# ===========================================================================

# Total number of simulated FL clients.
NUM_CLIENTS: int = 10

# Number of global communication rounds (server–client iterations).
NUM_ROUNDS: int = 30

# Fraction of clients sampled and activated each round.
# 1.0 = all clients participate every round (synchronous FL).
# < 1.0 = partial participation (asynchronous approximation).
CLIENT_SAMPLE_FRACTION: float = 1.0


# ===========================================================================
# SECTION 6 — COMMUNICATION (TOP-K + TRANSPORT BRIDGE)
# ===========================================================================

# --- Top-K Sparse Update Compression ---

# Fraction of weight deltas retained and transmitted per round.
# e.g., 0.1 means only the top 10% of |ΔW| entries are sent.
# Larger values reduce compression but improve reconstruction accuracy.
TOP_K_RATIO: float = 0.1

# Whether to apply error feedback (accumulate residual errors from the
# previous round and add them before the next Top-K selection).
# Enables convergence guarantees equivalent to full-gradient FL.
# Reference: Stich et al. (2018). "Sparsified SGD with Memory." NeurIPS.
USE_ERROR_FEEDBACK: bool = True

# --- Server-Side Reconstruction ---

# Strategy for filling positions not included in the sparse update.
# "zero" : set unselected positions to 0 (standard for Top-K FL).
RECONSTRUCTION_FILL: str = "zero"

# FedAvg aggregation weighting strategy.
# True  = weight each client's update by its local dataset size (standard).
# False = treat all clients equally regardless of shard size.
WEIGHTED_FEDAVG: bool = True

# --- Transport Bridge ---

# Protocol the bridge uses to simulate payload delivery.
# "tcp"  : always route through TCPAdapter (Controller module).
# "fc"   : always route through FCAdapter  (FC + Controller modules).
# "auto" : let the Controller PPO decide each round (requires trained model).
BRIDGE_MODE: str = "auto"

# Path to the pre-trained Controller PPO model zip file.
# Used when BRIDGE_MODE="auto".
CONTROLLER_MODEL_PATH: str = "Controller/models/controller_ppo_agent.zip"

# Path to the pre-trained FC PPO model zip file.
# Used by FCAdapter inside the bridge.
FC_MODEL_PATH: str = "FC/models/fc_ppo_agent_v2.zip"

# Path to the network channel statistics CSV.
# Required by TCPAdapter and FCAdapter for channel-condition simulation.
CHANNEL_STATS_CSV: str = "stats_cubic_aioquic.csv"


# ===========================================================================
# SECTION 7 — OUTPUT (RESULTS, LOGGING, EVALUATION)
# ===========================================================================

# Base directory for all module outputs.
RESULTS_DIR: str = "Federated/results"

# Subdirectory for final and periodic global model checkpoints (.pt files).
MODELS_DIR: str = "Federated/results/models"

# Subdirectory for intermediate per-round global model snapshots.
CHECKPOINTS_DIR: str = "Federated/results/checkpoints"

# Subdirectory for CSV metric logs (per-round accuracy, loss, bandwidth).
LOGS_DIR: str = "Federated/results/logs"

# Subdirectory for generated evaluation plots (PNG files).
PLOTS_DIR: str = "Federated/results/plots"

# Save a global model checkpoint every N rounds. 0 = only save the final model.
CHECKPOINT_EVERY_N_ROUNDS: int = 5

# Console verbosity level.
# 0 = silent, 1 = per-round summary, 2 = per-client detail.
VERBOSITY: int = 1

# --- Evaluation ---

# Evaluation mode definitions for the evaluator script.
# Each entry is a tuple: (mode_label, top_k_ratio, use_error_feedback).
EVAL_MODES = [
    ("Full Update",             1.0,  False),
    ("Top-K (10%)",             0.10, False),
    ("Top-K + Error Feedback",  0.10, True),
]

# Number of communication rounds to run during evaluation.
# May differ from NUM_ROUNDS (used in fl_round_runner).
EVAL_ROUNDS: int = 20

# DPI for saved plot PNG files.
PLOT_DPI: int = 150

# Default figure size (width, height in inches) for all plots.
PLOT_FIGSIZE = (10, 6)

# ETTUS: Intelligent Communication Optimization for Federated Learning over Wireless Networks

## Overview

Wireless communication is one of the biggest bottlenecks in modern distributed machine learning systems. This project focuses on reducing communication overhead and improving transmission efficiency for Federated Learning over dynamic wireless networks.

The project addresses communication optimization through three complementary research problems:

- **Problem Statement 1**: Adaptive Fountain Code Parameter Optimization using Reinforcement Learning.
- **Problem Statement 2**: Adaptive TCP / Fountain Code Switching using Reinforcement Learning.
- **Problem Statement 3**: Federated Learning using Partial Model Updates with Sparse Reconstruction.

Together, these modules create a complete intelligent communication pipeline that optimizes:

- What to transmit (Problem Statement 3)
- How to transmit (Problem Statement 2)
- How to optimize the chosen protocol (Problem Statement 1)

## Research Motivation

Traditional Federated Learning requires every client to transmit complete model updates over the network. As the number of participating clients and model size increase, communication becomes the dominant bottleneck.

Similarly, wireless networks experience varying channel conditions where a single transmission protocol is not always optimal.

This project addresses both challenges by combining Federated Learning, Reinforcement Learning, and Fountain Codes into a unified communication framework.

## Research Problems

### Problem Statement 1 – Adaptive Fountain Code Optimization
**Objective**
Optimize Fountain Code transmission parameters according to current wireless channel conditions.

**Motivation**
Traditional Fountain Codes use fixed redundancy.
However, wireless channels constantly change due to:
- packet loss
- latency
- BLER
- bandwidth variation

Using a fixed redundancy ratio wastes bandwidth under good conditions and fails under poor conditions.

**Solution**
A PPO reinforcement learning agent continuously observes channel metrics and dynamically selects:
- Source block size (k)
- Redundancy ratio
to maximize decoding success while minimizing transmission overhead.

### Problem Statement 2 – Adaptive TCP / Fountain Code Controller
**Objective**
Instead of always using TCP or always using Fountain Codes, dynamically choose the best transport protocol.

**Motivation**
TCP performs well under stable networks.
Fountain Codes perform better under lossy networks.
No single protocol is optimal under every condition.

**Solution**
A PPO-based controller observes:
- BLER
- RTT
- Packet Loss
- Bandwidth
- Queue statistics
and decides **TCP** or **Fountain Code** for every transmission.

### Problem Statement 3 – Communication-Efficient Federated Learning
**Objective**
Reduce communication cost by transmitting only partial model updates.

**Motivation**
Federated Learning requires repeated communication between clients and the server.
Instead of transmitting complete model updates, only the most important parameters are transmitted.

**Solution**
The implementation includes:
- Top-K Gradient Sparsification
- Sparse Compression
- Error Feedback
- FedAvg Aggregation
- Multi-run Evaluation

## Overall System Architecture

```text
                 Federated Learning

              Global Model (Server)
                      │
                      ▼
          Broadcast Model to Clients
                      │
──────────────────────────────────────────────

               Client Side

      Local Training on Client Dataset
                      │
                      ▼
          Compute Model Update (ΔW)
                      │
                      ▼
          Top-K Update Selection
                      │
                      ▼
           Sparse Compression
                      │
                      ▼
              Sparse Payload

──────────────────────────────────────────────

           Problem Statement 2

     PPO Controller observes network

             TCP      Fountain Code
                │          │
                └────┬─────┘
                     ▼

──────────────────────────────────────────────

           Problem Statement 1

    If Fountain Code selected

      PPO selects

      • Source Block Size (k)

      • Redundancy Ratio

──────────────────────────────────────────────

             Wireless Channel

──────────────────────────────────────────────

                Server Side

         Receive Sparse Payload
                    │
                    ▼
           Sparse Reconstruction
                    │
                    ▼
             Error Feedback
                    │
                    ▼
                FedAvg
                    │
                    ▼
          Updated Global Model
```

## Project Structure
```text
ETTUS-data-collection-main/

├── FC/
│   ├── fountain_code.py
│   ├── fc_environment.py
│   ├── fc_trainer.py
│   ├── fc_evaluator.py
│
├── Controller/
│   ├── controller_environment.py
│   ├── controller_trainer.py
│   ├── controller_evaluator.py
│   ├── tcp_adapter.py
│   ├── fc_adapter.py
│
├── Federated/
│   ├── model/
│   ├── client/
│   ├── server/
│   ├── simulation/
│   ├── evaluation/
│
├── RL/
│
├── aioquic/
│
└── datasets/
```

## Workflow

**Problem Statement 1**
Network Dataset → FC Environment → PPO Training → Adaptive Fountain Code Parameters → Evaluation

**Problem Statement 2**
Network Metrics → Controller Environment → PPO Controller → TCP or Fountain Code → Performance Evaluation

**Problem Statement 3**
Global Model → Local Training → Top-K Selection → Sparse Compression → Controller → TCP / Fountain Code → Server Reconstruction → FedAvg → Next Communication Round

## Technologies Used

- **Programming:** Python
- **Machine Learning:** PyTorch, Stable-Baselines3
- **Reinforcement Learning:** PPO (Proximal Policy Optimization)
- **Federated Learning:** FedAvg, Top-K Sparsification, Error Feedback
- **Networking:** Fountain Codes, TCP, AIOQUIC
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib

## Evaluation

The project evaluates:

**Problem Statement 1**
- Decode Success Rate
- Transmission Overhead
- Packet Loss
- Reward
- Latency

**Problem Statement 2**
- TCP Usage
- FC Usage
- Controller Reward
- Switching Frequency
- Delivery Success

**Problem Statement 3**
- Global Accuracy
- Training Loss
- Compression Ratio
- Communication Cost
- Reconstruction Error
- Bandwidth Reduction
- Model Size Reduction

## Running the Project

**Train Fountain Code Agent**
```bash
python -m FC.fc_trainer
```

**Evaluate Fountain Code Agent**
```bash
python -m FC.fc_evaluator
```

**Train Controller**
```bash
python -m Controller.controller_trainer
```

**Evaluate Controller**
```bash
python -m Controller.controller_evaluator
```

**Run Federated Learning**
```bash
python -m Federated.simulation.fl_round_runner
```

**Evaluate Federated Learning**
```bash
python -m Federated.evaluation.evaluator
```

## Research Contributions
- Adaptive Fountain Code optimization using PPO.
- Adaptive protocol switching between TCP and Fountain Codes.
- Communication-efficient Federated Learning using sparse model updates.
- Integration of Reinforcement Learning with wireless communication protocols.
- Modular architecture enabling independent evaluation of communication and learning components.

## Future Work
- RL-based adaptive Top-K selection.
- Quantized model updates.
- Asynchronous Federated Learning.
- Real-time deployment on Software Defined Radios (SDRs).
- Multi-agent reinforcement learning for protocol coordination.

## References
- McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data (2017).
- Stich et al., Sparsified SGD with Memory (2018).
- Karimireddy et al., Error Feedback Fixes SignSGD (2019).
- Schulman et al., Proximal Policy Optimization Algorithms (2017).
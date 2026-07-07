# Federated Learning with Top-K Gradient Compression

This repository implements a lightweight, fully deterministic, research-ready Federated Learning (FL) pipeline simulating the `FedAvg` algorithm over a Non-IID MNIST distribution. It features a complete simulation of **Top-K gradient compression** with **Error Feedback** to reduce communication overhead.

## 1. Overview
Federated Learning enables collaborative training over decentralized edge devices while keeping data strictly local. However, transmitting massive neural network updates across slow uplink connections severely limits scalability. This module addresses the communication bottleneck by implementing sparse Top-K gradient compression. 

## 2. Research Objective
The core objective is to reduce uplink communication costs by approximately 90% per round without permanently degrading the final model accuracy. This is achieved by transmitting only the largest magnitude weight changes and accumulating the skipped (residual) updates locally using an Error Feedback store.

## 3. Architecture Diagram
```mermaid
graph TD
    subgraph "Server Environment"
        GM[Global Model (CNN)]
        AGG[Aggregator (FedAvg)]
        REC[Reconstructor]
        EF_S[Error Feedback Store]
    end

    subgraph "Transport Bridge"
        PAY[SparsePayload (16 + 12K Bytes)]
    end

    subgraph "Client Environment (k)"
        LM[Local Trainer]
        SEL[Update Selector (Top-K)]
        COMP[Compressor]
    end

    GM -->|Broadcast W_t| LM
    LM -->|Local SGD| LM
    LM -->|W_local - W_t| SEL
    SEL -->|Top-K Sparse Indices/Values| COMP
    COMP -->|Serialize| PAY
    PAY -->|Transmit Uplink| REC
    REC -->|Fetch previous residuals| EF_S
    EF_S -->|Apply residuals + Scatter| REC
    REC -->|Dense Update| AGG
    AGG -->|Average| GM
```
*(Note: The global model implemented is an MLP, but for consistency purposes we represent the logic above. The architecture is officially a 4-layer fully connected MLP with 109,386 parameters.)*

## 4. Folder Structure
```text
Federated/
├── config.py                 # Global hyperparameters and seeding logic
├── model/
│   ├── fl_model.py           # Deep neural network (MLP for MNIST, 109,386 params)
│   └── model_utils.py        # Vector mapping and checkpointing
├── client/
│   ├── local_trainer.py      # Local epochs on client data
│   ├── update_selector.py    # Top-K sparse magnitude selection
│   └── compressor.py         # Serialization of sparse tensors
├── server/
│   ├── error_feedback_store.py # Client-specific residual tracking
│   ├── reconstructor.py      # O(1) alloc sparse-to-dense scattering
│   └── aggregator.py         # In-place FedAvg (weighted/unweighted)
├── transport_bridge/
│   └── payload.py            # Dataclasses mimicking network transmission
├── simulation/
│   ├── data_partitioner.py   # Pathological Non-IID (McMahan) Distribution
│   └── fl_round_runner.py    # Sequential orchestration and step-tracking
└── evaluation/
    ├── evaluator.py          # Auto-runner for statistical modes
    ├── metrics.py            # Means, CIs, and comparative math
    └── plot_generator.py     # Matplotlib publication generators
```

## 5. Federated Learning Pipeline
The training follows the standard FedAvg paradigm:
1. Initialize a global MLP model with 109,386 parameters.
2. Partition the MNIST training set across 100 clients using pathological non-IID shards (McMahan et al., 2017).
3. Sample a fraction of clients per round.
4. Clients download global weights, train locally, and compute the weight delta.

## 6. Reconstruction Pipeline
The `Reconstructor` avoids memory reallocation by maintaining a persistent pool of zero-initialized $P$-dimensional dense buffers. For an incoming `SparsePayload`, it invokes NumPy's advanced indexing setter to map the $K$ values directly into the buffer in $O(K)$ time. If Error Feedback is active, previous residuals are dynamically factored into the reconstruction.

## 7. Aggregation Pipeline
Uses a unified in-place `np.add(out=accumulator)` buffer to sum updates across all participating clients. It performs weighted aggregation relative to dataset sizes ($n_i$). The logic implements NaN/Inf guards to prevent silent state corruption and skips malformed payloads automatically.

## 8. Training Command
To execute a single custom FL run programmatically:
```python
python -m Federated.simulation.fl_round_runner
```

## 9. Evaluation Command
To execute the full statistical evaluation suite across Full FL, Top-K, and Top-K + Error Feedback modes:
```python
python -m Federated.evaluation.evaluator
```

## 10. Configuration
All hyperparameters (batch sizes, learning rates, epochs, dropout, top-K ratios) are strictly defined in `Federated/config.py`.

## 11. Generated Outputs
Evaluation results are exported into `Federated/results/`.
- **Checkpoints**: `results/checkpoints/best_model.pt`

### 12. Graphs
Generated via `plot_generator.py` (300 DPI, tight layout, grid, legend):
1. Accuracy vs Communication Round
2. Training Loss vs Round
3. Validation Loss vs Round
4. Communication Cost vs Round
5. Compression Ratio vs Round
6. Reconstruction Error vs Round
7. Aggregation Time vs Round
8. Round Execution Time vs Round
9. Communication Savings (Cumulative)
10. Accuracy vs Communication Cost
11. Bandwidth Saved (Bar Chart)
12. Model Size Reduction (Bar Chart)
13. Client Participation per Round
14. Compression Ratio Distribution

### 13. CSV Files
- `experiment_config.json`: Hardware timestamp, RNG seed, and setup constants.
- `summary_metrics.csv`: Final comparative table of Accuracy, MB Transmitted, Bytes Saved, etc.
- `summary_metrics.md`: Identical to above but formatted for GitHub rendering.

## 14. Computational Complexity
- **Local SGD**: $\mathcal{O}(E \cdot B \cdot P)$ per client.
- **Top-K Sort**: $\mathcal{O}(P + K \log K)$ using NumPy `argpartition`.
- **Reconstruction**: $\mathcal{O}(K)$ per client (assignment map).
- **Aggregation**: $\mathcal{O}(N \cdot P)$.

## 15. Communication Complexity
- **Downlink (Uncompressed)**: $4P$ bytes ($437,544$ bytes).
- **Uplink (Compressed)**: $16 + 12K$ bytes ($131,272$ bytes for $K=10\%$).
- For Top-K = 10%, the theoretical uplink communication reduction approaches approximately 70%, excluding fixed protocol/header overhead.

## 16. Memory Complexity
- **Clients**: $\mathcal{O}(P)$ to maintain the weight delta.
- **Server**: $\mathcal{O}(N \cdot P)$ strictly pre-allocated across `ErrorFeedbackStore` and `Reconstructor` pools to prevent GC spikes.

## 17. Limitations
- Downlink transmissions are currently sent fully dense without decompression architectures on the edge nodes.
- Execution simulates sequential clients synchronously; true distributed execution is omitted to focus on mathematical gradient algorithms.

## 18. Future Work
- Adaptive Top-K sparsity (adjusting the threshold dynamically based on gradient norms).
- Combining Top-K with INT8 quantization for a further $4\times$ multiplier on uplink bandwidth savings.

## 19. References
- **McMahan et al. (2017)**: *Communication-Efficient Learning of Deep Networks from Decentralized Data.* (Influenced the core FedAvg architecture and the Pathological Non-IID dataset partitioning logic).
- **Stich et al. (2018)**: *Local SGD converges fast and communicates little.* (Influenced the mathematical treatment of decentralized state differences).
- **Karimireddy et al. (2019)**: *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.* (Influenced the design of tracking accumulated local history against global drift, parallel to the Error Feedback architecture).

"""
Federated — Partial Update Federated Learning Module
======================================================
Problem Statement 3: Can we reconstruct the global model using only
partial model updates?

This module implements a complete Federated Learning simulation with
Top-K sparse update transmission, error feedback, and server-side
reconstruction — all layered on top of the existing FC/Controller stack.

Sub-modules:
    config              — All hyperparameters (no hardcoded constants)
    model               — Lightweight MLP (Input→128→64→Output)
    client              — Local trainer, Top-K selector, compressor
    server              — Reconstructor, FedAvg aggregator, error feedback
    transport_bridge    — Payload adapter + Controller reuse bridge
    simulation          — Data partitioner, FL round runner
    evaluation          — Metrics, evaluator, plot generator

Entry Points:
    python -m Federated.simulation.fl_round_runner
    python -m Federated.evaluation.evaluator
"""
